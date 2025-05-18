import unittest
from bin.database.core import QueryBuilder, SQLQuery

class TestQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = QueryBuilder()

    def test_select_simple(self):
        """Test basic SELECT query construction"""
        query = self.builder.select(['id', 'name']).from_table('users').build()
        self.assertEqual(query.text.strip(), "SELECT id, name FROM users")
        self.assertEqual(query.params, {})

    def test_where_clause(self):
        """Test WHERE clause with parameter binding"""
        query = (self.builder
                .select(['id', 'name'])
                .from_table('users')
                .where({'age': 25, 'active': True})
                .build())
        
        # Split query parts to check structure
        parts = [p.strip() for p in query.text.split() if p.strip()]
        self.assertEqual(parts[0], "SELECT")
        self.assertEqual(parts[1:3], ["id,", "name"])
        self.assertEqual(parts[3], "FROM")
        self.assertEqual(parts[4], "users")
        self.assertEqual(parts[5], "WHERE")
        
        # Check parameters were bound
        self.assertEqual(len(query.params), 2)
        self.assertTrue(any(v == 25 for v in query.params.values()))
        self.assertTrue(any(v is True for v in query.params.values()))

    def test_order_by(self):
        """Test ORDER BY clause construction"""
        query = (self.builder
                .select(['id'])
                .from_table('users')
                .order_by(['name'], desc=True)
                .build())
        
        self.assertEqual(query.text.strip(), "SELECT id FROM users ORDER BY name DESC")

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in identifiers"""
        malicious_table = "users; DROP TABLE students--"
        malicious_column = "name; DELETE FROM users--"
        
        query = (self.builder
                .select([malicious_column])
                .from_table(malicious_table)
                .build())
        
        # Expected sanitized identifiers
        expected_table = "usersDROPTABLEstudents"
        expected_column = "nameDELETEFROMusers"
        
        # Check query structure
        parts = [p.strip() for p in query.text.split() if p.strip()]
        self.assertEqual(parts[0], "SELECT")
        self.assertTrue(parts[1].startswith(expected_column))
        self.assertEqual(parts[2], "FROM")
        self.assertTrue(parts[3].startswith(expected_table))
        
        # Verify no SQL injection characters remain
        self.assertNotIn(';', query.text)
        self.assertNotIn('--', query.text)
        
    def test_complex_query(self):
        """Test construction of a more complex query"""
        query = (self.builder
                .select(['id', 'name', 'email'])
                .from_table('users')
                .where({
                    'age': 25,
                    'status': 'active'
                })
                .order_by(['created_at'], desc=True)
                .build())
        
        # Split and check each part of the query
        parts = [p.strip() for p in query.text.split() if p.strip()]
        
        # Check SELECT clause
        self.assertEqual(parts[0], "SELECT")
        select_cols = set(''.join(parts[1:4]).split(','))
        self.assertEqual(select_cols, {'id', 'name', 'email'})
        
        # Check FROM clause
        self.assertEqual(parts[4], "FROM")
        self.assertEqual(parts[5], "users")
        
        # Check WHERE clause
        self.assertEqual(parts[6], "WHERE")
        
        # Check ORDER BY clause
        order_idx = parts.index("ORDER")
        self.assertEqual(parts[order_idx:], ["ORDER", "BY", "created_at", "DESC"])
        
        # Check parameters
        self.assertEqual(len(query.params), 2)
        param_values = set(query.params.values())
        self.assertIn(25, param_values)
        self.assertIn('active', param_values)

    def test_identifier_sanitization(self):
        """Test sanitization of different identifier patterns"""
        cases = [
            ("normal_column", "normal_column"),
            ("table.column", "table.column"),
            ("malicious';--", "malicious"),
            ("bad-name", "badname"),
            ("spaces not allowed", "spacesnotallowed")
        ]
        
        for input_id, expected in cases:
            sanitized = self.builder._sanitize_identifier(input_id)
            self.assertEqual(sanitized, expected)

if __name__ == '__main__':
    unittest.main()