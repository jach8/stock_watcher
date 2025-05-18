import unittest
from bin.database.core import QueryBuilder, SQLQuery

class TestQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = QueryBuilder()

    def test_select_simple(self):
        """Test basic SELECT query construction"""
        query = self.builder.select(['id', 'name']).from_table('users').build()
        self.assertEqual(query.text.strip(), "SELECT [id], [name] FROM [users]")
        self.assertEqual(query.params, {})

    def test_where_clause(self):
        """Test WHERE clause with parameter binding"""
        query = (self.builder
                .select(['id', 'name'])
                .from_table('users')
                .where({'age': 25, 'active': True})
                .build())
        
        # Split query parts to check structure
        text = query.text.strip()
        self.assertIn("SELECT [id], [name]", text)
        self.assertIn("FROM [users]", text)
        self.assertIn("WHERE", text)
        self.assertIn("[age] = :p_", text)
        self.assertIn("[active] = :p_", text)
        
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
        
        self.assertEqual(query.text.strip(), "SELECT [id] FROM [users] ORDER BY [name] DESC")

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
        text = query.text.strip()
        self.assertIn(f"SELECT [{expected_column}]", text)
        self.assertIn(f"FROM [{expected_table}]", text)
        
        # Verify no SQL injection characters remain
        self.assertNotIn(';', text)
        self.assertNotIn('--', text)
        
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
        
        # Check the complete query structure
        text = query.text.strip()
        self.assertIn("SELECT [id], [name], [email]", text)
        self.assertIn("FROM [users]", text)
        self.assertIn("WHERE", text)
        self.assertIn("[age] = :p_", text)
        self.assertIn("[status] = :p_", text)
        self.assertIn("ORDER BY [created_at] DESC", text)
        
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