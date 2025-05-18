#!/usr/bin/env python3
"""
Test runner for database layer implementation tests.
Executes all test suites and generates a summary report.
"""

import unittest
import sys
import time
import os
from contextlib import contextmanager

# Adjust path to import test modules directly
current_dir = os.path.dirname(os.path.abspath(__file__))
test_files = [f[:-3] for f in os.listdir(current_dir) 
              if f.startswith('test_') and f.endswith('.py')]

def load_test_modules():
    """Dynamically import test modules from current directory"""
    modules = []
    sys.path.insert(0, current_dir)
    for test_file in test_files:
        if test_file != 'run_tests':
            try:
                modules.append(__import__(test_file))
            except ImportError as e:
                print(f"Error importing {test_file}: {e}")
    return modules

class TestResult(unittest.TestResult):
    """Custom test result class with timing information"""
    def __init__(self):
        super().__init__()
        self.test_times = {}
        self._current_test = None
        self._start_time = None

    def startTest(self, test):
        self._current_test = test
        self._start_time = time.time()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.time() - self._start_time
        self.test_times[test.id()] = elapsed
        super().stopTest(test)

def run_test_suite(suite_name: str, test_suite: unittest.TestSuite) -> TestResult:
    """Run a test suite and return results"""
    print(f"\nRunning {suite_name}...")
    result = TestResult()
    test_suite.run(result)
    return result

def print_summary(results: dict):
    """Print test execution summary"""
    print("\n" + "="*60)
    print("Test Execution Summary")
    print("="*60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_time = 0.0
    
    for suite_name, result in results.items():
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        suite_time = sum(result.test_times.values())
        
        print(f"\n{suite_name}:")
        print(f"  Tests run: {tests_run}")
        print(f"  Failures: {failures}")
        print(f"  Errors: {errors}")
        print(f"  Time: {suite_time:.2f}s")
        
        if failures > 0 or errors > 0:
            print("\n  Failed Tests:")
            for failure in result.failures:
                print(f"    - {failure[0].id()}")
            for error in result.errors:
                print(f"    - {error[0].id()}")
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_time += suite_time
    
    print("\n" + "-"*60)
    print(f"Total Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Total failures: {total_failures}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total time: {total_time:.2f}s")
    print("="*60 + "\n")
    
    return total_failures + total_errors

def main():
    """Main test runner entry point"""
    # Add parent directory to path for importing database package
    parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
    sys.path.insert(0, parent_dir)
    
    # Load test modules
    test_modules = load_test_modules()
    
    # Run tests
    test_suites = {}
    for module in test_modules:
        name = module.__name__.replace('test_', '').title()
        test_suites[f"{name} Tests"] = unittest.TestLoader().loadTestsFromModule(module)
    
    results = {}
    for suite_name, suite in test_suites.items():
        results[suite_name] = run_test_suite(suite_name, suite)
    
    return print_summary(results)

if __name__ == '__main__':
    sys.exit(main())