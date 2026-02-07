"""
Central registry for all statistical tests with metadata and validation
"""
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class TestMetadata:
    """Metadata for a statistical test"""
    test_id: str
    name: str
    category: str  # 'parametric' or 'non_parametric'
    subcategory: str  # e.g., 'comparison', 'correlation', 'goodness_of_fit', etc.
    description: str
    assumptions: List[str]
    input_requirements: Dict[str, Any]  # e.g., {'numeric_cols': 2, 'categorical_cols': 0}
    validator: Callable  # Function to validate inputs
    executor: Callable  # Function to execute the test
    example_use_case: str
    
class TestRegistry:
    """Central registry for all statistical tests"""
    
    def __init__(self):
        self._tests: Dict[str, TestMetadata] = {}
        
    def register(self, metadata: TestMetadata):
        """Register a test"""
        self._tests[metadata.test_id] = metadata
        
    def get_test(self, test_id: str) -> Optional[TestMetadata]:
        """Get test metadata by ID"""
        return self._tests.get(test_id)
    
    def list_all_tests(self) -> List[TestMetadata]:
        """List all registered tests"""
        return list(self._tests.values())
    
    def get_by_category(self, category: str) -> List[TestMetadata]:
        """Get tests by category (parametric/non_parametric)"""
        return [t for t in self._tests.values() if t.category == category]
    
    def get_by_subcategory(self, subcategory: str) -> List[TestMetadata]:
        """Get tests by subcategory"""
        return [t for t in self._tests.values() if t.subcategory == subcategory]
    
    def validate_inputs(self, test_id: str, df: pd.DataFrame, **params) -> Dict[str, Any]:
        """Validate inputs for a test"""
        test = self.get_test(test_id)
        if not test:
            return {'valid': False, 'error': f'Test {test_id} not found'}
        
        try:
            return test.validator(df, **params)
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def execute_test(self, test_id: str, df: pd.DataFrame, **params) -> Dict[str, Any]:
        """Execute a test"""
        test = self.get_test(test_id)
        if not test:
            return {'error': f'Test {test_id} not found'}
        
        # Validate first
        validation = self.validate_inputs(test_id, df, **params)
        if not validation.get('valid', False):
            return {'error': validation.get('error', 'Validation failed')}
        
        # Execute
        try:
            return test.executor(df, **params)
        except Exception as e:
            return {'error': str(e)}
    
    def get_ai_metadata(self) -> str:
        """Get formatted metadata for AI consumption"""
        output = []
        for test in self._tests.values():
            output.append(f"""
Test ID: {test.test_id}
Name: {test.name}
Category: {test.category}
Type: {test.subcategory}
Description: {test.description}
Requirements: {test.input_requirements}
Use case: {test.example_use_case}
Assumptions: {', '.join(test.assumptions)}
---""")
        return '\n'.join(output)

# Global registry instance
TEST_REGISTRY = TestRegistry()
