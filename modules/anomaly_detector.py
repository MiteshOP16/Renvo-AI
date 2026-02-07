"""
Anomaly Detection Module
Detects data type mismatches and formatting anomalies in dataset columns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import re


class AnomalyDetector:
    """Detects type mismatches and formatting anomalies in dataset columns"""
    
    def __init__(self):
        self.anomaly_cache = {}
    
    def detect_column_anomalies(self, df: pd.DataFrame, column: str, 
                                expected_type: str) -> Dict[str, Any]:
        """
        Detect anomalies in a specific column based on expected type
        
        Args:
            df: DataFrame containing the data
            column: Name of the column to check
            expected_type: Expected type (integer, continuous, binary, categorical, text, datetime, etc.)
        
        Returns:
            Dictionary containing anomaly information
        """
        series = df[column].copy()
        anomalies = {
            'column': column,
            'expected_type': expected_type,
            'total_values': len(series),
            'null_values': series.isnull().sum(),
            'anomalies': [],
            'anomaly_count': 0,
            'anomaly_percentage': 0.0,
            'summary': ''
        }
        
        # Skip null values - they are handled separately
        non_null_mask = series.notnull()
        non_null_series = series[non_null_mask]
        
        if len(non_null_series) == 0:
            anomalies['summary'] = 'All values are null'
            return anomalies
        
        # Detect anomalies based on expected type
        if expected_type in ['integer', 'continuous', 'ordinal']:
            anomaly_indices = self._detect_numeric_anomalies(non_null_series, expected_type)
        elif expected_type == 'binary':
            anomaly_indices = self._detect_binary_anomalies(non_null_series)
        elif expected_type == 'categorical':
            anomaly_indices = self._detect_categorical_anomalies(non_null_series)
        elif expected_type == 'datetime':
            anomaly_indices = self._detect_datetime_anomalies(non_null_series)
        elif expected_type == 'text':
            # Text columns are flexible, but we can detect encoding issues
            anomaly_indices = self._detect_text_anomalies(non_null_series)
        else:
            # Unknown type - minimal validation
            anomaly_indices = []
        
        # Build detailed anomaly list
        for idx in anomaly_indices:
            anomalies['anomalies'].append({
                'row_index': idx,
                'value': series.loc[idx],
                'reason': self._get_anomaly_reason(series.loc[idx], expected_type)
            })
        
        anomalies['anomaly_count'] = len(anomaly_indices)
        anomalies['anomaly_percentage'] = (len(anomaly_indices) / len(series)) * 100 if len(series) > 0 else 0
        
        # Generate summary
        if anomalies['anomaly_count'] == 0:
            anomalies['summary'] = f'✅ No anomalies detected - all values match expected type ({expected_type})'
        else:
            anomalies['summary'] = f'⚠️ Found {anomalies["anomaly_count"]} anomalies ({anomalies["anomaly_percentage"]:.2f}%) - values that don\'t match expected type ({expected_type})'
        
        return anomalies
    
    def _detect_numeric_anomalies(self, series: pd.Series, expected_type: str) -> List[int]:
        """Detect values that cannot be converted to numeric"""
        anomaly_indices = []
        
        for idx, value in series.items():
            try:
                # Try to convert to numeric
                if expected_type == 'integer':
                    # Check if it can be converted to integer
                    float_val = float(value)
                    if not np.isfinite(float_val):
                        anomaly_indices.append(idx)
                    elif float_val != int(float_val):
                        # Has decimal part - not a valid integer
                        anomaly_indices.append(idx)
                else:
                    # Continuous or ordinal - just needs to be numeric
                    float_val = float(value)
                    if not np.isfinite(float_val):
                        anomaly_indices.append(idx)
            except (ValueError, TypeError):
                # Cannot convert to numeric
                anomaly_indices.append(idx)
        
        return anomaly_indices
    
    def _detect_binary_anomalies(self, series: pd.Series) -> List[int]:
        """Detect values that don't fit binary pattern (should have exactly 2 unique values)"""
        unique_values = series.unique()
        
        # Binary columns should have exactly 2 unique values
        if len(unique_values) > 2:
            # Find which values are anomalous (least frequent ones)
            value_counts = series.value_counts()
            top_2_values = set(value_counts.head(2).index)
            
            anomaly_indices = []
            for idx, value in series.items():
                if value not in top_2_values:
                    anomaly_indices.append(idx)
            
            return anomaly_indices
        
        return []
    
    def _detect_categorical_anomalies(self, series: pd.Series) -> List[int]:
        """Detect potential anomalies in categorical data (encoding issues, inconsistent formatting)"""
        anomaly_indices = []
        
        # Check for encoding issues or special characters that might indicate data corruption
        for idx, value in series.items():
            value_str = str(value)
            
            # Check for encoding issues
            if any(ord(char) > 127 and ord(char) not in range(128, 256) for char in value_str):
                # Potential encoding issue (non-ASCII, non-Latin1)
                anomaly_indices.append(idx)
            
            # Check for excessive whitespace or formatting issues
            elif len(value_str) != len(value_str.strip()):
                # Leading/trailing whitespace might indicate data quality issue
                anomaly_indices.append(idx)
            
            # Check for mixed case inconsistency if most values follow a pattern
            # (This is more of a quality issue than anomaly, so we're lenient)
        
        return anomaly_indices
    
    def _detect_datetime_anomalies(self, series: pd.Series) -> List[int]:
        """Detect values that cannot be parsed as datetime"""
        anomaly_indices = []
        
        for idx, value in series.items():
            try:
                pd.to_datetime(value)
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                anomaly_indices.append(idx)
        
        return anomaly_indices
    
    def _detect_text_anomalies(self, series: pd.Series) -> List[int]:
        """Detect encoding or formatting issues in text columns"""
        anomaly_indices = []
        
        for idx, value in series.items():
            value_str = str(value)
            
            # Check for severe encoding issues or control characters
            if any(ord(char) < 32 and char not in '\n\r\t' for char in value_str):
                # Control characters (except newline, carriage return, tab)
                anomaly_indices.append(idx)
            
            # Check for replacement characters indicating encoding issues
            elif '\ufffd' in value_str:  # Unicode replacement character
                anomaly_indices.append(idx)
        
        return anomaly_indices
    
    def _get_anomaly_reason(self, value: Any, expected_type: str) -> str:
        """Generate human-readable reason for why a value is anomalous"""
        value_str = str(value)
        
        if expected_type in ['integer', 'continuous', 'ordinal']:
            try:
                float(value)
                if expected_type == 'integer':
                    return f'Contains decimal - expected integer'
                return f'Invalid numeric format'
            except (ValueError, TypeError):
                return f'Cannot convert "{value_str}" to numeric - expected {expected_type}'
        
        elif expected_type == 'binary':
            return f'Value "{value_str}" is not one of the two main categories'
        
        elif expected_type == 'categorical':
            if len(value_str) != len(value_str.strip()):
                return 'Has leading/trailing whitespace'
            return 'Potential encoding or formatting issue'
        
        elif expected_type == 'datetime':
            return f'Cannot parse "{value_str}" as datetime'
        
        elif expected_type == 'text':
            if '\ufffd' in value_str:
                return 'Contains encoding replacement characters'
            return 'Contains control characters or encoding issues'
        
        return 'Does not match expected format'
    
    def remove_anomalies(self, df: pd.DataFrame, column: str, 
                        anomaly_indices: List[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove anomalous cell values by setting them to null/NaN
        
        Args:
            df: DataFrame to clean
            column: Column name
            anomaly_indices: List of row indices where cell values should be set to null
        
        Returns:
            Tuple of (cleaned DataFrame, operation summary)
        """
        import numpy as np
        
        cleaned_df = df.copy()
        
        # Set anomalous cell values to NaN instead of removing entire rows
        for idx in anomaly_indices:
            cleaned_df.at[idx, column] = np.nan
        
        summary = {
            'operation': 'remove_anomalies',
            'column': column,
            'cells_nullified': len(anomaly_indices),
            'original_rows': len(df),
            'remaining_rows': len(cleaned_df),
            'timestamp': datetime.now().isoformat()
        }
        
        return cleaned_df, summary
    
    def replace_anomaly(self, df: pd.DataFrame, row_index: int, column: str, 
                       new_value: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Replace a specific anomalous value
        
        Args:
            df: DataFrame to modify
            row_index: Index of the row to replace
            column: Column name
            new_value: New value to set
        
        Returns:
            Tuple of (modified DataFrame, operation summary)
        """
        old_value = df.at[row_index, column]
        df.at[row_index, column] = new_value
        
        summary = {
            'operation': 'replace_anomaly',
            'column': column,
            'row_index': row_index,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        }
        
        return df, summary
    
    def batch_replace_anomalies(self, df: pd.DataFrame, column: str,
                                replacements: Dict[int, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Replace multiple anomalies at once
        
        Args:
            df: DataFrame to modify
            column: Column name
            replacements: Dictionary mapping row_index -> new_value
        
        Returns:
            Tuple of (modified DataFrame, operation summary)
        """
        modifications = []
        
        for row_index, new_value in replacements.items():
            old_value = df.at[row_index, column]
            df.at[row_index, column] = new_value
            modifications.append({
                'row_index': row_index,
                'old_value': old_value,
                'new_value': new_value
            })
        
        summary = {
            'operation': 'batch_replace_anomalies',
            'column': column,
            'replacements_count': len(replacements),
            'modifications': modifications,
            'timestamp': datetime.now().isoformat()
        }
        
        return df, summary
    
    def detect_all_anomalies(self, df: pd.DataFrame, 
                            column_types: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Detect anomalies in all columns based on their expected types
        
        Args:
            df: DataFrame to analyze
            column_types: Dictionary mapping column names to expected types
        
        Returns:
            Dictionary mapping column names to their anomaly information
        """
        all_anomalies = {}
        
        for column, expected_type in column_types.items():
            if column in df.columns:
                anomalies = self.detect_column_anomalies(df, column, expected_type)
                if anomalies['anomaly_count'] > 0:
                    all_anomalies[column] = anomalies
        
        return all_anomalies
