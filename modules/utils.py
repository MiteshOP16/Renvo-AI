import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

def initialize_session_state():
    """Initialize all session state variables"""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'original_dataset' not in st.session_state:
        st.session_state.original_dataset = None
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}
    if 'column_analysis' not in st.session_state:
        st.session_state.column_analysis = {}
    if 'cleaning_history' not in st.session_state:
        st.session_state.cleaning_history = {}
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []
    if 'redo_stack' not in st.session_state:
        st.session_state.redo_stack = []
    if 'ai_context' not in st.session_state:
        st.session_state.ai_context = {}
    if 'inter_column_violations' not in st.session_state:
        st.session_state.inter_column_violations = {}
    if 'weights_manager' not in st.session_state:
        st.session_state.weights_manager = None
    if 'cleaning_engine' not in st.session_state:
        from modules.cleaning_engine import DataCleaningEngine
        st.session_state.cleaning_engine = DataCleaningEngine()
    if 'data_analyzer' not in st.session_state:
        from modules.data_analyzer import ColumnAnalyzer
        st.session_state.data_analyzer = ColumnAnalyzer()
    if 'report_generator' not in st.session_state:
        from modules.report_generator import ReportGenerator
        st.session_state.report_generator = ReportGenerator()
    if 'anomaly_detector' not in st.session_state:
        from modules.anomaly_detector import AnomalyDetector
        st.session_state.anomaly_detector = AnomalyDetector()
    if 'anomaly_results' not in st.session_state:
        st.session_state.anomaly_results = {}
    if 'anomaly_last_updated' not in st.session_state:
        st.session_state.anomaly_last_updated = None

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Enhanced column type detection beyond basic pandas dtypes"""
    column_types = {}
    
    for col in df.columns:
        series = df[col].dropna()
        
        if len(series) == 0:
            column_types[col] = 'empty'
            continue
            
        # Check for binary columns
        unique_vals = series.unique()
        if len(unique_vals) == 2:
            column_types[col] = 'binary'
            continue
            
        # Check for categorical with low cardinality
        if series.dtype == 'object':
            if len(unique_vals) / len(series) < 0.1 and len(unique_vals) < 20:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
            continue
            
        # Numeric types
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually ordinal (integers with reasonable range)
            if pd.api.types.is_integer_dtype(series):
                if len(unique_vals) < 10 and series.min() >= 0:
                    column_types[col] = 'ordinal'
                else:
                    column_types[col] = 'integer'
            else:
                column_types[col] = 'continuous'
            continue
            
        # Date/time
        if pd.api.types.is_datetime64_any_dtype(series):
            column_types[col] = 'datetime'
            continue
            
        column_types[col] = 'unknown'
    
    return column_types

def calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculate basic statistics for each column"""
    stats = {}
    
    for col in df.columns:
        series = df[col]
        col_stats = {
            'count': len(series),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(series):
            col_stats.update({
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75)
            })
        
        stats[col] = col_stats
    
    return stats

def save_cleaning_operation(operation: Dict[str, Any]):
    """Save a cleaning operation to the history"""
    timestamp = datetime.now().isoformat()
    operation['timestamp'] = timestamp
    
    if 'cleaning_history' not in st.session_state:
        st.session_state.cleaning_history = {}
    
    column = operation.get('column')
    if column not in st.session_state.cleaning_history:
        st.session_state.cleaning_history[column] = []
    
    st.session_state.cleaning_history[column].append(operation)

def create_backup():
    """Create a backup of current dataset state for undo functionality"""
    if st.session_state.dataset is not None:
        backup = {
            'dataset': st.session_state.dataset.copy(),
            'column_analysis': st.session_state.column_analysis.copy(),
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.undo_stack.append(backup)
        
        # Limit undo stack size
        if len(st.session_state.undo_stack) > 20:
            st.session_state.undo_stack.pop(0)
        
        # Clear redo stack when new operation is performed
        st.session_state.redo_stack.clear()

def undo_last_operation():
    """Undo the last cleaning operation"""
    if st.session_state.undo_stack:
        # Move current state to redo stack
        current_state = {
            'dataset': st.session_state.dataset.copy(),
            'column_analysis': st.session_state.column_analysis.copy(),
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.redo_stack.append(current_state)
        
        # Restore previous state
        previous_state = st.session_state.undo_stack.pop()
        st.session_state.dataset = previous_state['dataset']
        st.session_state.column_analysis = previous_state['column_analysis']
        
        return True
    return False

def redo_last_operation():
    """Redo the last undone operation"""
    if st.session_state.redo_stack:
        # Move current state to undo stack
        current_state = {
            'dataset': st.session_state.dataset.copy(),
            'column_analysis': st.session_state.column_analysis.copy(),
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.undo_stack.append(current_state)
        
        # Restore next state
        next_state = st.session_state.redo_stack.pop()
        st.session_state.dataset = next_state['dataset']
        st.session_state.column_analysis = next_state['column_analysis']
        
        return True
    return False

def export_configuration() -> str:
    """Export current configuration as JSON"""
    config = {
        'column_types': st.session_state.get('column_types', {}),
        'cleaning_history': st.session_state.get('cleaning_history', {}),
        'timestamp': datetime.now().isoformat()
    }
    return json.dumps(config, indent=2)

def import_configuration(config_json: str) -> bool:
    """Import configuration from JSON"""
    try:
        config = json.loads(config_json)
        st.session_state.column_types = config.get('column_types', {})
        st.session_state.cleaning_history = config.get('cleaning_history', {})
        return True
    except Exception as e:
        st.error(f"Error importing configuration: {str(e)}")
        return False

def format_number(value: Any) -> str:
    """Format numbers for display"""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1000000:
            return f"{value/1000000:.2f}M"
        elif abs(value) >= 1000:
            return f"{value/1000:.2f}K"
        elif isinstance(value, float):
            return f"{value:.3f}"
        else:
            return str(value)
    return str(value)

def get_column_summary(df: pd.DataFrame, column: str) -> str:
    """Generate a summary description of a column"""
    if column not in df.columns:
        return "Column not found"
    
    series = df[column]
    total_count = len(series)
    missing_count = series.isnull().sum()
    unique_count = series.nunique()
    
    summary = f"Column '{column}' has {total_count:,} total values"
    
    if missing_count > 0:
        summary += f", {missing_count:,} missing ({missing_count/total_count*100:.1f}%)"
    
    summary += f", {unique_count:,} unique values"
    
    if pd.api.types.is_numeric_dtype(series):
        summary += f". Range: {series.min():.2f} to {series.max():.2f}"
    
    return summary
