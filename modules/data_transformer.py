import pandas as pd
import numpy as np
import json
import ast
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, time


class DataTransformer:
    
    SUPPORTED_DTYPES = [
        'integer', 'float', 'string', 'boolean', 'datetime', 
        'categorical', 'list', 'dictionary', 'date', 'time',
        'complex', 'timedelta', 'object'
    ]
    
    DATE_FORMATS = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y%m%d', '%d%m%Y', '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y'
    ]
    
    TIME_FORMATS = [
        '%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p', '%H%M%S', '%H%M'
    ]
    
    DATETIME_FORMATS = [
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
        '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
    ]
    
    def __init__(self):
        self.operation_history = []
    
    def detect_column_dtype(self, series: pd.Series) -> str:
        if series.dropna().empty:
            return 'object'
        
        sample = series.dropna()
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'
        
        if pd.api.types.is_integer_dtype(series):
            return 'integer'
        
        if pd.api.types.is_float_dtype(series):
            return 'float'
        
        if pd.api.types.is_categorical_dtype(series):
            return 'categorical'
        
        if series.dtype == 'object':
            first_valid = sample.iloc[0] if len(sample) > 0 else None
            
            if isinstance(first_valid, (list, np.ndarray)):
                return 'list'
            
            if isinstance(first_valid, dict):
                return 'dictionary'
            
            if isinstance(first_valid, str):
                if self._is_json_like(first_valid):
                    parsed = self._try_parse_json(first_valid)
                    if isinstance(parsed, list):
                        return 'list'
                    if isinstance(parsed, dict):
                        return 'dictionary'
                
                if self._is_datetime_string(first_valid):
                    return 'datetime'
            
            return 'string'
        
        return 'object'
    
    def _is_json_like(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        value = value.strip()
        return (value.startswith('[') and value.endswith(']')) or \
               (value.startswith('{') and value.endswith('}'))
    
    def _try_parse_json(self, value: str) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    
    def _is_datetime_string(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        for fmt in self.DATE_FORMATS + self.DATETIME_FORMATS + self.TIME_FORMATS:
            try:
                datetime.strptime(value.strip(), fmt)
                return True
            except ValueError:
                continue
        return False
    
    def _parse_datetime(self, value: str) -> Optional[datetime]:
        if not isinstance(value, str):
            return None
        for fmt in self.DATETIME_FORMATS + self.DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
        return None
    
    def _parse_date(self, value: str) -> Optional[date]:
        if not isinstance(value, str):
            return None
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
        return None
    
    def _parse_time(self, value: str) -> Optional[time]:
        if not isinstance(value, str):
            return None
        for fmt in self.TIME_FORMATS:
            try:
                return datetime.strptime(value.strip(), fmt).time()
            except ValueError:
                continue
        return None
    
    def merge_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        separators: Union[str, List[str]],
        new_column_name: str,
        handle_missing: str = 'skip',
        output_format: Optional[str] = None,
        is_datetime_merge: bool = False,
        datetime_format: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        result_df = df.copy()
        
        if not all(col in df.columns for col in columns):
            missing_cols = [col for col in columns if col not in df.columns]
            return df, {'success': False, 'error': f'Columns not found: {missing_cols}'}
        
        if isinstance(separators, str):
            separators = [separators] * (len(columns) - 1)
        
        if len(separators) != len(columns) - 1:
            return df, {'success': False, 'error': 'Number of separators must be one less than columns'}
        
        try:
            if is_datetime_merge:
                merged = self._merge_datetime_columns(
                    result_df, columns, separators, datetime_format
                )
            else:
                merged = self._merge_regular_columns(
                    result_df, columns, separators, handle_missing
                )
            
            result_df[new_column_name] = merged
            
            operation_info = {
                'success': True,
                'operation': 'merge',
                'source_columns': columns,
                'new_column': new_column_name,
                'separators': separators,
                'rows_affected': len(merged.dropna()),
                'null_count': merged.isna().sum()
            }
            
            self.operation_history.append(operation_info)
            return result_df, operation_info
            
        except Exception as e:
            return df, {'success': False, 'error': str(e)}
    
    def _merge_regular_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        separators: List[str],
        handle_missing: str
    ) -> pd.Series:
        def merge_row(row):
            values = []
            for i, col in enumerate(columns):
                val = row[col]
                if pd.isna(val):
                    if handle_missing == 'skip':
                        continue
                    elif handle_missing == 'empty':
                        values.append('')
                    elif handle_missing == 'null_string':
                        values.append('NULL')
                    else:
                        return np.nan
                else:
                    values.append(str(val))
            
            if not values:
                return np.nan
            
            result = values[0]
            for i, val in enumerate(values[1:]):
                sep = separators[min(i, len(separators) - 1)]
                result = result + sep + val
            
            return result
        
        return df.apply(merge_row, axis=1)
    
    def _parse_datetime_value(self, val: Any) -> Optional[datetime]:
        if pd.isna(val):
            return None
        
        if isinstance(val, datetime):
            return val
        
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime()
        
        if hasattr(val, 'year') and hasattr(val, 'month') and hasattr(val, 'day'):
            try:
                return datetime(val.year, val.month, val.day, 
                               getattr(val, 'hour', 0), 
                               getattr(val, 'minute', 0), 
                               getattr(val, 'second', 0))
            except Exception:
                pass
        
        if isinstance(val, str):
            try:
                parsed = pd.to_datetime(val, errors='coerce')
                if pd.notna(parsed):
                    return parsed.to_pydatetime()
            except Exception:
                pass
        
        return None
    
    def _merge_datetime_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        separators: List[str],
        output_format: Optional[str]
    ) -> pd.Series:
        def merge_datetime_row(row):
            try:
                date_parts = {}
                time_parts = {}
                parsed_datetimes = []
                
                for col in columns:
                    val = row[col]
                    if pd.isna(val):
                        continue
                    
                    col_lower = col.lower()
                    
                    if 'year' in col_lower:
                        try:
                            date_parts['year'] = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                    elif 'month' in col_lower:
                        try:
                            date_parts['month'] = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                    elif 'day' in col_lower and 'weekday' not in col_lower:
                        try:
                            date_parts['day'] = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                    elif 'hour' in col_lower:
                        try:
                            time_parts['hour'] = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                    elif 'minute' in col_lower or 'min' in col_lower:
                        try:
                            time_parts['minute'] = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                    elif 'second' in col_lower or 'sec' in col_lower:
                        try:
                            time_parts['second'] = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                    elif 'date' in col_lower or 'datetime' in col_lower:
                        parsed_dt = self._parse_datetime_value(val)
                        if parsed_dt:
                            date_parts.update({
                                'year': parsed_dt.year,
                                'month': parsed_dt.month,
                                'day': parsed_dt.day
                            })
                            if parsed_dt.hour or parsed_dt.minute or parsed_dt.second:
                                time_parts.update({
                                    'hour': parsed_dt.hour,
                                    'minute': parsed_dt.minute,
                                    'second': parsed_dt.second
                                })
                            parsed_datetimes.append(parsed_dt)
                    elif 'time' in col_lower:
                        parsed_dt = self._parse_datetime_value(val)
                        if parsed_dt:
                            time_parts.update({
                                'hour': parsed_dt.hour,
                                'minute': parsed_dt.minute,
                                'second': parsed_dt.second
                            })
                        else:
                            parsed_time = self._parse_time(str(val).strip())
                            if parsed_time:
                                time_parts.update({
                                    'hour': parsed_time.hour,
                                    'minute': parsed_time.minute,
                                    'second': parsed_time.second
                                })
                    else:
                        parsed_dt = self._parse_datetime_value(val)
                        if parsed_dt:
                            parsed_datetimes.append(parsed_dt)
                
                if 'year' in date_parts and 'month' in date_parts and 'day' in date_parts:
                    dt = datetime(
                        year=date_parts['year'],
                        month=date_parts['month'],
                        day=date_parts['day'],
                        hour=time_parts.get('hour', 0),
                        minute=time_parts.get('minute', 0),
                        second=time_parts.get('second', 0)
                    )
                    
                    if output_format:
                        return dt.strftime(output_format)
                    return dt
                
                if len(parsed_datetimes) == 1:
                    dt = parsed_datetimes[0]
                    if time_parts and (time_parts.get('hour', 0) or time_parts.get('minute', 0) or time_parts.get('second', 0)):
                        dt = datetime(
                            year=dt.year,
                            month=dt.month,
                            day=dt.day,
                            hour=time_parts.get('hour', 0),
                            minute=time_parts.get('minute', 0),
                            second=time_parts.get('second', 0)
                        )
                    if output_format:
                        return dt.strftime(output_format)
                    return dt
                
                if len(parsed_datetimes) >= 2:
                    date_dt = parsed_datetimes[0]
                    time_dt = parsed_datetimes[1]
                    combined = datetime(
                        year=date_dt.year,
                        month=date_dt.month,
                        day=date_dt.day,
                        hour=time_dt.hour,
                        minute=time_dt.minute,
                        second=time_dt.second
                    )
                    if output_format:
                        return combined.strftime(output_format)
                    return combined
                
                return pd.NaT
                
            except Exception:
                return pd.NaT
        
        result_series = df.apply(merge_datetime_row, axis=1)
        try:
            return pd.to_datetime(result_series, errors='coerce')
        except Exception:
            return result_series
    
    def split_column(
        self,
        df: pd.DataFrame,
        column: str,
        separator: str,
        new_column_prefix: Optional[str] = None,
        max_splits: int = -1,
        is_datetime_split: bool = False,
        datetime_components: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        result_df = df.copy()
        
        if column not in df.columns:
            return df, {'success': False, 'error': f'Column not found: {column}'}
        
        prefix = new_column_prefix or f"{column}_part"
        
        try:
            if is_datetime_split:
                new_cols, col_names = self._split_datetime_column(
                    result_df[column], datetime_components or ['year', 'month', 'day']
                )
            else:
                new_cols, col_names = self._split_regular_column(
                    result_df[column], separator, prefix, max_splits
                )
            
            for col_name, col_data in zip(col_names, new_cols):
                result_df[col_name] = col_data
            
            operation_info = {
                'success': True,
                'operation': 'split',
                'source_column': column,
                'new_columns': col_names,
                'separator': separator,
                'rows_processed': len(result_df)
            }
            
            self.operation_history.append(operation_info)
            return result_df, operation_info
            
        except Exception as e:
            return df, {'success': False, 'error': str(e)}
    
    def _split_regular_column(
        self,
        series: pd.Series,
        separator: str,
        prefix: str,
        max_splits: int
    ) -> Tuple[List[pd.Series], List[str]]:
        if separator:
            split_df = series.astype(str).str.split(separator, expand=True, n=max_splits if max_splits > 0 else None)
        else:
            split_df = series.astype(str).str.split(expand=True)
        
        col_names = [f"{prefix}_{i+1}" for i in range(len(split_df.columns))]
        new_cols = [split_df[col] for col in split_df.columns]
        
        return new_cols, col_names
    
    def _split_datetime_column(
        self,
        series: pd.Series,
        components: List[str]
    ) -> Tuple[List[pd.Series], List[str]]:
        try:
            dt_series = pd.to_datetime(series, errors='coerce')
        except Exception:
            dt_series = series.apply(lambda x: self._parse_datetime(str(x)) if pd.notna(x) else None)
            dt_series = pd.to_datetime(dt_series, errors='coerce')
        
        new_cols = []
        col_names = []
        
        component_map = {
            'year': lambda dt: dt.dt.year,
            'month': lambda dt: dt.dt.month,
            'day': lambda dt: dt.dt.day,
            'hour': lambda dt: dt.dt.hour,
            'minute': lambda dt: dt.dt.minute,
            'second': lambda dt: dt.dt.second,
            'weekday': lambda dt: dt.dt.weekday,
            'week': lambda dt: dt.dt.isocalendar().week,
            'quarter': lambda dt: dt.dt.quarter,
            'dayofyear': lambda dt: dt.dt.dayofyear,
            'date': lambda dt: dt.dt.date,
            'time': lambda dt: dt.dt.time
        }
        
        for comp in components:
            if comp in component_map:
                new_cols.append(component_map[comp](dt_series))
                col_names.append(comp)
        
        return new_cols, col_names
    
    def detect_json_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        json_columns = []
        
        for col in df.columns:
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            
            json_count = 0
            sample_parsed = []
            all_keys = set()
            
            check_sample = sample.head(100)
            
            for val in check_sample:
                if isinstance(val, str) and self._is_json_like(val):
                    parsed = self._try_parse_json(val)
                    if isinstance(parsed, (dict, list)):
                        json_count += 1
                        sample_parsed.append(parsed)
                        
                        if isinstance(parsed, dict):
                            all_keys.update(parsed.keys())
                        elif isinstance(parsed, list) and parsed:
                            if isinstance(parsed[0], dict):
                                for item in parsed:
                                    if isinstance(item, dict):
                                        all_keys.update(item.keys())
                elif isinstance(val, (dict, list)):
                    json_count += 1
                    sample_parsed.append(val)
                    
                    if isinstance(val, dict):
                        all_keys.update(val.keys())
                    elif isinstance(val, list) and val:
                        if isinstance(val[0], dict):
                            for item in val:
                                if isinstance(item, dict):
                                    all_keys.update(item.keys())
            
            if json_count / len(check_sample) > 0.5:
                is_array = any(isinstance(p, list) for p in sample_parsed)
                is_nested = is_array and any(
                    isinstance(p, list) and p and isinstance(p[0], dict) 
                    for p in sample_parsed
                )
                
                json_columns.append({
                    'column': col,
                    'json_percentage': (json_count / len(check_sample)) * 100,
                    'keys': sorted(list(all_keys)),
                    'is_array': is_array,
                    'is_nested': is_nested,
                    'sample_count': len(sample_parsed)
                })
        
        return json_columns
    
    def expand_json_column(
        self,
        df: pd.DataFrame,
        column: str,
        keys_to_extract: List[str],
        explode_arrays: bool = False,
        prefix: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        result_df = df.copy()
        
        if column not in df.columns:
            return df, {'success': False, 'error': f'Column not found: {column}'}
        
        prefix = prefix or column
        
        try:
            def parse_and_extract(val):
                if pd.isna(val):
                    return None
                
                if isinstance(val, str):
                    parsed = self._try_parse_json(val)
                else:
                    parsed = val
                
                return parsed
            
            parsed_series = df[column].apply(parse_and_extract)
            
            if explode_arrays:
                def expand_array(val):
                    if isinstance(val, list):
                        return val
                    elif isinstance(val, dict):
                        return [val]
                    return [None]
                
                result_df['_temp_expanded'] = parsed_series.apply(expand_array)
                result_df = result_df.explode('_temp_expanded')
                parsed_series = result_df['_temp_expanded']
                result_df = result_df.drop(columns=['_temp_expanded'])
            
            for key in keys_to_extract:
                def extract_key(val, k=key):
                    if isinstance(val, dict):
                        return val.get(k)
                    elif isinstance(val, list) and val:
                        if isinstance(val[0], dict):
                            values = [item.get(k) for item in val if isinstance(item, dict)]
                            return values if values else None
                    return None
                
                result_df[f"{prefix}_{key}"] = parsed_series.apply(extract_key)
            
            new_columns = [f"{prefix}_{key}" for key in keys_to_extract]
            
            operation_info = {
                'success': True,
                'operation': 'expand_json',
                'source_column': column,
                'new_columns': new_columns,
                'keys_extracted': keys_to_extract,
                'exploded': explode_arrays,
                'rows_after': len(result_df)
            }
            
            self.operation_history.append(operation_info)
            return result_df, operation_info
            
        except Exception as e:
            return df, {'success': False, 'error': str(e)}
    
    def columns_to_json(
        self,
        df: pd.DataFrame,
        columns: List[str],
        new_column_name: str,
        group_by: Optional[str] = None,
        as_array: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        result_df = df.copy()
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return df, {'success': False, 'error': f'Columns not found: {missing_cols}'}
        
        try:
            if group_by and group_by in df.columns:
                def create_grouped_json(group):
                    records = group[columns].to_dict('records')
                    return json.dumps(records) if as_array else json.dumps(records[0] if records else {})
                
                grouped = result_df.groupby(group_by).apply(
                    lambda g: g[columns].to_dict('records')
                ).reset_index()
                grouped.columns = [group_by, new_column_name]
                
                result_df = result_df.merge(grouped, on=group_by, how='left')
                result_df[new_column_name] = result_df[new_column_name].apply(json.dumps)
            else:
                def row_to_json(row):
                    data = {col: row[col] if pd.notna(row[col]) else None for col in columns}
                    if as_array:
                        return json.dumps([data])
                    return json.dumps(data)
                
                result_df[new_column_name] = result_df.apply(row_to_json, axis=1)
            
            operation_info = {
                'success': True,
                'operation': 'columns_to_json',
                'source_columns': columns,
                'new_column': new_column_name,
                'grouped_by': group_by,
                'as_array': as_array
            }
            
            self.operation_history.append(operation_info)
            return result_df, operation_info
            
        except Exception as e:
            return df, {'success': False, 'error': str(e)}
    
    def convert_column_dtype(
        self,
        df: pd.DataFrame,
        column: str,
        target_dtype: str,
        datetime_format: Optional[str] = None,
        errors: str = 'coerce'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        result_df = df.copy()
        
        if column not in df.columns:
            return df, {'success': False, 'error': f'Column not found: {column}'}
        
        try:
            series = result_df[column]
            original_dtype = str(series.dtype)
            
            if target_dtype == 'integer':
                result_df[column] = pd.to_numeric(series, errors=errors).astype('Int64')
            
            elif target_dtype == 'float':
                result_df[column] = pd.to_numeric(series, errors=errors)
            
            elif target_dtype == 'string':
                result_df[column] = series.astype(str).replace('nan', pd.NA)
            
            elif target_dtype == 'boolean':
                def to_bool(val):
                    if pd.isna(val):
                        return pd.NA
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float)):
                        return bool(val)
                    if isinstance(val, str):
                        lower = val.lower().strip()
                        if lower in ['true', 'yes', '1', 't', 'y']:
                            return True
                        elif lower in ['false', 'no', '0', 'f', 'n']:
                            return False
                    return pd.NA
                result_df[column] = series.apply(to_bool)
            
            elif target_dtype == 'datetime':
                if datetime_format:
                    result_df[column] = pd.to_datetime(series, format=datetime_format, errors=errors)
                else:
                    result_df[column] = pd.to_datetime(series, errors=errors)
            
            elif target_dtype == 'date':
                dt_series = pd.to_datetime(series, errors=errors)
                result_df[column] = dt_series.dt.date
            
            elif target_dtype == 'time':
                dt_series = pd.to_datetime(series, errors=errors)
                result_df[column] = dt_series.dt.time
            
            elif target_dtype == 'categorical':
                result_df[column] = pd.Categorical(series)
            
            elif target_dtype == 'list':
                def to_list(val):
                    if pd.isna(val):
                        return None
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str):
                        parsed = self._try_parse_json(val)
                        if isinstance(parsed, list):
                            return parsed
                        return [val]
                    return [val]
                result_df[column] = series.apply(to_list)
            
            elif target_dtype == 'dictionary':
                def to_dict(val):
                    if pd.isna(val):
                        return None
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, str):
                        parsed = self._try_parse_json(val)
                        if isinstance(parsed, dict):
                            return parsed
                    return None
                result_df[column] = series.apply(to_dict)
            
            conversion_success = result_df[column].notna().sum()
            conversion_failed = result_df[column].isna().sum() - series.isna().sum()
            
            operation_info = {
                'success': True,
                'operation': 'convert_dtype',
                'column': column,
                'original_dtype': original_dtype,
                'target_dtype': target_dtype,
                'successful_conversions': int(conversion_success),
                'failed_conversions': int(max(0, conversion_failed))
            }
            
            self.operation_history.append(operation_info)
            return result_df, operation_info
            
        except Exception as e:
            return df, {'success': False, 'error': str(e)}
    
    def validate_merge_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        is_datetime: bool = False
    ) -> Dict[str, Any]:
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'dtype_info': {}
        }
        
        for col in columns:
            if col not in df.columns:
                validation['errors'].append(f"Column '{col}' not found")
                validation['valid'] = False
                continue
            
            dtype = self.detect_column_dtype(df[col])
            validation['dtype_info'][col] = dtype
            
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 50:
                validation['warnings'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
        
        if is_datetime:
            date_related = ['year', 'month', 'day', 'hour', 'minute', 'second', 'date', 'time']
            has_date_col = any(
                any(dt in col.lower() for dt in date_related) 
                for col in columns
            )
            if not has_date_col:
                validation['warnings'].append(
                    "Selected columns don't appear to contain date/time components. "
                    "Merge will use standard string concatenation."
                )
        
        return validation
    
    def validate_split_column(
        self,
        df: pd.DataFrame,
        column: str,
        separator: str
    ) -> Dict[str, Any]:
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'preview': [],
            'estimated_columns': 0
        }
        
        if column not in df.columns:
            validation['errors'].append(f"Column '{column}' not found")
            validation['valid'] = False
            return validation
        
        sample = df[column].dropna().head(10)
        
        if separator:
            split_counts = sample.astype(str).str.count(re.escape(separator)) + 1
            validation['estimated_columns'] = int(split_counts.max()) if len(split_counts) > 0 else 0
            
            if split_counts.nunique() > 1:
                validation['warnings'].append(
                    f"Inconsistent split counts detected (range: {split_counts.min()} to {split_counts.max()} parts)"
                )
            
            validation['preview'] = sample.astype(str).str.split(separator, expand=True).head(5).to_dict('records')
        else:
            validation['warnings'].append("No separator provided - will split on whitespace")
        
        return validation
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        return self.operation_history.copy()
    
    def clear_history(self):
        self.operation_history = []
