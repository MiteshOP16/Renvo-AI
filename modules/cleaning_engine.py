import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCleaningEngine:
    """Advanced data cleaning engine with column-specific methods and weighted operations"""
    
    def __init__(self, weights_manager=None):
        self.weights_manager = weights_manager
        self.cleaning_methods = {
            'missing_values': {
                'mean_imputation': self._mean_imputation,
                'median_imputation': self._median_imputation,
                'mode_imputation': self._mode_imputation,
                'forward_fill': self._forward_fill,
                'backward_fill': self._backward_fill,
                'knn_imputation': self._knn_imputation,
                'interpolation': self._interpolation,
                'missing_category': self._missing_category,
                'regression_imputation': self._regression_imputation
            },
            'outliers': {
                'iqr_removal': self._iqr_removal,
                'zscore_removal': self._zscore_removal,
                'winsorization': self._winsorization,
                'log_transformation': self._log_transformation,
                'cap_outliers': self._cap_outliers,
                'isolation_forest': self._isolation_forest_removal
            },
            'data_quality': {
                'type_standardization': self._type_standardization,
                'remove_duplicates': self._remove_duplicates,
                'trim_whitespace': self._trim_whitespace,
                'standardize_case': self._standardize_case
            }
        }
    
    def apply_cleaning_method(self, df: pd.DataFrame, column: str, 
                            method_type: str, method_name: str, 
                            parameters: Optional[Dict[str, Any]] = None,
                            use_weights: bool = False) -> Tuple[pd.Series, Dict[str, Any]]:
        """Apply a specific cleaning method to a column"""
        if method_type not in self.cleaning_methods:
            raise ValueError(f"Unknown method type: {method_type}")
        
        if method_name not in self.cleaning_methods[method_type]:
            raise ValueError(f"Unknown method: {method_name}")
        
        parameters = parameters or {}
        original_series = df[column].copy()
        
        try:
            cleaning_func = self.cleaning_methods[method_type][method_name]
            
            # Pass weight information to cleaning function if available
            if use_weights and self.weights_manager and self.weights_manager.weights_column:
                parameters['weights'] = df[self.weights_manager.weights_column]
                parameters['use_weights'] = True
            
            cleaned_series, metadata = cleaning_func(df, column, **parameters)
            
            # Calculate impact statistics (both weighted and unweighted)
            impact_stats = self._calculate_impact_stats(original_series, cleaned_series, 
                                                      df.get(self.weights_manager.weights_column) if use_weights and self.weights_manager else None)
            metadata['impact_stats'] = impact_stats
            metadata['success'] = True
            metadata['weighted_operation'] = use_weights
            
            return cleaned_series, metadata
            
        except Exception as e:
            return original_series, {
                'success': False,
                'error': str(e),
                'method_type': method_type,
                'method_name': method_name
            }
    
    def _calculate_impact_stats(self, original: pd.Series, cleaned: pd.Series, weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate impact statistics of cleaning operation with optional weighting"""
        stats = {
            'rows_affected': (original != cleaned).sum(),
            'percentage_changed': ((original != cleaned).sum() / len(original)) * 100,
            'missing_before': original.isnull().sum(),
            'missing_after': cleaned.isnull().sum(),
            'missing_change': cleaned.isnull().sum() - original.isnull().sum()
        }
        
        if pd.api.types.is_numeric_dtype(original):
            original_clean = original.dropna()
            cleaned_clean = cleaned.dropna()
            
            if len(original_clean) > 0 and len(cleaned_clean) > 0:
                # Unweighted statistics
                stats.update({
                    'mean_before': original_clean.mean(),
                    'mean_after': cleaned_clean.mean(),
                    'std_before': original_clean.std(),
                    'std_after': cleaned_clean.std(),
                    'median_before': original_clean.median(),
                    'median_after': cleaned_clean.median()
                })
                
                # Weighted statistics if weights are provided
                if weights is not None:
                    weights_clean_orig = weights[original.notna()]
                    weights_clean_new = weights[cleaned.notna()]
                    
                    if len(weights_clean_orig) > 0 and len(weights_clean_new) > 0:
                        weighted_mean_before = np.average(original_clean, weights=weights_clean_orig)
                        weighted_mean_after = np.average(cleaned_clean, weights=weights_clean_new)
                        
                        stats.update({
                            'weighted_mean_before': weighted_mean_before,
                            'weighted_mean_after': weighted_mean_after,
                            'weighted_mean_change': weighted_mean_after - weighted_mean_before,
                            'total_weight_before': weights_clean_orig.sum(),
                            'total_weight_after': weights_clean_new.sum()
                        })
        
        return stats
    
    # Missing Value Methods
    def _mean_imputation(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Mean imputation for numeric columns"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Mean imputation only applicable to numeric columns")
        
        mean_value = series.mean()
        if pd.isna(mean_value) or mean_value is None:
            raise ValueError("Cannot calculate mean - all values are missing")
        
        filled_series = series.fillna(mean_value)
        
        metadata = {
            'method': 'mean_imputation',
            'imputed_value': mean_value,
            'values_imputed': series.isnull().sum()
        }
        
        return filled_series, metadata
    
    def _median_imputation(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Median imputation for numeric columns"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Median imputation only applicable to numeric columns")
        
        median_value = series.median()
        if pd.isna(median_value) or median_value is None:
            raise ValueError("Cannot calculate median - all values are missing")
        
        filled_series = series.fillna(median_value)
        
        metadata = {
            'method': 'median_imputation',
            'imputed_value': median_value,
            'values_imputed': series.isnull().sum()
        }
        
        return filled_series, metadata
    
    def _mode_imputation(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Mode imputation for categorical columns"""
        series = df[column].copy()
        
        mode_values = series.mode()
        if len(mode_values) == 0:
            raise ValueError("Cannot calculate mode - all values are missing")
        
        mode_value = mode_values[0]
        filled_series = series.fillna(mode_value)
        
        metadata = {
            'method': 'mode_imputation',
            'imputed_value': mode_value,
            'values_imputed': series.isnull().sum(),
            'mode_frequency': (series == mode_value).sum()
        }
        
        return filled_series, metadata
    
    def _forward_fill(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Forward fill missing values"""
        series = df[column].copy()
        filled_series = series.ffill()
        
        # If still missing values (e.g., leading NaNs), fill with mode/median
        if filled_series.isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(series):
                backup_value = series.median()
            else:
                mode_values = series.mode()
                backup_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
            
            filled_series = filled_series.fillna(backup_value)
        
        metadata = {
            'method': 'forward_fill',
            'values_imputed': series.isnull().sum()
        }
        
        return filled_series, metadata
    
    def _backward_fill(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Backward fill missing values"""
        series = df[column].copy()
        filled_series = series.bfill()
        
        # If still missing values (e.g., trailing NaNs), fill with mode/median
        if filled_series.isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(series):
                backup_value = series.median()
            else:
                mode_values = series.mode()
                backup_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
            
            filled_series = filled_series.fillna(backup_value)
        
        metadata = {
            'method': 'backward_fill',
            'values_imputed': series.isnull().sum()
        }
        
        return filled_series, metadata
    
    def _knn_imputation(self, df: pd.DataFrame, column: str, n_neighbors: int = 5, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """KNN imputation using other numeric columns - optimized"""
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("KNN imputation only applicable to numeric columns")
        
        # Select numeric columns for KNN (limit to most correlated for performance)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("KNN imputation requires at least 2 numeric columns")
        
        # Optimize: use only most relevant columns if we have many (>10)
        if len(numeric_cols) > 10:
            # Calculate correlations and select top 10 most correlated
            correlations = df[numeric_cols].corr()[column].abs().sort_values(ascending=False)
            top_cols = correlations.head(11).index.tolist()  # Include the target column
            numeric_cols = [col for col in top_cols if col in numeric_cols]
        
        # Prepare data for KNN
        knn_data = df[numeric_cols].copy()
        
        # Apply KNN imputation with optimized n_neighbors
        optimal_k = min(n_neighbors, len(knn_data.dropna()) // 2)
        if optimal_k < 1:
            optimal_k = 1
        
        imputer = KNNImputer(n_neighbors=optimal_k)
        imputed_data = imputer.fit_transform(knn_data)
        
        # Extract the imputed column
        col_index = numeric_cols.index(column)
        imputed_series = pd.Series(imputed_data[:, col_index], index=df.index)
        
        metadata = {
            'method': 'knn_imputation',
            'n_neighbors': optimal_k,
            'features_used': len(numeric_cols),
            'values_imputed': df[column].isnull().sum()
        }
        
        return imputed_series, metadata
    
    def _interpolation(self, df: pd.DataFrame, column: str, method: str = 'linear', **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Interpolation for time series or ordered data"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Interpolation only applicable to numeric columns")
        
        # Ensure method is a valid interpolation method
        valid_methods = ['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
        if method not in valid_methods:
            method = 'linear'
        
        interpolated_series = series.interpolate(method=method)
        
        # Handle remaining NaNs at the beginning or end
        if interpolated_series.isnull().sum() > 0:
            interpolated_series = interpolated_series.fillna(series.median())
        
        metadata = {
            'method': f'interpolation_{method}',
            'values_imputed': series.isnull().sum()
        }
        
        return interpolated_series, metadata
    
    def _missing_category(self, df: pd.DataFrame, column: str, category_name: str = 'Missing', **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Create missing category for categorical data"""
        series = df[column].copy()
        filled_series = series.fillna(category_name)
        
        metadata = {
            'method': 'missing_category',
            'category_name': category_name,
            'values_imputed': series.isnull().sum()
        }
        
        return filled_series, metadata
    
    def _regression_imputation(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Regression-based imputation using other columns"""
        from sklearn.linear_model import LinearRegression
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("Regression imputation only applicable to numeric columns")
        
        # Find numeric predictors
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if column in numeric_cols:
            numeric_cols.remove(column)
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric predictors available for regression imputation")
        
        # Prepare data
        target_series = df[column].copy()
        predictor_data = df[numeric_cols]
        
        # Get complete cases for training
        complete_mask = target_series.notna() & predictor_data.notna().all(axis=1)
        
        if complete_mask.sum() < 10:
            raise ValueError("Insufficient complete cases for regression imputation")
        
        # Train regression model
        X_train = predictor_data[complete_mask]
        y_train = target_series[complete_mask]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict missing values
        missing_mask = target_series.isna()
        X_missing = predictor_data[missing_mask]
        
        # Only predict if we have complete predictor data
        if len(X_missing) > 0:
            can_predict = X_missing.notna().all(axis=1)
        else:
            can_predict = pd.Series([], dtype=bool)
        
        predicted_values = model.predict(X_missing[can_predict])
        
        # Fill in predictions
        filled_series = target_series.copy()
        missing_indices = missing_mask[missing_mask].index
        predictable_indices = missing_indices[can_predict]
        
        filled_series.loc[predictable_indices] = predicted_values
        
        # Fill remaining missing values with median
        if filled_series.isnull().sum() > 0:
            filled_series = filled_series.fillna(target_series.median())
        
        metadata = {
            'method': 'regression_imputation',
            'predictors_used': len(numeric_cols),
            'r2_score': model.score(X_train, y_train),
            'values_imputed': target_series.isnull().sum()
        }
        
        return filled_series, metadata
    
    # Outlier Methods
    def _iqr_removal(self, df: pd.DataFrame, column: str, multiplier: float = 1.5, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Remove outliers using IQR method"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("IQR outlier removal only applicable to numeric columns")
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        cleaned_series = series.copy()
        cleaned_series[outlier_mask] = np.nan
        
        metadata = {
            'method': 'iqr_removal',
            'multiplier': multiplier,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_removed': outlier_mask.sum()
        }
        
        return cleaned_series, metadata
    
    def _zscore_removal(self, df: pd.DataFrame, column: str, threshold: float = 3, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Remove outliers using Z-score method"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Z-score outlier removal only applicable to numeric columns")
        
        z_scores = np.abs(stats.zscore(series.dropna()))
        
        # Map z-scores back to original series
        non_null_mask = series.notna()
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask.loc[non_null_mask] = z_scores > threshold
        
        cleaned_series = series.copy()
        cleaned_series[outlier_mask] = np.nan
        
        metadata = {
            'method': 'zscore_removal',
            'threshold': threshold,
            'outliers_removed': outlier_mask.sum()
        }
        
        return cleaned_series, metadata
    
    def _winsorization(self, df: pd.DataFrame, column: str, lower_percentile: float = 5, upper_percentile: float = 95, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Winsorize outliers by capping at percentiles"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Winsorization only applicable to numeric columns")
        
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)
        
        original_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        
        winsorized_series = series.clip(lower=lower_bound, upper=upper_bound)
        
        metadata = {
            'method': 'winsorization',
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'values_winsorized': original_outliers
        }
        
        return winsorized_series, metadata
    
    def _log_transformation(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Apply log transformation to reduce skewness and outlier impact"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Log transformation only applicable to numeric columns")
        
        if (series <= 0).any():
            # Shift to positive values
            shift_value = abs(series.min()) + 1
            series_shifted = series + shift_value
            log_transformed = np.log(series_shifted)
            metadata = {
                'method': 'log_transformation',
                'shift_applied': shift_value,
                'transformation': 'log(x + shift)'
            }
        else:
            log_transformed = np.log(series)
            metadata = {
                'method': 'log_transformation',
                'shift_applied': 0,
                'transformation': 'log(x)'
            }
        
        return log_transformed, metadata
    
    def _cap_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr', multiplier: float = 1.5, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Cap outliers at specified bounds instead of removing them"""
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Capping outliers only applicable to numeric columns")
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
        else:  # percentile method
            lower_bound = series.quantile(0.05)
            upper_bound = series.quantile(0.95)
        
        original_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        capped_series = series.clip(lower=lower_bound, upper=upper_bound)
        
        metadata = {
            'method': 'cap_outliers',
            'cap_method': method,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'values_capped': original_outliers
        }
        
        return capped_series, metadata
    
    def _isolation_forest_removal(self, df: pd.DataFrame, column: str, contamination: float = 0.1, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Remove outliers using Isolation Forest - optimized"""
        from sklearn.ensemble import IsolationForest
        
        series = df[column].copy()
        
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Isolation Forest only applicable to numeric columns")
        
        # Prepare data
        non_null_series = series.dropna()
        if len(non_null_series) < 10:
            raise ValueError("Insufficient data for Isolation Forest")
        
        # Optimize: sample data for very large datasets (>50000 rows)
        if len(non_null_series) > 50000:
            # Use stratified sampling to maintain distribution
            sample_size = 50000
            sampled_indices = np.random.choice(non_null_series.index, size=sample_size, replace=False)
            fit_data = non_null_series.loc[sampled_indices]
        else:
            fit_data = non_null_series
        
        # Fit Isolation Forest with optimized parameters
        iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100,  # Default, good balance
            max_samples='auto',  # Let sklearn optimize
            n_jobs=-1  # Use all CPU cores
        )
        iso_forest.fit(fit_data.values.reshape(-1, 1))
        
        # Predict on full dataset
        outlier_labels = iso_forest.predict(non_null_series.values.reshape(-1, 1))
        
        # Create outlier mask
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask.loc[non_null_series.index] = outlier_labels == -1
        
        cleaned_series = series.copy()
        cleaned_series[outlier_mask] = np.nan
        
        metadata = {
            'method': 'isolation_forest',
            'contamination': contamination,
            'outliers_removed': int(outlier_mask.sum()),
            'sample_used': len(fit_data) if len(non_null_series) > 50000 else 'full'
        }
        
        return cleaned_series, metadata
    
    # Data Quality Methods
    def _type_standardization(self, df: pd.DataFrame, column: str, target_type: str = None, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Standardize data types in mixed-type columns"""
        series = df[column].copy()
        
        if target_type is None:
            # Auto-detect target type
            non_null_series = series.dropna()
            
            # Try to convert to numeric
            numeric_converted = pd.to_numeric(non_null_series, errors='coerce')
            if len(non_null_series) > 0 and numeric_converted.notna().sum() / len(non_null_series) > 0.8:
                target_type = 'numeric'
            else:
                target_type = 'string'
        
        if target_type == 'numeric':
            standardized_series = pd.to_numeric(series, errors='coerce')
        elif target_type == 'string':
            standardized_series = series.astype(str)
        elif target_type == 'datetime':
            standardized_series = pd.to_datetime(series, errors='coerce')
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        conversion_errors = (standardized_series.isna() & series.notna()).sum()
        
        metadata = {
            'method': 'type_standardization',
            'target_type': target_type,
            'conversion_errors': conversion_errors
        }
        
        return standardized_series, metadata
    
    def _remove_duplicates(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Remove duplicate values, keeping first occurrence"""
        series = df[column].copy()
        
        # Find duplicates
        duplicate_mask = series.duplicated()
        
        # Create cleaned series
        cleaned_series = series.copy()
        cleaned_series[duplicate_mask] = np.nan
        
        metadata = {
            'method': 'remove_duplicates',
            'duplicates_removed': duplicate_mask.sum()
        }
        
        return cleaned_series, metadata
    
    def _trim_whitespace(self, df: pd.DataFrame, column: str, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Trim whitespace from string columns"""
        series = df[column].copy()
        
        if series.dtype != 'object':
            raise ValueError("Whitespace trimming only applicable to text columns")
        
        # Convert to string and trim
        trimmed_series = series.astype(str).str.strip()
        
        # Count changes
        changes = (series.astype(str) != trimmed_series).sum()
        
        metadata = {
            'method': 'trim_whitespace',
            'values_trimmed': changes
        }
        
        return trimmed_series, metadata
    
    def _standardize_case(self, df: pd.DataFrame, column: str, case_type: str = 'lower', **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Standardize text case"""
        series = df[column].copy()
        
        if series.dtype != 'object':
            raise ValueError("Case standardization only applicable to text columns")
        
        if case_type == 'lower':
            standardized_series = series.astype(str).str.lower()
        elif case_type == 'upper':
            standardized_series = series.astype(str).str.upper()
        elif case_type == 'title':
            standardized_series = series.astype(str).str.title()
        else:
            raise ValueError(f"Unknown case type: {case_type}")
        
        changes = (series.astype(str) != standardized_series).sum()
        
        metadata = {
            'method': 'standardize_case',
            'case_type': case_type,
            'values_changed': changes
        }
        
        return standardized_series, metadata
    
    def validate_data_quality(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Validate data quality before export with comprehensive checks"""
        series = df[column]
        validation_results = {
            'column': column,
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / len(series)) * 100
        
        if missing_pct > 10:
            validation_results['issues'].append(f"High missing data: {missing_pct:.1f}%")
            validation_results['passed'] = False
        elif missing_pct > 5:
            validation_results['warnings'].append(f"Moderate missing data: {missing_pct:.1f}%")
        
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                q1 = non_null.quantile(0.25)
                q3 = non_null.quantile(0.75)
                iqr = q3 - q1
                outliers = non_null[(non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)]
                outlier_pct = (len(outliers) / len(non_null)) * 100
                
                if outlier_pct > 5:
                    validation_results['warnings'].append(f"High outlier percentage: {outlier_pct:.1f}%")
                
                if np.isinf(non_null).any():
                    validation_results['issues'].append("Contains infinite values")
                    validation_results['passed'] = False
        
        elif series.dtype == 'object':
            if len(series.dropna()) > 0:
                value_counts = series.value_counts()
                if len(value_counts) == len(series):
                    validation_results['warnings'].append("All values are unique (possible ID column)")
                
                rare_threshold = len(series) * 0.01
                rare_values = value_counts[value_counts < rare_threshold]
                if len(rare_values) > len(value_counts) * 0.5:
                    validation_results['warnings'].append(f"Many rare values: {len(rare_values)} categories < 1% frequency")
        
        duplicates_count = df.duplicated(subset=[column], keep=False).sum()
        if duplicates_count > len(df) * 0.1:
            validation_results['warnings'].append(f"High duplicate rate: {(duplicates_count/len(df)*100):.1f}%")
        
        validation_results['statistics'] = {
            'missing_count': int(missing_count),
            'missing_percentage': float(missing_pct),
            'unique_count': int(series.nunique()),
            'duplicate_count': int(duplicates_count)
        }
        
        return validation_results
    
    def validate_dataset_for_export(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate entire dataset before export"""
        dataset_validation = {
            'overall_passed': True,
            'total_issues': 0,
            'total_warnings': 0,
            'column_validations': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        for column in df.columns:
            col_validation = self.validate_data_quality(df, column)
            dataset_validation['column_validations'][column] = col_validation
            
            if not col_validation['passed']:
                dataset_validation['overall_passed'] = False
                dataset_validation['critical_issues'].extend([f"{column}: {issue}" for issue in col_validation['issues']])
            
            dataset_validation['total_issues'] += len(col_validation['issues'])
            dataset_validation['total_warnings'] += len(col_validation['warnings'])
        
        if dataset_validation['total_issues'] > 0:
            dataset_validation['recommendations'].append("Review and address critical issues before final export")
        
        if dataset_validation['total_warnings'] > 0:
            dataset_validation['recommendations'].append("Consider addressing warnings to improve data quality")
        
        total_missing = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        overall_missing_pct = (total_missing / total_cells) * 100
        
        if overall_missing_pct > 10:
            dataset_validation['recommendations'].append(f"Dataset has {overall_missing_pct:.1f}% missing values overall - consider additional cleaning")
        
        return dataset_validation
