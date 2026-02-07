import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class SurveyWeightsManager:
    """Manages design weights and weighted calculations for survey data"""
    
    def __init__(self):
        self.weights_column = None
        self.weights_metadata = {}
    
    def set_weights_column(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """Set the column to use as design weights"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset")
        
        weights = df[column_name]
        
        if not pd.api.types.is_numeric_dtype(weights):
            raise ValueError("Weights column must be numeric")
        
        # Validate weights
        validation_results = self._validate_weights(weights)
        
        if validation_results['valid']:
            self.weights_column = column_name
            self.weights_metadata = {
                'column': column_name,
                'mean': weights.mean(),
                'min': weights.min(),
                'max': weights.max(),
                'std': weights.std(),
                'total_weight': weights.sum(),
                'negative_weights': (weights < 0).sum(),
                'zero_weights': (weights == 0).sum(),
                'validation': validation_results
            }
        
        return validation_results
    
    def _validate_weights(self, weights: pd.Series) -> Dict[str, Any]:
        """Validate design weights"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for missing weights
        missing_count = weights.isnull().sum()
        if missing_count > 0:
            validation['warnings'].append(f"Found {missing_count} missing weights")
        
        # Check for negative weights
        negative_count = (weights < 0).sum()
        if negative_count > 0:
            validation['warnings'].append(f"Found {negative_count} negative weights")
        
        # Check for zero weights
        zero_count = (weights == 0).sum()
        if zero_count > 0:
            validation['warnings'].append(f"Found {zero_count} zero weights")
        
        # Check for extremely large weights (potential outliers)
        if len(weights.dropna()) > 0:
            q99 = weights.quantile(0.99)
            q1 = weights.quantile(0.01)
            if q99 / q1 > 100:  # Ratio of 99th to 1st percentile
                validation['warnings'].append("Weights have high variability - check for outliers")
        
        # Check if all weights are the same (no weighting needed)
        if weights.nunique() <= 1:
            validation['warnings'].append("All weights are the same - no weighting effect")
        
        return validation
    
    def calculate_weighted_summary(self, df: pd.DataFrame, column: str, 
                                 include_unweighted: bool = True) -> Dict[str, Any]:
        """Calculate weighted and unweighted summary statistics"""
        if self.weights_column is None:
            raise ValueError("No weights column set. Use set_weights_column() first.")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        series = df[column]
        weights = df[self.weights_column]
        
        # Remove rows with missing values in either column or weights
        valid_mask = series.notna() & weights.notna()
        clean_series = series[valid_mask]
        clean_weights = weights[valid_mask]
        
        summary = {
            'column': column,
            'weights_column': self.weights_column,
            'valid_observations': len(clean_series)
        }
        
        # Weighted statistics
        if len(clean_series) > 0 and clean_weights.sum() > 0:
            summary['weighted'] = self._calculate_weighted_stats(clean_series, clean_weights)
        else:
            summary['weighted'] = {'error': 'No valid weighted observations'}
        
        # Unweighted statistics for comparison
        if include_unweighted:
            summary['unweighted'] = self._calculate_unweighted_stats(clean_series)
        
        return summary
    
    def _calculate_weighted_stats(self, series: pd.Series, weights: pd.Series) -> Dict[str, Any]:
        """Calculate weighted statistics"""
        total_weight = weights.sum()
        
        stats = {
            'count': len(series),
            'total_weight': total_weight,
            'effective_sample_size': (weights.sum() ** 2) / (weights ** 2).sum()
        }
        
        if pd.api.types.is_numeric_dtype(series):
            # Weighted mean
            weighted_mean = float((series * weights).sum() / total_weight)
            stats['mean'] = weighted_mean
            
            # Weighted variance and std
            weighted_variance = float(((series - weighted_mean) ** 2 * weights).sum() / total_weight)
            stats['variance'] = weighted_variance
            stats['std'] = float(np.sqrt(weighted_variance))
            
            # Weighted percentiles (approximation)
            sorted_indices = np.argsort(series.values)
            sorted_values = series.iloc[sorted_indices]
            sorted_weights = weights.iloc[sorted_indices]
            cumulative_weights = np.cumsum(sorted_weights)
            
            # Find weighted percentiles
            percentiles = [0.25, 0.5, 0.75]
            for p in percentiles:
                target_weight = p * total_weight
                idx = np.searchsorted(cumulative_weights, target_weight, side='left')
                if idx < len(sorted_values):
                    stats[f'q{int(p*100)}'] = sorted_values.iloc[idx]
                else:
                    stats[f'q{int(p*100)}'] = sorted_values.iloc[-1]
            
            stats['min'] = float(series.min())
            stats['max'] = float(series.max())
            
        elif series.dtype == 'object':
            # Weighted mode and frequencies
            value_counts = {}
            for val, weight in zip(series, weights):
                if pd.notna(val):
                    value_counts[val] = value_counts.get(val, 0) + weight
            
            if value_counts:
                stats['mode'] = max(value_counts, key=value_counts.get)
                stats['mode_weight'] = value_counts[stats['mode']]
                stats['unique_values'] = len(value_counts)
                
                # Top 5 most frequent values
                sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                stats['top_values'] = sorted_counts[:5]
        
        return stats
    
    def _calculate_unweighted_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate unweighted statistics for comparison"""
        stats = {
            'count': len(series),
            'unique_values': series.nunique()
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': series.mean(),
                'std': series.std(),
                'variance': series.var(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q50': series.quantile(0.5),
                'q75': series.quantile(0.75)
            })
        elif series.dtype == 'object':
            mode_result = series.mode()
            stats['mode'] = mode_result.iloc[0] if len(mode_result) > 0 else None
            value_counts = series.value_counts()
            stats['top_values'] = [(val, count) for val, count in value_counts.head().items()]
        
        return stats
    
    def calculate_margin_of_error(self, df: pd.DataFrame, column: str, 
                                confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate margin of error for weighted estimates"""
        if self.weights_column is None:
            raise ValueError("No weights column set. Use set_weights_column() first.")
        
        from scipy import stats as scipy_stats
        
        series = df[column]
        weights = df[self.weights_column]
        
        # Remove missing values
        valid_mask = series.notna() & weights.notna()
        clean_series = series[valid_mask]
        clean_weights = weights[valid_mask]
        
        if len(clean_series) == 0:
            return {'error': 'No valid observations'}
        
        result = {
            'column': column,
            'confidence_level': confidence_level,
            'sample_size': len(clean_series),
            'total_weight': clean_weights.sum()
        }
        
        if pd.api.types.is_numeric_dtype(clean_series):
            # Calculate design effect
            effective_sample_size = (clean_weights.sum() ** 2) / (clean_weights ** 2).sum()
            design_effect = len(clean_series) / effective_sample_size
            
            # Weighted mean and variance
            weighted_mean = (clean_series * clean_weights).sum() / clean_weights.sum()
            weighted_variance = ((clean_series - weighted_mean) ** 2 * clean_weights).sum() / clean_weights.sum()
            
            # Standard error with design effect
            standard_error = np.sqrt(weighted_variance / effective_sample_size)
            
            # Critical value for given confidence level
            alpha = 1 - confidence_level
            critical_value = scipy_stats.t.ppf(1 - alpha/2, df=effective_sample_size - 1)
            
            # Margin of error
            margin_of_error = critical_value * standard_error
            
            result.update({
                'weighted_mean': weighted_mean,
                'standard_error': standard_error,
                'design_effect': design_effect,
                'effective_sample_size': effective_sample_size,
                'margin_of_error': margin_of_error,
                'confidence_interval': [
                    weighted_mean - margin_of_error,
                    weighted_mean + margin_of_error
                ]
            })
        
        elif clean_series.dtype == 'object':
            # For categorical variables, calculate margins for proportions
            total_weight = clean_weights.sum()
            proportions = {}
            margins = {}
            
            for category in clean_series.unique():
                if pd.notna(category):
                    category_mask = clean_series == category
                    category_weight = clean_weights[category_mask].sum()
                    proportion = category_weight / total_weight
                    
                    # Standard error for proportion with design effect
                    effective_n = (clean_weights.sum() ** 2) / (clean_weights ** 2).sum()
                    design_effect = len(clean_series) / effective_n
                    
                    se_proportion = np.sqrt((proportion * (1 - proportion)) / effective_n * design_effect)
                    
                    # Critical value
                    alpha = 1 - confidence_level
                    critical_value = scipy_stats.norm.ppf(1 - alpha/2)
                    
                    margin = critical_value * se_proportion
                    
                    proportions[category] = proportion
                    margins[category] = margin
            
            result.update({
                'proportions': proportions,
                'margins_of_error': margins,
                'confidence_intervals': {
                    cat: [prop - margins[cat], prop + margins[cat]] 
                    for cat, prop in proportions.items()
                }
            })
        
        return result
    
    def compare_weighted_unweighted(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Compare weighted vs unweighted results for multiple columns"""
        if self.weights_column is None:
            raise ValueError("No weights column set. Use set_weights_column() first.")
        
        comparison = {
            'weights_column': self.weights_column,
            'weights_metadata': self.weights_metadata,
            'columns_compared': columns,
            'results': {}
        }
        
        for column in columns:
            if column in df.columns:
                summary = self.calculate_weighted_summary(df, column, include_unweighted=True)
                
                # Calculate differences
                if 'weighted' in summary and 'unweighted' in summary:
                    weighted = summary['weighted']
                    unweighted = summary['unweighted']
                    
                    differences = {}
                    if 'mean' in weighted and 'mean' in unweighted:
                        differences['mean_difference'] = weighted['mean'] - unweighted['mean']
                        differences['mean_percent_change'] = (differences['mean_difference'] / unweighted['mean']) * 100 if unweighted['mean'] != 0 else 0
                    
                    if 'std' in weighted and 'std' in unweighted:
                        differences['std_difference'] = weighted['std'] - unweighted['std']
                    
                    summary['differences'] = differences
                
                comparison['results'][column] = summary
        
        return comparison
    
    def get_weights_diagnostics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get diagnostic information about the weights"""
        if self.weights_column is None:
            return {'error': 'No weights column set'}
        
        weights = df[self.weights_column]
        
        diagnostics = {
            'basic_stats': {
                'count': len(weights),
                'missing': weights.isnull().sum(),
                'mean': weights.mean(),
                'median': weights.median(),
                'std': weights.std(),
                'min': weights.min(),
                'max': weights.max(),
                'q25': weights.quantile(0.25),
                'q75': weights.quantile(0.75)
            },
            'distribution_analysis': {
                'unique_values': weights.nunique(),
                'coefficient_of_variation': weights.std() / weights.mean() if weights.mean() != 0 else 0,
                'range_ratio': weights.max() / weights.min() if weights.min() > 0 else float('inf'),
                'negative_weights': (weights < 0).sum(),
                'zero_weights': (weights == 0).sum(),
                'extreme_weights': ((weights < weights.quantile(0.01)) | 
                                  (weights > weights.quantile(0.99))).sum()
            },
            'design_effect_estimate': {
                'effective_sample_size': (weights.sum() ** 2) / (weights ** 2).sum() if (weights ** 2).sum() > 0 else 0,
                'design_effect': len(weights) / ((weights.sum() ** 2) / (weights ** 2).sum()) if (weights ** 2).sum() > 0 else 1
            }
        }
        
        # Add recommendations
        recommendations = []
        if diagnostics['basic_stats']['missing'] > 0:
            recommendations.append("Handle missing weights before analysis")
        
        if diagnostics['distribution_analysis']['negative_weights'] > 0:
            recommendations.append("Review negative weights - they may indicate data quality issues")
        
        if diagnostics['distribution_analysis']['coefficient_of_variation'] > 2:
            recommendations.append("High weight variability detected - consider weight trimming")
        
        if diagnostics['design_effect_estimate']['design_effect'] > 2:
            recommendations.append("High design effect - weights substantially affect variance estimates")
        
        diagnostics['recommendations'] = recommendations
        
        return diagnostics