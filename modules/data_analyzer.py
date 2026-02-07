import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import streamlit as st
import hashlib
import warnings
warnings.filterwarnings('ignore')

class ColumnAnalyzer:
    """Individual column analysis engine with multiple detection methods"""
    
    def __init__(self):
        self.analysis_cache = {}
        self._cached_correlations = {}
    
    def analyze_column(self, df: pd.DataFrame, column: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Comprehensive analysis of a single column"""
        # Create a robust cache key that detects any data changes (order-aware, deterministic)
        series = df[column]
        # Use order-aware deterministic hash for stable caching across sessions
        hash_array = pd.util.hash_pandas_object(series, index=False).values
        data_hash = hashlib.sha256(hash_array.tobytes()).hexdigest()[:16]  # Deterministic hash
        cache_key = f"{column}_{len(df)}_{data_hash}"
        
        if not force_refresh and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        series = df[column]
        analysis = {
            'column_name': column,
            'basic_info': self._get_basic_info(series),
            'missing_analysis': self._analyze_missing_patterns(df, column),
            'outlier_analysis': self._detect_outliers(series),
            'distribution_analysis': self._analyze_distribution(series),
            'data_quality': self._assess_data_quality(series),
            'relationships': self._analyze_relationships(df, column),
            'rule_violations': self._detect_rule_violations(series, column),
            'cleaning_recommendations': []
        }
        
        # Generate specific cleaning recommendations
        analysis['cleaning_recommendations'] = self._generate_cleaning_recommendations(analysis)
        
        self.analysis_cache[cache_key] = analysis
        return analysis
    
    def _get_basic_info(self, series: pd.Series) -> Dict[str, Any]:
        """Get basic information about the column"""
        info = {
            'dtype': str(series.dtype),
            'count': len(series),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
            'memory_usage': series.memory_usage(deep=True)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                info.update({
                    'mean': non_null_series.mean(),
                    'median': non_null_series.median(),
                    'std': non_null_series.std(),
                    'min': non_null_series.min(),
                    'max': non_null_series.max(),
                    'q25': non_null_series.quantile(0.25),
                    'q75': non_null_series.quantile(0.75),
                    'skewness': stats.skew(non_null_series),
                    'kurtosis': stats.kurtosis(non_null_series)
                })
        
        return info
    
    def _analyze_missing_patterns(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze missing data patterns for the specific column - optimized"""
        series = df[column]
        missing_mask = series.isnull()
        
        missing_sum = missing_mask.sum()
        if missing_sum == 0:
            return {
                'total_missing': 0,
                'percentage': 0,
                'pattern_type': 'none', 
                'consecutive_missing': [],
                'missing_by_position': {},
                'max_consecutive': 0,
                'analysis': 'No missing values found'
            }
        
        series_len = len(series)
        analysis = {
            'total_missing': missing_sum,
            'percentage': (missing_sum / series_len) * 100,
            'pattern_type': 'unknown',
            'consecutive_missing': [],
            'missing_by_position': {}
        }
        
        # Find consecutive missing values - optimized with numpy
        missing_array = missing_mask.values.astype(int)
        diff = np.diff(np.concatenate(([0], missing_array, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        consecutive = (ends - starts).tolist()
        
        analysis['consecutive_missing'] = consecutive
        analysis['max_consecutive'] = int(max(consecutive)) if consecutive else 0
        
        # Analyze missing pattern type
        if analysis['percentage'] < 5:
            analysis['pattern_type'] = 'sporadic'
        elif analysis['max_consecutive'] > len(series) * 0.1:
            analysis['pattern_type'] = 'systematic_blocks'
        elif missing_mask.iloc[:int(len(series) * 0.1)].sum() > missing_mask.iloc[int(len(series) * 0.9):].sum() * 2:
            analysis['pattern_type'] = 'front_loaded'
        elif missing_mask.iloc[int(len(series) * 0.9):].sum() > missing_mask.iloc[:int(len(series) * 0.1)].sum() * 2:
            analysis['pattern_type'] = 'tail_loaded'
        else:
            analysis['pattern_type'] = 'random'
        
        return analysis
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple methods - optimized"""
        if not pd.api.types.is_numeric_dtype(series):
            return {
                'method_results': {}, 
                'summary': {
                    'methods_agree': False,
                    'consensus_outliers': 0,
                    'severity': 'low',
                    'analysis': 'Not applicable for non-numeric data'
                }
            }
        
        non_null_series = series.dropna()
        if len(non_null_series) < 5:
            return {
                'method_results': {}, 
                'summary': {
                    'methods_agree': False,
                    'consensus_outliers': 0,
                    'severity': 'low',
                    'analysis': 'Insufficient data for outlier detection (need at least 5 values)'
                }
            }
        
        outlier_results = {}
        
        # IQR Method - vectorized calculation
        Q1, Q3 = non_null_series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_mask = (non_null_series < lower_bound) | (non_null_series > upper_bound)
        iqr_outliers = non_null_series[iqr_mask]
        outlier_results['iqr'] = {
            'method': 'Interquartile Range (IQR)',
            'outlier_count': len(iqr_outliers),
            'outlier_percentage': (len(iqr_outliers) / len(non_null_series)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': iqr_outliers.tolist()[:20]  # Limit to 20 for display
        }
        
        # Z-Score Method (only for larger samples where it's more reliable)
        if len(non_null_series) >= 10:
            z_scores = np.abs(stats.zscore(non_null_series))
            z_outliers = non_null_series[z_scores > 3]
            outlier_results['zscore'] = {
                'method': 'Z-Score (|z| > 3)',
                'outlier_count': len(z_outliers),
                'outlier_percentage': (len(z_outliers) / len(non_null_series)) * 100,
                'threshold': 3,
                'outlier_values': z_outliers.tolist()[:20]
            }
            
            # Modified Z-Score Method
            median = np.median(non_null_series)
            mad = np.median(np.abs(non_null_series - median))
            modified_z_scores = 0.6745 * (non_null_series - median) / mad if mad != 0 else np.zeros(len(non_null_series))
            modified_z_outliers = non_null_series[np.abs(modified_z_scores) > 3.5]
            outlier_results['modified_zscore'] = {
                'method': 'Modified Z-Score (|Mz| > 3.5)',
                'outlier_count': len(modified_z_outliers),
                'outlier_percentage': (len(modified_z_outliers) / len(non_null_series)) * 100,
                'threshold': 3.5,
                'outlier_values': modified_z_outliers.tolist()[:20]
            }
        else:
            # For small samples, add a note about why some methods are skipped
            outlier_results['note_small_sample'] = {
                'method': 'Note: Z-Score methods',
                'outlier_count': 0,
                'outlier_percentage': 0,
                'note': f'Skipped for small sample size (n={len(non_null_series)}). Z-score methods require larger samples for reliability.'
            }
        
        # Statistical summary
        total_outliers = set()
        for method_result in outlier_results.values():
            outlier_values = method_result.get('outlier_values', [])
            if outlier_values:
                total_outliers.update(outlier_values)
        
        # Calculate severity based on outlier percentages (excluding note entries)
        outlier_percentages = [
            r.get('outlier_percentage', 0) for r in outlier_results.values() 
            if 'note' not in r
        ]
        max_percentage = max(outlier_percentages) if outlier_percentages else 0
        
        # Check if methods agree (only if both IQR and Z-score methods were run)
        methods_agree = False
        if len(non_null_series) >= 10 and 'iqr' in outlier_results and 'zscore' in outlier_results:
            iqr_set = set(outlier_results['iqr']['outlier_values'])
            zscore_set = set(outlier_results['zscore']['outlier_values'])
            methods_agree = len(iqr_set & zscore_set) > 0
        
        summary = {
            'methods_agree': methods_agree,
            'consensus_outliers': len(total_outliers),
            'severity': 'high' if max_percentage > 10 else 'moderate' if max_percentage > 5 else 'low'
        }
        
        return {
            'method_results': outlier_results,
            'summary': summary
        }
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution characteristics of the column"""
        if not pd.api.types.is_numeric_dtype(series):
            # For categorical data
            value_counts = series.value_counts()
            return {
                'type': 'categorical',
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequency_distribution': value_counts.head(10).to_dict(),
                'entropy': stats.entropy(value_counts.values) if len(value_counts) > 0 else 0
            }
        
        non_null_series = series.dropna()
        if len(non_null_series) < 10:
            return {'type': 'insufficient_data'}
        
        # Statistical tests for normality (limit sample size for performance)
        sample_size = min(5000, len(non_null_series))
        sample_data = non_null_series.sample(sample_size) if len(non_null_series) > sample_size else non_null_series
        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
        
        analysis = {
            'type': 'numeric',
            'skewness': stats.skew(non_null_series),
            'kurtosis': stats.kurtosis(non_null_series),
            'normality_test': {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        }
        
        # Distribution characterization
        skew_val = abs(analysis['skewness'])
        if skew_val < 0.5:
            analysis['distribution_shape'] = 'approximately_normal'
        elif skew_val < 1:
            analysis['distribution_shape'] = 'moderately_skewed'
        else:
            analysis['distribution_shape'] = 'highly_skewed'
        
        return analysis
    
    def _assess_data_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess overall data quality for the column"""
        quality_score = 100
        issues = []
        
        # Missing data penalty
        missing_pct = (series.isnull().sum() / len(series)) * 100
        if missing_pct > 50:
            quality_score -= 40
            issues.append(f"High missing data rate ({missing_pct:.1f}%)")
        elif missing_pct > 20:
            quality_score -= 20
            issues.append(f"Moderate missing data rate ({missing_pct:.1f}%)")
        elif missing_pct > 5:
            quality_score -= 10
            issues.append(f"Low missing data rate ({missing_pct:.1f}%)")
        
        # Data type consistency
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            # Check for mixed types in object columns
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                sample_types = set(type(x).__name__ for x in non_null_series.sample(min(1000, len(non_null_series))))
                if len(sample_types) > 1:
                    quality_score -= 15
                    issues.append("Mixed data types detected")
        
        # Uniqueness assessment
        unique_pct = (series.nunique() / len(series)) * 100
        if unique_pct > 95 and len(series) > 100:
            issues.append("Very high uniqueness - possible identifier column")
        elif unique_pct < 1:
            quality_score -= 10
            issues.append("Very low uniqueness - mostly repeated values")
        
        return {
            'score': max(0, quality_score),
            'grade': 'A' if quality_score >= 90 else 'B' if quality_score >= 80 else 'C' if quality_score >= 70 else 'D' if quality_score >= 60 else 'F',
            'issues': issues
        }
    
    def _analyze_relationships(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze relationships with other columns"""
        series = df[column]
        relationships = {}
        
        # Find numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if column in numeric_cols and len(numeric_cols) > 1:
            numeric_cols.remove(column)
            correlations = {}
            for col in numeric_cols[:10]:  # Limit to top 10 for performance
                try:
                    corr = series.corr(df[col])
                    if pd.notna(corr) and abs(corr) > 0.1:
                        correlations[col] = corr
                except Exception:
                    continue
            
            relationships['correlations'] = dict(sorted(correlations.items(), 
                                                      key=lambda x: abs(x[1]), 
                                                      reverse=True)[:5])
        
        return relationships
    
    def _detect_rule_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Detect rule-based violations in the column data"""
        violations = {
            'total_violations': 0,
            'violation_types': [],
            'severity': 'low',
            'details': {}
        }
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return violations
        
        # Numeric range violations
        if pd.api.types.is_numeric_dtype(series):
            violations.update(self._check_numeric_range_violations(non_null_series, column_name))
        
        # Text format violations
        elif series.dtype == 'object':
            violations.update(self._check_text_format_violations(non_null_series, column_name))
        
        # Categorical consistency violations
        violations.update(self._check_categorical_violations(non_null_series, column_name))
        
        # Determine overall severity
        if violations['total_violations'] > len(series) * 0.1:
            violations['severity'] = 'high'
        elif violations['total_violations'] > len(series) * 0.05:
            violations['severity'] = 'moderate'
        else:
            violations['severity'] = 'low'
            
        return violations
    
    def detect_inter_column_violations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect violations based on inter-column dependencies"""
        violations = {
            'total_violations': 0,
            'violation_types': [],
            'details': {},
            'severity': 'low',
            'affected_rows': set(),
            'rule_checks': []
        }
        
        # Check age vs birth_year consistency
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        birth_year_cols = [col for col in df.columns if 'birth_year' in col.lower() or 'birth' in col.lower()]
        
        if age_cols and birth_year_cols:
            from datetime import datetime
            current_year = datetime.now().year
            
            for age_col in age_cols:
                for birth_col in birth_year_cols:
                    if pd.api.types.is_numeric_dtype(df[age_col]) and pd.api.types.is_numeric_dtype(df[birth_col]):
                        # Calculate expected age from birth year
                        expected_ages = current_year - df[birth_col]
                        age_diff = abs(df[age_col] - expected_ages)
                        # Allow for 1-2 years difference (depending on birth month)
                        inconsistent_mask = age_diff > 2
                        
                        if inconsistent_mask.sum() > 0:
                            violations['total_violations'] += inconsistent_mask.sum()
                            violations['violation_types'].append(f'Age-birth year inconsistency ({age_col} vs {birth_col})')
                            violations['details'][f'{age_col}_{birth_col}_inconsistency'] = {
                                'count': inconsistent_mask.sum(),
                                'rule': 'Age should match birth year within 1-2 years',
                                'affected_rows': df.index[inconsistent_mask].tolist()[:10]
                            }
        
        # Check start_date vs end_date consistency
        start_cols = [col for col in df.columns if 'start' in col.lower() and ('date' in col.lower() or 'time' in col.lower())]
        end_cols = [col for col in df.columns if 'end' in col.lower() and ('date' in col.lower() or 'time' in col.lower())]
        
        for start_col in start_cols:
            for end_col in end_cols:
                try:
                    start_dates = pd.to_datetime(df[start_col], errors='coerce')
                    end_dates = pd.to_datetime(df[end_col], errors='coerce')
                    
                    # Check if end dates are before start dates
                    invalid_dates = (end_dates < start_dates) & start_dates.notna() & end_dates.notna()
                    
                    if invalid_dates.sum() > 0:
                        violations['total_violations'] += invalid_dates.sum()
                        violations['violation_types'].append(f'End date before start date ({start_col} vs {end_col})')
                        violations['details'][f'{start_col}_{end_col}_date_logic'] = {
                            'count': invalid_dates.sum(),
                            'rule': 'End date should be after start date',
                            'affected_rows': df.index[invalid_dates].tolist()[:10]
                        }
                except:
                    pass  # Skip if date conversion fails
        
        # Check income vs education level consistency
        income_cols = [col for col in df.columns if any(word in col.lower() for word in ['income', 'salary', 'wage'])]
        education_cols = [col for col in df.columns if 'education' in col.lower() or 'degree' in col.lower()]
        
        if income_cols and education_cols:
            for income_col in income_cols:
                for edu_col in education_cols:
                    if pd.api.types.is_numeric_dtype(df[income_col]) and df[edu_col].dtype == 'object':
                        # Simple check: higher education should generally correlate with higher income
                        education_levels = df[edu_col].str.lower()
                        high_education = education_levels.str.contains('master|phd|doctorate|graduate', na=False)
                        low_education = education_levels.str.contains('high school|elementary|primary', na=False)
                        
                        high_edu_low_income = high_education & (df[income_col] < df[income_col].quantile(0.25))
                        low_edu_high_income = low_education & (df[income_col] > df[income_col].quantile(0.75))
                        
                        total_anomalies = high_edu_low_income.sum() + low_edu_high_income.sum()
                        if total_anomalies > 0:
                            violations['total_violations'] += total_anomalies
                            violations['violation_types'].append(f'Education-income mismatch ({edu_col} vs {income_col})')
                            violations['details'][f'{edu_col}_{income_col}_mismatch'] = {
                                'count': total_anomalies,
                                'rule': 'Education level and income should generally correlate',
                                'high_edu_low_income': high_edu_low_income.sum(),
                                'low_edu_high_income': low_edu_high_income.sum()
                            }
        
        # Add comprehensive rule checks
        self._check_demographic_consistency(df, violations)
        self._check_logical_sequences(df, violations)
        self._check_survey_specific_rules(df, violations)
        self._check_mathematical_relationships(df, violations)
        self._check_missing_data_patterns(df, violations)
        self._check_conditional_dependencies(df, violations)
        
        # Convert affected_rows set to list for JSON serialization
        violations['affected_rows'] = list(violations['affected_rows'])
        
        # Determine overall severity
        total_rows = len(df)
        if violations['total_violations'] > total_rows * 0.1:
            violations['severity'] = 'high'
        elif violations['total_violations'] > total_rows * 0.05:
            violations['severity'] = 'moderate'
        else:
            violations['severity'] = 'low'
        
        return violations
    
    def _check_demographic_consistency(self, df: pd.DataFrame, violations: Dict[str, Any]) -> None:
        """Check demographic consistency rules"""
        # Gender vs title consistency
        gender_cols = [col for col in df.columns if any(word in col.lower() for word in ['gender', 'sex'])]
        title_cols = [col for col in df.columns if any(word in col.lower() for word in ['title', 'mr', 'mrs', 'ms'])]
        
        for gender_col in gender_cols:
            for title_col in title_cols:
                if df[gender_col].dtype == 'object' and df[title_col].dtype == 'object':
                    # Check for Mr. with Female or Mrs./Ms. with Male
                    male_indicators = df[gender_col].str.lower().str.contains('male|m', na=False)
                    female_indicators = df[gender_col].str.lower().str.contains('female|f', na=False)
                    mr_title = df[title_col].str.lower().str.contains('mr', na=False)
                    mrs_ms_title = df[title_col].str.lower().str.contains('mrs|ms', na=False)
                    
                    inconsistent = (male_indicators & mrs_ms_title) | (female_indicators & mr_title)
                    
                    if inconsistent.sum() > 0:
                        violations['total_violations'] += inconsistent.sum()
                        violations['violation_types'].append(f'Gender-title mismatch ({gender_col} vs {title_col})')
                        violations['affected_rows'].update(df.index[inconsistent].tolist())
                        violations['rule_checks'].append({
                            'rule_type': 'demographic_consistency',
                            'columns': [gender_col, title_col],
                            'violations': inconsistent.sum(),
                            'description': 'Gender and title should be consistent'
                        })
        
        # Age vs retirement status
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        retirement_cols = [col for col in df.columns if any(word in col.lower() for word in ['retired', 'retirement'])]
        
        for age_col in age_cols:
            for retire_col in retirement_cols:
                if pd.api.types.is_numeric_dtype(df[age_col]) and df[retire_col].dtype == 'object':
                    young_retired = (df[age_col] < 50) & df[retire_col].str.lower().str.contains('yes|retired|true', na=False)
                    old_working = (df[age_col] > 70) & df[retire_col].str.lower().str.contains('no|working|false', na=False)
                    
                    inconsistent = young_retired | old_working
                    if inconsistent.sum() > 0:
                        violations['total_violations'] += inconsistent.sum()
                        violations['violation_types'].append(f'Age-retirement inconsistency ({age_col} vs {retire_col})')
                        violations['affected_rows'].update(df.index[inconsistent].tolist())
    
    def _check_logical_sequences(self, df: pd.DataFrame, violations: Dict[str, Any]) -> None:
        """Check logical sequence rules"""
        # Employment start before end dates
        start_employment_cols = [col for col in df.columns if any(word in col.lower() for word in ['start', 'begin']) and 'employ' in col.lower()]
        end_employment_cols = [col for col in df.columns if any(word in col.lower() for word in ['end', 'finish']) and 'employ' in col.lower()]
        
        for start_col in start_employment_cols:
            for end_col in end_employment_cols:
                try:
                    start_dates = pd.to_datetime(df[start_col], errors='coerce')
                    end_dates = pd.to_datetime(df[end_col], errors='coerce')
                    
                    invalid_sequence = (end_dates < start_dates) & start_dates.notna() & end_dates.notna()
                    
                    if invalid_sequence.sum() > 0:
                        violations['total_violations'] += invalid_sequence.sum()
                        violations['violation_types'].append(f'Invalid employment sequence ({start_col} vs {end_col})')
                        violations['affected_rows'].update(df.index[invalid_sequence].tolist())
                        violations['rule_checks'].append({
                            'rule_type': 'logical_sequence',
                            'columns': [start_col, end_col],
                            'violations': invalid_sequence.sum(),
                            'description': 'Employment end date should be after start date'
                        })
                except:
                    pass
    
    def _check_survey_specific_rules(self, df: pd.DataFrame, violations: Dict[str, Any]) -> None:
        """Check survey-specific business rules"""
        # Household income vs individual income
        household_income_cols = [col for col in df.columns if 'household' in col.lower() and 'income' in col.lower()]
        individual_income_cols = [col for col in df.columns if 'income' in col.lower() and 'household' not in col.lower()]
        
        for household_col in household_income_cols:
            for individual_col in individual_income_cols:
                if pd.api.types.is_numeric_dtype(df[household_col]) and pd.api.types.is_numeric_dtype(df[individual_col]):
                    # Individual income should not exceed household income
                    invalid_income = (df[individual_col] > df[household_col]) & df[household_col].notna() & df[individual_col].notna()
                    
                    if invalid_income.sum() > 0:
                        violations['total_violations'] += invalid_income.sum()
                        violations['violation_types'].append(f'Individual income exceeds household ({individual_col} vs {household_col})')
                        violations['affected_rows'].update(df.index[invalid_income].tolist())
        
        # Education level progression
        education_cols = [col for col in df.columns if 'education' in col.lower() or 'degree' in col.lower()]
        for edu_col in education_cols:
            if df[edu_col].dtype == 'object':
                # Check for impossible education progressions
                edu_values = df[edu_col].str.lower()
                # This is a simplified check - would need domain-specific rules
                invalid_edu = edu_values.str.contains('phd.*high school|doctorate.*elementary', na=False)
                if invalid_edu.sum() > 0:
                    violations['total_violations'] += invalid_edu.sum()
                    violations['violation_types'].append(f'Invalid education progression in {edu_col}')
                    violations['affected_rows'].update(df.index[invalid_edu].tolist())
    
    def _check_mathematical_relationships(self, df: pd.DataFrame, violations: Dict[str, Any]) -> None:
        """Check mathematical relationships between columns"""
        # Total vs sum of parts
        total_cols = [col for col in df.columns if 'total' in col.lower()]
        
        for total_col in total_cols:
            if pd.api.types.is_numeric_dtype(df[total_col]):
                # Find related component columns
                base_name = total_col.lower().replace('total', '').replace('_', '').strip()
                related_cols = [col for col in df.columns 
                              if base_name in col.lower() and col != total_col and pd.api.types.is_numeric_dtype(df[col])]
                
                if len(related_cols) >= 2:  # Need at least 2 components
                    calculated_total = df[related_cols].sum(axis=1)
                    # Allow for small rounding differences
                    tolerance = df[total_col].std() * 0.01 if df[total_col].std() > 0 else 1
                    significant_diff = abs(df[total_col] - calculated_total) > tolerance
                    
                    valid_data = df[total_col].notna() & calculated_total.notna()
                    violations_mask = significant_diff & valid_data
                    
                    if violations_mask.sum() > 0:
                        violations['total_violations'] += violations_mask.sum()
                        violations['violation_types'].append(f'Total does not match sum of components ({total_col})')
                        violations['affected_rows'].update(df.index[violations_mask].tolist())
                        violations['rule_checks'].append({
                            'rule_type': 'mathematical_relationship',
                            'columns': [total_col] + related_cols,
                            'violations': violations_mask.sum(),
                            'description': f'Total {total_col} should equal sum of {related_cols}'
                        })
    
    def _check_missing_data_patterns(self, df: pd.DataFrame, violations: Dict[str, Any]) -> None:
        """Check for suspicious missing data patterns across columns"""
        # Check for columns that are always missing together
        missing_matrix = df.isnull()
        
        # Find column pairs with high correlation in missingness
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if missing_matrix[col1].sum() > 0 and missing_matrix[col2].sum() > 0:
                    # Calculate correlation between missing patterns
                    missing_corr = missing_matrix[col1].corr(missing_matrix[col2])
                    
                    # High positive correlation in missingness might indicate systematic issues
                    if missing_corr > 0.8:
                        violations['total_violations'] += min(missing_matrix[col1].sum(), missing_matrix[col2].sum())
                        violations['violation_types'].append(f'Correlated missing data pattern ({col1} vs {col2})')
                        violations['rule_checks'].append({
                            'rule_type': 'missing_data_pattern',
                            'columns': [col1, col2],
                            'correlation': missing_corr,
                            'description': f'Columns {col1} and {col2} have highly correlated missing patterns'
                        })
        
        # Check for columns that should not be missing when others are present
        required_pairs = self._get_required_column_pairs(df.columns)
        for primary_col, required_col in required_pairs:
            if primary_col in df.columns and required_col in df.columns:
                # When primary column has data, required column should also have data
                primary_present = df[primary_col].notna()
                required_missing = df[required_col].isna()
                violation_mask = primary_present & required_missing
                
                if violation_mask.sum() > 0:
                    violations['total_violations'] += violation_mask.sum()
                    violations['violation_types'].append(f'Required data missing ({required_col} when {primary_col} present)')
                    violations['affected_rows'].update(df.index[violation_mask].tolist())
                    violations['rule_checks'].append({
                        'rule_type': 'conditional_missing',
                        'columns': [primary_col, required_col],
                        'violations': violation_mask.sum(),
                        'description': f'{required_col} should not be missing when {primary_col} has data'
                    })
    
    def _check_conditional_dependencies(self, df: pd.DataFrame, violations: Dict[str, Any]) -> None:
        """Check for conditional dependencies between columns"""
        # Income vs employment status
        income_cols = [col for col in df.columns if any(word in col.lower() for word in ['income', 'salary', 'wage'])]
        employment_cols = [col for col in df.columns if any(word in col.lower() for word in ['employ', 'job', 'work'])]
        
        for income_col in income_cols:
            for employ_col in employment_cols:
                if pd.api.types.is_numeric_dtype(df[income_col]) and df[employ_col].dtype == 'object':
                    # People with income should generally be employed
                    has_income = df[income_col] > 0
                    unemployed = df[employ_col].str.lower().str.contains('unemployed|not working|retired', na=False)
                    
                    contradiction = has_income & unemployed
                    if contradiction.sum() > 0:
                        violations['total_violations'] += contradiction.sum()
                        violations['violation_types'].append(f'Income without employment ({income_col} vs {employ_col})')
                        violations['affected_rows'].update(df.index[contradiction].tolist())
        
        # Family size vs number of children
        family_cols = [col for col in df.columns if any(word in col.lower() for word in ['family_size', 'household_size'])]
        children_cols = [col for col in df.columns if any(word in col.lower() for word in ['children', 'kids', 'child'])]
        
        for family_col in family_cols:
            for children_col in children_cols:
                if pd.api.types.is_numeric_dtype(df[family_col]) and pd.api.types.is_numeric_dtype(df[children_col]):
                    # Number of children cannot exceed family size
                    invalid_family = (df[children_col] > df[family_col]) & df[family_col].notna() & df[children_col].notna()
                    
                    if invalid_family.sum() > 0:
                        violations['total_violations'] += invalid_family.sum()
                        violations['violation_types'].append(f'Children exceed family size ({children_col} vs {family_col})')
                        violations['affected_rows'].update(df.index[invalid_family].tolist())
        
        # Marital status vs age
        marital_cols = [col for col in df.columns if any(word in col.lower() for word in ['marital', 'married', 'spouse'])]
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        
        for marital_col in marital_cols:
            for age_col in age_cols:
                if df[marital_col].dtype == 'object' and pd.api.types.is_numeric_dtype(df[age_col]):
                    # Very young people should not be married
                    married = df[marital_col].str.lower().str.contains('married|spouse', na=False)
                    very_young = df[age_col] < 16
                    
                    unlikely_married = married & very_young
                    if unlikely_married.sum() > 0:
                        violations['total_violations'] += unlikely_married.sum()
                        violations['violation_types'].append(f'Unlikely marriage age ({age_col} vs {marital_col})')
                        violations['affected_rows'].update(df.index[unlikely_married].tolist())
    
    def _get_required_column_pairs(self, columns: List[str]) -> List[Tuple[str, str]]:
        """Get pairs of columns where one requires the other to be present"""
        pairs = []
        column_list = list(columns)
        
        # Common required pairs
        required_patterns = [
            ('email', 'phone'),  # If email exists, phone might be required
            ('first_name', 'last_name'),  # First name requires last name
            ('street', 'city'),  # Street address requires city
            ('start_date', 'end_date'),  # Start date might require end date
            ('income', 'employment'),  # Income requires employment info
        ]
        
        for primary_pattern, required_pattern in required_patterns:
            primary_cols = [col for col in column_list if primary_pattern in col.lower()]
            required_cols = [col for col in column_list if required_pattern in col.lower()]
            
            for primary_col in primary_cols:
                for required_col in required_cols:
                    if primary_col != required_col:
                        pairs.append((primary_col, required_col))
        
        return pairs
    
    def _check_numeric_range_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check for numeric range violations based on column type inference"""
        violations = {'total_violations': 0, 'violation_types': [], 'details': {}}
        
        # Infer expected ranges based on column name patterns
        column_lower = column_name.lower()
        
        # Age-related columns
        if any(keyword in column_lower for keyword in ['age', 'years_old', 'birth_year']):
            if 'age' in column_lower:
                invalid_ages = series[(series < 0) | (series > 120)]
                if len(invalid_ages) > 0:
                    violations['total_violations'] += len(invalid_ages)
                    violations['violation_types'].append('Invalid age range')
                    violations['details']['age_violations'] = {
                        'count': len(invalid_ages),
                        'rule': 'Age should be between 0 and 120',
                        'invalid_values': invalid_ages.tolist()[:10]
                    }
            
            elif 'birth_year' in column_lower:
                from datetime import datetime
                current_year = datetime.now().year
                invalid_years = series[(series < 1900) | (series > current_year)]
                if len(invalid_years) > 0:
                    violations['total_violations'] += len(invalid_years)
                    violations['violation_types'].append('Invalid birth year')
                    violations['details']['birth_year_violations'] = {
                        'count': len(invalid_years),
                        'rule': f'Birth year should be between 1900 and {current_year}',
                        'invalid_values': invalid_years.tolist()[:10]
                    }
        
        # Percentage columns
        elif any(keyword in column_lower for keyword in ['percent', 'percentage', 'rate', 'ratio']):
            invalid_percentages = series[(series < 0) | (series > 100)]
            if len(invalid_percentages) > 0:
                violations['total_violations'] += len(invalid_percentages)
                violations['violation_types'].append('Invalid percentage range')
                violations['details']['percentage_violations'] = {
                    'count': len(invalid_percentages),
                    'rule': 'Percentage should be between 0 and 100',
                    'invalid_values': invalid_percentages.tolist()[:10]
                }
        
        # Score columns
        elif any(keyword in column_lower for keyword in ['score', 'rating', 'grade']):
            # Check for impossible negative scores
            negative_scores = series[series < 0]
            if len(negative_scores) > 0:
                violations['total_violations'] += len(negative_scores)
                violations['violation_types'].append('Negative scores detected')
                violations['details']['negative_score_violations'] = {
                    'count': len(negative_scores),
                    'rule': 'Scores should not be negative',
                    'invalid_values': negative_scores.tolist()[:10]
                }
        
        # Income/salary columns
        elif any(keyword in column_lower for keyword in ['income', 'salary', 'wage', 'earnings']):
            negative_income = series[series < 0]
            if len(negative_income) > 0:
                violations['total_violations'] += len(negative_income)
                violations['violation_types'].append('Negative income values')
                violations['details']['income_violations'] = {
                    'count': len(negative_income),
                    'rule': 'Income should not be negative',
                    'invalid_values': negative_income.tolist()[:10]
                }
        
        return violations
    
    def _check_text_format_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check for text format violations"""
        violations = {'total_violations': 0, 'violation_types': [], 'details': {}}
        
        column_lower = column_name.lower()
        
        # Email format validation
        if any(keyword in column_lower for keyword in ['email', 'e_mail', 'mail']):
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = series[~series.str.match(email_pattern, na=False)]
            if len(invalid_emails) > 0:
                violations['total_violations'] += len(invalid_emails)
                violations['violation_types'].append('Invalid email format')
                violations['details']['email_violations'] = {
                    'count': len(invalid_emails),
                    'rule': 'Email should follow valid email format',
                    'invalid_values': invalid_emails.tolist()[:10]
                }
        
        # Phone number format validation
        elif any(keyword in column_lower for keyword in ['phone', 'telephone', 'mobile']):
            import re
            # Basic phone number pattern (flexible for international formats)
            phone_pattern = r'^[\+]?[1-9][\d\s\-\(\)]{6,}$'
            invalid_phones = series[~series.str.match(phone_pattern, na=False)]
            if len(invalid_phones) > 0:
                violations['total_violations'] += len(invalid_phones)
                violations['violation_types'].append('Invalid phone format')
                violations['details']['phone_violations'] = {
                    'count': len(invalid_phones),
                    'rule': 'Phone number should follow valid format',
                    'invalid_values': invalid_phones.tolist()[:10]
                }
        
        # Postal/ZIP code validation
        elif any(keyword in column_lower for keyword in ['zip', 'postal', 'postcode']):
            import re
            # Basic postal code pattern (numbers and letters, 3-10 characters)
            postal_pattern = r'^[A-Z0-9\s\-]{3,10}$'
            invalid_postal = series[~series.str.upper().str.match(postal_pattern, na=False)]
            if len(invalid_postal) > 0:
                violations['total_violations'] += len(invalid_postal)
                violations['violation_types'].append('Invalid postal code format')
                violations['details']['postal_violations'] = {
                    'count': len(invalid_postal),
                    'rule': 'Postal code should be 3-10 alphanumeric characters',
                    'invalid_values': invalid_postal.tolist()[:10]
                }
        
        return violations
    
    def _check_categorical_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check for categorical consistency violations"""
        violations = {'total_violations': 0, 'violation_types': [], 'details': {}}
        
        column_lower = column_name.lower()
        
        # Gender consistency
        if any(keyword in column_lower for keyword in ['gender', 'sex']):
            valid_genders = {'male', 'm', 'female', 'f', 'other', 'non-binary', 'prefer not to say', 'unknown'}
            if series.dtype == 'object':
                invalid_genders = series[~series.str.lower().isin(valid_genders)]
                if len(invalid_genders) > 0:
                    violations['total_violations'] += len(invalid_genders)
                    violations['violation_types'].append('Invalid gender values')
                    violations['details']['gender_violations'] = {
                        'count': len(invalid_genders),
                        'rule': 'Gender should be from standard categories',
                        'invalid_values': invalid_genders.tolist()[:10]
                    }
        
        # Yes/No questions
        elif any(keyword in column_lower for keyword in ['yes_no', 'yn', 'boolean', 'flag']):
            valid_responses = {'yes', 'y', 'no', 'n', 'true', 'false', '1', '0'}
            if series.dtype == 'object':
                invalid_responses = series[~series.str.lower().isin(valid_responses)]
                if len(invalid_responses) > 0:
                    violations['total_violations'] += len(invalid_responses)
                    violations['violation_types'].append('Invalid yes/no responses')
                    violations['details']['yesno_violations'] = {
                        'count': len(invalid_responses),
                        'rule': 'Should be Yes/No or True/False',
                        'invalid_values': invalid_responses.tolist()[:10]
                    }
        
        # Check for unusual characters or encoding issues
        if series.dtype == 'object':
            import re
            # Check for unusual Unicode characters that might indicate encoding issues
            unusual_chars = series[series.str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
            if len(unusual_chars) > 0:
                violations['total_violations'] += len(unusual_chars)
                violations['violation_types'].append('Unusual character encoding')
                violations['details']['encoding_violations'] = {
                    'count': len(unusual_chars),
                    'rule': 'Contains non-ASCII characters that may indicate encoding issues',
                    'invalid_values': unusual_chars.tolist()[:5]
                }
        
        return violations
    
    def _generate_cleaning_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate context-specific cleaning recommendations for the column"""
        recommendations = []
        
        basic_info = analysis['basic_info']
        missing_analysis = analysis['missing_analysis']
        outlier_analysis = analysis['outlier_analysis']
        quality = analysis['data_quality']
        
        # Missing value recommendations
        if missing_analysis.get('percentage', 0) > 0:
            missing_pct = missing_analysis['percentage']
            pattern_type = missing_analysis.get('pattern_type', 'unknown')
            
            if missing_pct < 5:
                if pd.api.types.is_numeric_dtype(pd.Series(dtype=basic_info['dtype'])):
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'median_imputation',
                        'priority': 'high',
                        'description': 'Use median imputation for low missing rate in numeric column',
                        'pros': ['Simple', 'Robust to outliers', 'Preserves distribution center'],
                        'cons': ['Reduces variance', 'May not preserve relationships'],
                        'applicability_score': 85
                    })
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'knn_imputation',
                        'priority': 'medium',
                        'description': 'Use KNN imputation to preserve relationships',
                        'pros': ['Preserves relationships', 'More sophisticated'],
                        'cons': ['Computationally expensive', 'Sensitive to scaling'],
                        'applicability_score': 75
                    })
                else:
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'mode_imputation',
                        'priority': 'high',
                        'description': 'Use mode imputation for categorical column',
                        'pros': ['Preserves most common category', 'Simple'],
                        'cons': ['May increase bias toward common values'],
                        'applicability_score': 80
                    })
            
            elif missing_pct < 20:
                if pattern_type == 'systematic_blocks':
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'interpolation',
                        'priority': 'high',
                        'description': 'Use interpolation for systematic missing blocks',
                        'pros': ['Good for time series', 'Preserves trends'],
                        'cons': ['May not work for non-sequential data'],
                        'applicability_score': 70
                    })
                else:
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'regression_imputation',
                        'priority': 'medium',
                        'description': 'Use regression-based imputation',
                        'pros': ['Preserves relationships', 'More accurate'],
                        'cons': ['Complex', 'Requires related columns'],
                        'applicability_score': 75
                    })
            else:
                recommendations.append({
                    'type': 'missing_values',
                    'method': 'missing_category',
                    'priority': 'high',
                    'description': 'Create "Missing" category for high missing rate',
                    'pros': ['Preserves all data', 'Explicit about missingness'],
                    'cons': ['Changes interpretation', 'May need special handling'],
                    'applicability_score': 90
                })
        
        # Outlier recommendations
        if 'method_results' in outlier_analysis and outlier_analysis['method_results']:
            severity = outlier_analysis.get('summary', {}).get('severity', 'low')
            
            if severity == 'high':
                recommendations.append({
                    'type': 'outliers',
                    'method': 'winsorization',
                    'priority': 'high',
                    'description': 'Use winsorization to cap extreme values',
                    'pros': ['Preserves sample size', 'Reduces extreme influence'],
                    'cons': ['Changes distribution', 'May mask important patterns'],
                    'applicability_score': 85
                })
                recommendations.append({
                    'type': 'outliers',
                    'method': 'removal',
                    'priority': 'medium',
                    'description': 'Remove outliers (use with caution)',
                    'pros': ['Clean dataset', 'Improves normality'],
                    'cons': ['Loses information', 'Reduces sample size'],
                    'applicability_score': 60
                })
            elif severity == 'moderate':
                recommendations.append({
                    'type': 'outliers',
                    'method': 'log_transformation',
                    'priority': 'medium',
                    'description': 'Use log transformation to reduce outlier impact',
                    'pros': ['Natural approach', 'Improves normality'],
                    'cons': ['Changes interpretation', 'Requires positive values'],
                    'applicability_score': 75
                })
        
        # Data quality recommendations
        if quality['score'] < 70:
            for issue in quality['issues']:
                if 'mixed data types' in issue.lower():
                    recommendations.append({
                        'type': 'data_quality',
                        'method': 'type_standardization',
                        'priority': 'high',
                        'description': 'Standardize data types in column',
                        'pros': ['Consistent processing', 'Prevents errors'],
                        'cons': ['May lose information', 'Manual review needed'],
                        'applicability_score': 95
                    })
        
        # Sort recommendations by priority and applicability
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['applicability_score']), reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations
