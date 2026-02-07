import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.formula.api import logit
import warnings

class HypothesisAnalyzer:
    """Comprehensive hypothesis testing and statistical analysis module"""
    
    def __init__(self):
        self.alpha = 0.05  # Default significance level
        
    def set_alpha(self, alpha: float):
        """Set significance level"""
        self.alpha = alpha
    
    def recommend_test(self, df: pd.DataFrame, columns: List[str], data_types: Dict[str, str]) -> Dict[str, Any]:
        """Intelligently recommend the most suitable hypothesis test"""
        
        if len(columns) < 1:
            return {'error': 'Select at least one column'}
        
        # Determine column types
        col_types = [data_types.get(col, self._infer_type(df[col])) for col in columns]
        
        recommendations = []
        
        if len(columns) == 1:
            col_type = col_types[0]
            if col_type == 'numeric':
                recommendations.append({
                    'test': 'one_sample_ttest',
                    'reason': 'Test if the mean differs from a specific value',
                    'priority': 'medium'
                })
        
        elif len(columns) == 2:
            # Two numeric columns
            if all(ct == 'numeric' for ct in col_types):
                # Check normality
                normality_checks = [self._check_normality(df[col]) for col in columns]
                
                if all(normality_checks):
                    recommendations.append({
                        'test': 'pearson_correlation',
                        'reason': 'Both variables are normally distributed - Pearson correlation recommended',
                        'priority': 'high'
                    })
                else:
                    recommendations.append({
                        'test': 'spearman_correlation',
                        'reason': 'Non-normal distribution detected - Spearman correlation recommended',
                        'priority': 'high'
                    })
                
                recommendations.append({
                    'test': 'simple_linear_regression',
                    'reason': 'Explore linear relationship between variables',
                    'priority': 'medium'
                })
            
            # One numeric, one categorical
            elif 'numeric' in col_types and 'categorical' in col_types:
                num_col = columns[col_types.index('numeric')]
                cat_col = columns[col_types.index('categorical')]
                
                unique_categories = df[cat_col].nunique()
                
                if unique_categories == 2:
                    # Check normality and equal variance
                    groups = [df[df[cat_col] == cat][num_col].dropna() for cat in df[cat_col].unique()[:2]]
                    normality = all(self._check_normality(g) for g in groups if len(g) > 2)
                    
                    if not normality:
                        recommendations.append({
                            'test': 'mann_whitney',
                            'reason': 'Non-normal distribution - Mann-Whitney U test recommended',
                            'priority': 'high'
                        })
                    else:
                        recommendations.append({
                            'test': 'welch_ttest',
                            'reason': 'Two groups with numeric outcome - Welch\'s t-test recommended (robust to unequal variances)',
                            'priority': 'high'
                        })
                        recommendations.append({
                            'test': 'independent_ttest',
                            'reason': 'Alternative: Independent t-test (assumes equal variances)',
                            'priority': 'medium'
                        })
                
                elif unique_categories > 2:
                    recommendations.append({
                        'test': 'one_way_anova',
                        'reason': f'{unique_categories} groups - One-way ANOVA to compare means',
                        'priority': 'high'
                    })
                    recommendations.append({
                        'test': 'kruskal_wallis',
                        'reason': 'Non-parametric alternative to ANOVA',
                        'priority': 'medium'
                    })
            
            # Two categorical columns
            elif all(ct == 'categorical' for ct in col_types):
                recommendations.append({
                    'test': 'chi_square',
                    'reason': 'Two categorical variables - Chi-square test for independence',
                    'priority': 'high'
                })
                
                # Check if suitable for Fisher's exact test
                contingency = pd.crosstab(df[columns[0]], df[columns[1]])
                if contingency.size <= 20 and contingency.min().min() < 5:
                    recommendations.append({
                        'test': 'fisher_exact',
                        'reason': 'Small cell counts - Fisher\'s exact test recommended',
                        'priority': 'high'
                    })
        
        return {
            'recommendations': recommendations,
            'column_types': dict(zip(columns, col_types)),
            'alpha': self.alpha
        }
    
    def one_sample_ttest(self, df: pd.DataFrame, column: str, test_value: float = 0) -> Dict[str, Any]:
        """One-sample t-test"""
        try:
            data = df[column].dropna()
            
            if len(data) < 2:
                return {'error': 'Insufficient data for one-sample t-test'}
            
            # Perform one-sample t-test
            statistic, p_value = stats.ttest_1samp(data, test_value)
            
            # Calculate effect size (Cohen's d)
            cohens_d = (data.mean() - test_value) / data.std() if data.std() > 0 else 0
            
            # Confidence interval for mean
            se = data.std() / np.sqrt(len(data))
            t_crit = stats.t.ppf(1 - self.alpha/2, len(data) - 1)
            ci = (data.mean() - t_crit * se, data.mean() + t_crit * se)
            
            # Assumption checks
            assumptions = {
                'normality': self._check_normality(data),
                'sample_size': len(data)
            }
            
            return {
                'test_name': 'One-sample t-test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': len(data) - 1,
                'effect_size': {'type': "Cohen's d", 'value': float(cohens_d)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': ci},
                'alpha': self.alpha,
                'decision': 'Mean differs significantly from test value' if p_value < self.alpha else 'No significant difference from test value',
                'sample_sizes': {'n': len(data)},
                'sample_mean': float(data.mean()),
                'test_value': float(test_value),
                'missing_count': df[column].isna().sum(),
                'assumption_checks': assumptions,
                'visualizations': ['histogram', 'box_plot'],
                'interpretation': f"The sample mean ({data.mean():.3f}) {'differs significantly' if p_value < self.alpha else 'does not differ significantly'} from {test_value} (p = {p_value:.4f})",
                'warnings': self._generate_test_warnings(assumptions, 'one_sample_ttest')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def welch_ttest(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Welch's t-test (doesn't assume equal variances)"""
        try:
            groups = df[group_col].dropna().unique()[:2]
            group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
            group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                return {'error': 'Insufficient data in one or both groups'}
            
            # Perform Welch's t-test
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Calculate confidence interval
            se_diff = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
            df_welch = (group1.var()/len(group1) + group2.var()/len(group2))**2 / \
                       ((group1.var()/len(group1))**2/(len(group1)-1) + (group2.var()/len(group2))**2/(len(group2)-1))
            t_crit = stats.t.ppf(1 - self.alpha/2, df_welch)
            ci = ((group1.mean() - group2.mean()) - t_crit * se_diff,
                  (group1.mean() - group2.mean()) + t_crit * se_diff)
            
            # Assumption checks
            assumptions = {
                'normality_group1': self._check_normality(group1),
                'normality_group2': self._check_normality(group2),
                'sample_sizes': {'group1': len(group1), 'group2': len(group2)}
            }
            
            return {
                'test_name': "Welch's t-test",
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': float(df_welch),
                'effect_size': {'type': "Cohen's d", 'value': float(cohens_d)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': ci},
                'alpha': self.alpha,
                'decision': 'Reject H0' if p_value < self.alpha else 'Fail to reject H0',
                'sample_sizes': {'group1': len(group1), 'group2': len(group2)},
                'group_stats': {
                    str(groups[0]): {'mean': float(group1.mean()), 'std': float(group1.std()), 'n': len(group1)},
                    str(groups[1]): {'mean': float(group2.mean()), 'std': float(group2.std()), 'n': len(group2)}
                },
                'missing_count': df[numeric_col].isna().sum(),
                'assumption_checks': assumptions,
                'visualizations': ['box_plot', 'violin_plot'],
                'interpretation': self._interpret_ttest(p_value, cohens_d, groups[0], groups[1]),
                'warnings': self._generate_test_warnings(assumptions, 'welch_ttest')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def mann_whitney(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Mann-Whitney U test (non-parametric alternative to t-test)"""
        try:
            groups = df[group_col].dropna().unique()[:2]
            group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
            group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                return {'error': 'Insufficient data in one or both groups'}
            
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            r = 1 - (2*statistic) / (len(group1) * len(group2))
            
            return {
                'test_name': 'Mann-Whitney U test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'Rank-biserial correlation', 'value': float(r)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Reject H0' if p_value < self.alpha else 'Fail to reject H0',
                'sample_sizes': {'group1': len(group1), 'group2': len(group2)},
                'group_stats': {
                    str(groups[0]): {'median': float(group1.median()), 'mean_rank': float(group1.rank().mean()), 'n': len(group1)},
                    str(groups[1]): {'median': float(group2.median()), 'mean_rank': float(group2.rank().mean()), 'n': len(group2)}
                },
                'missing_count': df[numeric_col].isna().sum(),
                'assumption_checks': {'independence': 'Assumed', 'ordinal_or_continuous': 'Yes'},
                'visualizations': ['box_plot', 'violin_plot'],
                'notes': 'Non-parametric test - does not assume normal distribution',
                'interpretation': f"The {'median' if p_value < self.alpha else 'distributions'} of {groups[0]} and {groups[1]} {'differ significantly' if p_value < self.alpha else 'do not differ significantly'} (p = {p_value:.4f})",
                'warnings': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def pearson_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Pearson correlation coefficient"""
        try:
            # Remove missing values
            valid_data = df[[col1, col2]].dropna()
            
            if len(valid_data) < 3:
                return {'error': 'Insufficient data for correlation'}
            
            # Calculate Pearson correlation
            r, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
            
            # Calculate confidence interval
            z = np.arctanh(r)
            se = 1/np.sqrt(len(valid_data)-3)
            z_crit = stats.norm.ppf(1 - self.alpha/2)
            ci_z = (z - z_crit*se, z + z_crit*se)
            ci = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
            
            # R-squared
            r_squared = r**2
            
            # Assumption checks
            assumptions = {
                'linearity': 'Visual inspection recommended',
                'normality_col1': self._check_normality(valid_data[col1]),
                'normality_col2': self._check_normality(valid_data[col2]),
                'homoscedasticity': 'Visual inspection recommended'
            }
            
            return {
                'test_name': 'Pearson Correlation',
                'statistic': float(r),
                'p_value': float(p_value),
                'df': len(valid_data) - 2,
                'effect_size': {'type': 'R-squared', 'value': float(r_squared)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': ci},
                'alpha': self.alpha,
                'decision': 'Significant correlation' if p_value < self.alpha else 'No significant correlation',
                'sample_sizes': {'n': len(valid_data)},
                'missing_count': len(df) - len(valid_data),
                'assumption_checks': assumptions,
                'visualizations': ['scatter_plot', 'regression_line'],
                'interpretation': self._interpret_correlation(r, p_value),
                'warnings': self._generate_test_warnings(assumptions, 'pearson')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def spearman_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Spearman rank correlation coefficient"""
        try:
            # Remove missing values
            valid_data = df[[col1, col2]].dropna()
            
            if len(valid_data) < 3:
                return {'error': 'Insufficient data for correlation'}
            
            # Calculate Spearman correlation
            rho, p_value = stats.spearmanr(valid_data[col1], valid_data[col2])
            
            return {
                'test_name': 'Spearman Correlation',
                'statistic': float(rho),
                'p_value': float(p_value),
                'df': len(valid_data) - 2,
                'effect_size': {'type': 'Spearman rho', 'value': float(rho)},
                'confidence_interval': {'level': 'N/A', 'interval': 'Use bootstrap for CI'},
                'alpha': self.alpha,
                'decision': 'Significant correlation' if p_value < self.alpha else 'No significant correlation',
                'sample_sizes': {'n': len(valid_data)},
                'missing_count': len(df) - len(valid_data),
                'assumption_checks': {'monotonic_relationship': 'Assumed', 'ordinal_or_continuous': 'Yes'},
                'visualizations': ['scatter_plot'],
                'notes': 'Non-parametric - based on ranks, robust to outliers',
                'interpretation': self._interpret_correlation(rho, p_value, correlation_type='monotonic'),
                'warnings': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def chi_square(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Chi-square test of independence"""
        try:
            # Create contingency table
            contingency = pd.crosstab(df[col1], df[col2])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Calculate effect size (Cramér's V)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape[0], contingency.shape[1])
            cramers_v = np.sqrt(chi2 / (n * (min_dim - 1)))
            
            # Check assumptions
            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()
            
            warnings_list = []
            if min_expected < 1:
                warnings_list.append('Some expected frequencies < 1. Consider Fisher\'s exact test.')
            elif cells_below_5 > 0.2 * expected.size:
                warnings_list.append(f'{cells_below_5} cells have expected frequency < 5. Results may be unreliable.')
            
            return {
                'test_name': 'Chi-square test of independence',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'df': int(dof),
                'effect_size': {'type': "Cramér's V", 'value': float(cramers_v)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Variables are associated' if p_value < self.alpha else 'No significant association',
                'sample_sizes': {'n': int(n)},
                'missing_count': df[[col1, col2]].isna().any(axis=1).sum(),
                'assumption_checks': {
                    'min_expected_frequency': float(min_expected),
                    'cells_below_5': int(cells_below_5),
                    'independence': 'Assumed'
                },
                'contingency_table': contingency.to_dict(),
                'expected_frequencies': pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).to_dict(),
                'visualizations': ['heatmap', 'grouped_bar'],
                'interpretation': f"The variables {col1} and {col2} {'are significantly associated' if p_value < self.alpha else 'are independent'} (χ² = {chi2:.2f}, p = {p_value:.4f})",
                'warnings': warnings_list
            }
        except Exception as e:
            return {'error': str(e)}
    
    def fisher_exact(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Fisher's exact test (for 2x2 tables)"""
        try:
            # Create contingency table
            contingency = pd.crosstab(df[col1], df[col2])
            
            # Check if 2x2
            if contingency.shape != (2, 2):
                return {'error': 'Fisher\'s exact test requires a 2x2 contingency table'}
            
            # Perform Fisher's exact test
            oddsratio, p_value = stats.fisher_exact(contingency)
            
            return {
                'test_name': "Fisher's exact test",
                'statistic': float(oddsratio),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'Odds Ratio', 'value': float(oddsratio)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Variables are associated' if p_value < self.alpha else 'No significant association',
                'sample_sizes': {'n': int(contingency.sum().sum())},
                'missing_count': df[[col1, col2]].isna().any(axis=1).sum(),
                'assumption_checks': {'independence': 'Assumed'},
                'contingency_table': contingency.to_dict(),
                'visualizations': ['grouped_bar'],
                'notes': 'Exact test - suitable for small sample sizes',
                'interpretation': f"The odds ratio is {oddsratio:.2f}, indicating {col1} and {col2} {'are significantly associated' if p_value < self.alpha else 'are independent'} (p = {p_value:.4f})",
                'warnings': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def one_way_anova(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """One-way ANOVA"""
        try:
            # Get groups
            valid_categories = df[group_col].dropna().unique()
            groups = [df[df[group_col] == cat][numeric_col].dropna() for cat in valid_categories]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups for ANOVA'}
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            grand_mean = df[numeric_col].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum((df[numeric_col] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Degrees of freedom
            df_between = len(groups) - 1
            df_within = len(df[numeric_col].dropna()) - len(groups)
            
            # Assumption checks
            normality_checks = {f'group_{i}': self._check_normality(g) for i, g in enumerate(groups) if len(g) > 2}
            
            # Levene's test for homogeneity of variances
            levene_stat, levene_p = stats.levene(*groups)
            
            assumptions = {
                'normality_per_group': normality_checks,
                'homogeneity_of_variance': {'statistic': float(levene_stat), 'p_value': float(levene_p), 'passed': levene_p > 0.05}
            }
            
            return {
                'test_name': 'One-way ANOVA',
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'df': {'between_groups': df_between, 'within_groups': df_within},
                'effect_size': {'type': 'Eta-squared', 'value': float(eta_squared)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'At least one group mean differs' if p_value < self.alpha else 'No significant difference',
                'sample_sizes': {str(cat): len(df[df[group_col] == cat][numeric_col].dropna()) for cat in valid_categories},
                'missing_count': df[numeric_col].isna().sum(),
                'assumption_checks': assumptions,
                'group_stats': {
                    str(cat): {
                        'mean': float(df[df[group_col] == cat][numeric_col].mean()),
                        'std': float(df[df[group_col] == cat][numeric_col].std()),
                        'n': len(df[df[group_col] == cat][numeric_col].dropna())
                    } for cat in valid_categories
                },
                'visualizations': ['box_plot', 'violin_plot'],
                'interpretation': f"{'At least one group mean differs significantly' if p_value < self.alpha else 'No significant differences'} across groups (F = {f_stat:.2f}, p = {p_value:.4f})",
                'warnings': self._generate_test_warnings(assumptions, 'anova'),
                'notes': 'Use post-hoc tests (e.g., Tukey HSD) if significant'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def kruskal_wallis(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Kruskal-Wallis H test (non-parametric alternative to ANOVA)"""
        try:
            # Get groups
            valid_categories = df[group_col].dropna().unique()
            groups = [df[df[group_col] == cat][numeric_col].dropna() for cat in valid_categories]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups for Kruskal-Wallis test'}
            
            # Perform Kruskal-Wallis test
            h_stat, p_value = stats.kruskal(*groups)
            
            # Calculate effect size (epsilon-squared)
            n = len(df[numeric_col].dropna())
            k = len(groups)
            epsilon_squared = (h_stat - k + 1) / (n - k) if (n - k) > 0 else 0
            
            return {
                'test_name': 'Kruskal-Wallis H test',
                'statistic': float(h_stat),
                'p_value': float(p_value),
                'df': len(groups) - 1,
                'effect_size': {'type': 'Epsilon-squared', 'value': float(epsilon_squared)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'At least one group distribution differs' if p_value < self.alpha else 'No significant difference',
                'sample_sizes': {str(cat): len(df[df[group_col] == cat][numeric_col].dropna()) for cat in valid_categories},
                'missing_count': df[numeric_col].isna().sum(),
                'assumption_checks': {'independence': 'Assumed', 'ordinal_or_continuous': 'Yes'},
                'group_stats': {
                    str(cat): {
                        'median': float(df[df[group_col] == cat][numeric_col].median()),
                        'mean_rank': float(df[df[group_col] == cat][numeric_col].rank().mean()),
                        'n': len(df[df[group_col] == cat][numeric_col].dropna())
                    } for cat in valid_categories
                },
                'visualizations': ['box_plot', 'violin_plot'],
                'notes': 'Non-parametric test - does not assume normal distribution',
                'interpretation': f"{'At least one group distribution differs significantly' if p_value < self.alpha else 'No significant differences'} across groups (H = {h_stat:.2f}, p = {p_value:.4f})",
                'warnings': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def tukey_hsd(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Tukey's Honestly Significant Difference post-hoc test"""
        try:
            # Prepare data
            valid_data = df[[numeric_col, group_col]].dropna()
            
            if len(valid_data) < 3:
                return {'error': 'Insufficient data for Tukey HSD'}
            
            # Perform Tukey HSD
            tukey_result = pairwise_tukeyhsd(valid_data[numeric_col], valid_data[group_col], alpha=self.alpha)
            
            # Parse results
            comparisons = []
            for i in range(len(tukey_result.groupsunique)):
                for j in range(i+1, len(tukey_result.groupsunique)):
                    idx = i * (len(tukey_result.groupsunique) - 1) + j - 1 if i < j else None
                    if idx is not None and idx < len(tukey_result.reject):
                        comparisons.append({
                            'group1': str(tukey_result.groupsunique[i]),
                            'group2': str(tukey_result.groupsunique[j]),
                            'mean_diff': float(tukey_result.meandiffs[idx] if idx < len(tukey_result.meandiffs) else 0),
                            'reject': bool(tukey_result.reject[idx]),
                            'significant': 'Yes' if tukey_result.reject[idx] else 'No'
                        })
            
            return {
                'test_name': "Tukey's HSD post-hoc test",
                'statistic': 'Multiple comparisons',
                'p_value': 'See individual comparisons',
                'df': None,
                'effect_size': {'type': 'Mean differences', 'value': 'See comparisons'},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': 'See individual comparisons'},
                'alpha': self.alpha,
                'decision': f"{sum(c['reject'] for c in comparisons)} significant pairwise differences found",
                'sample_sizes': {str(cat): len(valid_data[valid_data[group_col] == cat]) for cat in valid_data[group_col].unique()},
                'missing_count': len(df) - len(valid_data),
                'assumption_checks': {'normality': 'Inherited from ANOVA', 'homogeneity_of_variance': 'Inherited from ANOVA'},
                'comparisons': comparisons,
                'visualizations': ['comparison_plot'],
                'interpretation': f"Found {sum(c['reject'] for c in comparisons)} significant pairwise differences out of {len(comparisons)} comparisons",
                'warnings': ['Use only after significant ANOVA result']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def two_proportion_ztest(self, successes: List[int], totals: List[int]) -> Dict[str, Any]:
        """Two-proportion z-test"""
        try:
            if len(successes) != 2 or len(totals) != 2:
                return {'error': 'Need exactly 2 proportions to compare'}
            
            # Perform test
            z_stat, p_value = proportions_ztest(successes, totals)
            
            # Calculate proportions
            p1 = successes[0] / totals[0]
            p2 = successes[1] / totals[1]
            
            # Calculate effect size (difference in proportions)
            diff = p1 - p2
            
            # Pooled proportion for SE
            pooled_p = sum(successes) / sum(totals)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/totals[0] + 1/totals[1]))
            
            # Confidence interval
            z_crit = stats.norm.ppf(1 - self.alpha/2)
            ci = (diff - z_crit * se, diff + z_crit * se)
            
            return {
                'test_name': 'Two-proportion z-test',
                'statistic': float(z_stat),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'Difference in proportions', 'value': float(diff)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': ci},
                'alpha': self.alpha,
                'decision': 'Proportions differ significantly' if p_value < self.alpha else 'No significant difference',
                'sample_sizes': {'group1': totals[0], 'group2': totals[1]},
                'proportions': {'p1': float(p1), 'p2': float(p2)},
                'missing_count': 0,
                'assumption_checks': {
                    'sample_size_group1': 'Adequate' if totals[0] * p1 >= 5 and totals[0] * (1-p1) >= 5 else 'Insufficient',
                    'sample_size_group2': 'Adequate' if totals[1] * p2 >= 5 and totals[1] * (1-p2) >= 5 else 'Insufficient'
                },
                'visualizations': ['bar_chart'],
                'interpretation': f"The proportions {'differ significantly' if p_value < self.alpha else 'do not differ significantly'} (p1 = {p1:.3f}, p2 = {p2:.3f}, p = {p_value:.4f})",
                'warnings': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def paired_ttest(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Paired t-test (for matched samples)"""
        try:
            # Remove missing values pairwise
            valid_data = df[[col1, col2]].dropna()
            
            if len(valid_data) < 2:
                return {'error': 'Insufficient paired data'}
            
            # Perform paired t-test
            statistic, p_value = stats.ttest_rel(valid_data[col1], valid_data[col2])
            
            # Calculate effect size (Cohen's d for paired samples)
            differences = valid_data[col1] - valid_data[col2]
            cohens_d = differences.mean() / differences.std() if differences.std() > 0 else 0
            
            # Confidence interval for mean difference
            se = differences.std() / np.sqrt(len(differences))
            t_crit = stats.t.ppf(1 - self.alpha/2, len(differences) - 1)
            ci = (differences.mean() - t_crit * se, differences.mean() + t_crit * se)
            
            # Assumption checks
            assumptions = {
                'normality_of_differences': self._check_normality(differences),
                'paired_observations': 'Assumed'
            }
            
            return {
                'test_name': 'Paired t-test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': len(differences) - 1,
                'effect_size': {'type': "Cohen's d", 'value': float(cohens_d)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': ci},
                'alpha': self.alpha,
                'decision': 'Significant difference' if p_value < self.alpha else 'No significant difference',
                'sample_sizes': {'n_pairs': len(valid_data)},
                'mean_difference': float(differences.mean()),
                'missing_count': len(df) - len(valid_data),
                'assumption_checks': assumptions,
                'visualizations': ['before_after_plot', 'difference_histogram'],
                'interpretation': f"The mean difference is {differences.mean():.3f}, which {'is' if p_value < self.alpha else 'is not'} statistically significant (p = {p_value:.4f})",
                'warnings': self._generate_test_warnings(assumptions, 'paired_ttest')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def simple_linear_regression(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """Simple linear regression"""
        try:
            # Remove missing values
            valid_data = df[[x_col, y_col]].dropna()
            
            if len(valid_data) < 3:
                return {'error': 'Insufficient data for regression'}
            
            # Prepare data
            X = sm.add_constant(valid_data[x_col])
            y = valid_data[y_col]
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Get predictions for visualization
            predictions = model.predict(X)
            residuals = y - predictions
            
            # Assumption checks
            # Normality of residuals
            normality = self._check_normality(residuals)
            
            # Homoscedasticity (constant variance)
            # Using Breusch-Pagan test
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_test = het_breuschpagan(residuals, X)
            homoscedasticity = bp_test[1] > 0.05  # p-value > 0.05 means homoscedastic
            
            assumptions = {
                'linearity': 'Visual inspection recommended',
                'normality_of_residuals': normality,
                'homoscedasticity': {'passed': homoscedasticity, 'p_value': float(bp_test[1])},
                'independence': 'Assumed'
            }
            
            return {
                'test_name': 'Simple Linear Regression',
                'statistic': 'F-statistic: ' + str(float(model.fvalue)),
                'p_value': float(model.f_pvalue),
                'df': {'model': 1, 'residual': len(valid_data) - 2},
                'effect_size': {'type': 'R-squared', 'value': float(model.rsquared)},
                'confidence_interval': {
                    'slope': model.conf_int().loc[x_col].tolist(),
                    'intercept': model.conf_int().loc['const'].tolist()
                },
                'alpha': self.alpha,
                'decision': 'Significant relationship' if model.f_pvalue < self.alpha else 'No significant relationship',
                'sample_sizes': {'n': len(valid_data)},
                'coefficients': {
                    'intercept': float(model.params['const']),
                    'slope': float(model.params[x_col])
                },
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'missing_count': len(df) - len(valid_data),
                'assumption_checks': assumptions,
                'equation': f"y = {model.params['const']:.3f} + {model.params[x_col]:.3f}*x",
                'visualizations': ['scatter_with_line', 'residual_plot'],
                'interpretation': f"A one-unit increase in {x_col} is associated with a {model.params[x_col]:.3f} change in {y_col} (R² = {model.rsquared:.3f}, p = {model.f_pvalue:.4f})",
                'warnings': self._generate_test_warnings(assumptions, 'regression')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def logistic_regression(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """Logistic regression (binary outcome)"""
        try:
            # Remove missing values
            valid_data = df[[x_col, y_col]].dropna()
            
            # Check if y is binary
            unique_y = valid_data[y_col].nunique()
            if unique_y != 2:
                return {'error': f'Outcome variable must be binary (found {unique_y} unique values)'}
            
            if len(valid_data) < 10:
                return {'error': 'Insufficient data for logistic regression'}
            
            # Prepare data
            X = sm.add_constant(valid_data[x_col])
            y = valid_data[y_col]
            
            # Encode y if necessary
            if y.dtype == 'object':
                y_values = sorted(y.unique())
                y = (y == y_values[1]).astype(int)
            
            # Fit model
            model = logit(y, X).fit(disp=False)
            
            # Calculate pseudo R-squared (McFadden's)
            null_model = logit(y, sm.add_constant(np.ones(len(y)))).fit(disp=False)
            pseudo_r2 = 1 - (model.llf / null_model.llf)
            
            return {
                'test_name': 'Logistic Regression',
                'statistic': 'LR chi2: ' + str(float(model.llr)),
                'p_value': float(model.llr_pvalue),
                'df': 1,
                'effect_size': {'type': "McFadden's Pseudo R-squared", 'value': float(pseudo_r2)},
                'confidence_interval': {
                    'slope': model.conf_int().loc[x_col].tolist() if x_col in model.conf_int().index else 'N/A',
                    'intercept': model.conf_int().loc['const'].tolist()
                },
                'alpha': self.alpha,
                'decision': 'Significant relationship' if model.llr_pvalue < self.alpha else 'No significant relationship',
                'sample_sizes': {'n': len(valid_data)},
                'coefficients': {
                    'intercept': float(model.params['const']),
                    'slope': float(model.params[x_col]) if x_col in model.params.index else 'N/A'
                },
                'odds_ratio': float(np.exp(model.params[x_col])) if x_col in model.params.index else 'N/A',
                'pseudo_r_squared': float(pseudo_r2),
                'missing_count': len(df) - len(valid_data),
                'assumption_checks': {'binary_outcome': 'Yes', 'independence': 'Assumed'},
                'visualizations': ['logistic_curve'],
                'interpretation': f"A one-unit increase in {x_col} multiplies the odds of the outcome by {np.exp(model.params[x_col]) if x_col in model.params.index else 'N/A':.3f} (p = {model.llr_pvalue:.4f})",
                'warnings': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def independent_ttest(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Independent t-test (assumes equal variances)"""
        try:
            groups = df[group_col].dropna().unique()[:2]
            group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
            group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                return {'error': 'Insufficient data in one or both groups'}
            
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=True)
            pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
            
            levene_stat, levene_p = stats.levene(group1, group2)
            assumptions = {
                'equal_variance': {'statistic': float(levene_stat), 'p_value': float(levene_p), 'passed': levene_p > 0.05},
                'normality_group1': self._check_normality(group1),
                'normality_group2': self._check_normality(group2)
            }
            
            return {
                'test_name': 'Independent t-test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': len(group1) + len(group2) - 2,
                'effect_size': {'type': "Cohen's d", 'value': float(cohens_d)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Reject H0' if p_value < self.alpha else 'Fail to reject H0',
                'assumption_checks': assumptions,
                'interpretation': self._interpret_ttest(p_value, cohens_d, groups[0], groups[1]),
                'warnings': ['Use Welch\'s t-test if variances are unequal'] if not assumptions['equal_variance']['passed'] else []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def wilcoxon_signed_rank(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Wilcoxon Signed-Rank test (non-parametric paired test)"""
        try:
            valid_data = df[[col1, col2]].dropna()
            if len(valid_data) < 5:
                return {'error': 'Insufficient paired data (need at least 5 pairs)'}
            
            statistic, p_value = stats.wilcoxon(valid_data[col1], valid_data[col2])
            r = statistic / (len(valid_data) * (len(valid_data) + 1) / 2)
            
            return {
                'test_name': 'Wilcoxon Signed-Rank Test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'Rank-biserial correlation', 'value': float(r)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Significant difference' if p_value < self.alpha else 'No significant difference',
                'sample_sizes': {'n_pairs': len(valid_data)},
                'interpretation': f"Non-parametric test shows {'significant' if p_value < self.alpha else 'no significant'} difference between paired samples (p = {p_value:.4f})",
                'notes': 'Non-parametric alternative to paired t-test'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def sign_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Sign test (non-parametric paired test)"""
        try:
            valid_data = df[[col1, col2]].dropna()
            differences = valid_data[col1] - valid_data[col2]
            differences = differences[differences != 0]
            
            if len(differences) < 5:
                return {'error': 'Insufficient non-zero differences'}
            
            n_positive = (differences > 0).sum()
            n = len(differences)
            p_value = 2 * min(stats.binom.cdf(n_positive, n, 0.5), 1 - stats.binom.cdf(n_positive-1, n, 0.5))
            
            return {
                'test_name': 'Sign Test',
                'statistic': int(n_positive),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'Proportion positive', 'value': float(n_positive/n)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Significant difference' if p_value < self.alpha else 'No significant difference',
                'sample_sizes': {'n_pairs': n, 'n_positive': int(n_positive), 'n_negative': int(n - n_positive)},
                'interpretation': f"Sign test: {n_positive}/{n} pairs favor {col1} over {col2} (p = {p_value:.4f})",
                'notes': 'Most robust non-parametric paired test'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def kendall_tau(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Kendall's Tau correlation coefficient"""
        try:
            valid_data = df[[col1, col2]].dropna()
            if len(valid_data) < 3:
                return {'error': 'Insufficient data for correlation'}
            
            tau, p_value = stats.kendalltau(valid_data[col1], valid_data[col2])
            
            return {
                'test_name': "Kendall's Tau",
                'statistic': float(tau),
                'p_value': float(p_value),
                'df': len(valid_data) - 2,
                'effect_size': {'type': "Kendall's Tau", 'value': float(tau)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Significant correlation' if p_value < self.alpha else 'No significant correlation',
                'sample_sizes': {'n': len(valid_data)},
                'interpretation': f"Kendall's Tau = {tau:.3f} (p = {p_value:.4f}). {'Significant' if p_value < self.alpha else 'No significant'} monotonic association.",
                'notes': 'Robust to outliers, better for small samples than Spearman'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def shapiro_wilk_test(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Shapiro-Wilk normality test"""
        try:
            data = df[column].dropna()
            if len(data) < 3:
                return {'error': 'Need at least 3 observations'}
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
                
            statistic, p_value = stats.shapiro(data)
            
            return {
                'test_name': 'Shapiro-Wilk Normality Test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'W statistic', 'value': float(statistic)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Data is NOT normally distributed' if p_value < self.alpha else 'Data appears normally distributed',
                'sample_sizes': {'n': len(data)},
                'interpretation': f"Shapiro-Wilk test: W = {statistic:.4f}, p = {p_value:.4f}. Data {'deviates significantly from' if p_value < self.alpha else 'is consistent with'} normal distribution.",
                'notes': 'Most powerful normality test for small to medium samples'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def ks_test(self, df: pd.DataFrame, column: str, distribution: str = 'norm') -> Dict[str, Any]:
        """Kolmogorov-Smirnov goodness-of-fit test"""
        try:
            data = df[column].dropna()
            if len(data) < 3:
                return {'error': 'Need at least 3 observations'}
            
            if distribution == 'norm':
                statistic, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                dist_name = 'Normal'
            elif distribution == 'uniform':
                statistic, p_value = stats.kstest(data, 'uniform', args=(data.min(), data.max()-data.min()))
                dist_name = 'Uniform'
            else:
                return {'error': f'Unsupported distribution: {distribution}'}
            
            return {
                'test_name': 'Kolmogorov-Smirnov Test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'KS statistic', 'value': float(statistic)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': f'Data is NOT {dist_name} distributed' if p_value < self.alpha else f'Data appears {dist_name} distributed',
                'sample_sizes': {'n': len(data)},
                'interpretation': f"KS test: D = {statistic:.4f}, p = {p_value:.4f}. Data {'deviates from' if p_value < self.alpha else 'is consistent with'} {dist_name} distribution.",
                'notes': 'General goodness-of-fit test, sensitive to differences in location and shape'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def anderson_darling_test(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Anderson-Darling normality test"""
        try:
            data = df[column].dropna()
            if len(data) < 3:
                return {'error': 'Need at least 3 observations'}
            
            result = stats.anderson(data, dist='norm')
            critical_value_5 = result.critical_values[2]
            
            return {
                'test_name': 'Anderson-Darling Normality Test',
                'statistic': float(result.statistic),
                'p_value': 'See critical values',
                'df': None,
                'effect_size': {'type': 'A² statistic', 'value': float(result.statistic)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Data is NOT normally distributed' if result.statistic > critical_value_5 else 'Data appears normally distributed',
                'sample_sizes': {'n': len(data)},
                'critical_values': {'significance_levels': result.significance_level.tolist(), 'critical_values': result.critical_values.tolist()},
                'interpretation': f"Anderson-Darling: A² = {result.statistic:.4f}. Data {'deviates from' if result.statistic > critical_value_5 else 'is consistent with'} normal distribution (at 5% level).",
                'notes': 'More sensitive to tails than Shapiro-Wilk'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def chi_square_gof(self, df: pd.DataFrame, column: str, expected_freq: Optional[List[float]] = None) -> Dict[str, Any]:
        """Chi-square goodness-of-fit test"""
        try:
            observed = df[column].value_counts().sort_index()
            
            if expected_freq is None:
                expected_freq = [len(df[column]) / len(observed)] * len(observed)
            
            if len(observed) != len(expected_freq):
                return {'error': 'Length of expected frequencies must match number of categories'}
            
            chi2, p_value = stats.chisquare(observed.values, expected_freq)
            
            return {
                'test_name': 'Chi-Square Goodness-of-Fit Test',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'df': len(observed) - 1,
                'effect_size': {'type': 'Chi-square', 'value': float(chi2)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Observed distribution differs from expected' if p_value < self.alpha else 'Observed distribution matches expected',
                'sample_sizes': {'n': int(observed.sum())},
                'observed_frequencies': observed.to_dict(),
                'expected_frequencies': dict(zip(observed.index, expected_freq)),
                'interpretation': f"Chi-square GOF: χ² = {chi2:.2f}, p = {p_value:.4f}. Observed distribution {'differs significantly from' if p_value < self.alpha else 'matches'} expected distribution.",
                'notes': 'Tests if categorical data follows expected distribution'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def levene_test(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Levene's test for equality of variances"""
        try:
            valid_categories = df[group_col].dropna().unique()
            groups = [df[df[group_col] == cat][numeric_col].dropna() for cat in valid_categories]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups'}
            
            statistic, p_value = stats.levene(*groups)
            
            return {
                'test_name': "Levene's Test for Equality of Variances",
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': {'between': len(groups) - 1, 'within': sum(len(g) for g in groups) - len(groups)},
                'effect_size': {'type': 'Levene statistic', 'value': float(statistic)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Variances are NOT equal' if p_value < self.alpha else 'Variances appear equal',
                'sample_sizes': {str(i): len(g) for i, g in enumerate(groups)},
                'interpretation': f"Levene's test: W = {statistic:.4f}, p = {p_value:.4f}. Variances {'differ significantly' if p_value < self.alpha else 'do not differ significantly'} across groups.",
                'notes': 'Used to test homogeneity of variance assumption for ANOVA'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def bartlett_test(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Bartlett's test for equality of variances"""
        try:
            valid_categories = df[group_col].dropna().unique()
            groups = [df[df[group_col] == cat][numeric_col].dropna() for cat in valid_categories]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups'}
            
            statistic, p_value = stats.bartlett(*groups)
            
            return {
                'test_name': "Bartlett's Test for Equality of Variances",
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': len(groups) - 1,
                'effect_size': {'type': 'Bartlett statistic', 'value': float(statistic)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Variances are NOT equal' if p_value < self.alpha else 'Variances appear equal',
                'sample_sizes': {str(i): len(g) for i, g in enumerate(groups)},
                'interpretation': f"Bartlett's test: T = {statistic:.4f}, p = {p_value:.4f}. Variances {'differ significantly' if p_value < self.alpha else 'do not differ significantly'} across groups.",
                'notes': 'More sensitive to normality than Levene test',
                'warnings': ['Assumes normal distributions'] if not all(self._check_normality(g) for g in groups if len(g) > 2) else []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def mcnemar_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """McNemar's test for paired nominal data"""
        try:
            contingency = pd.crosstab(df[col1], df[col2])
            
            if contingency.shape != (2, 2):
                return {'error': 'McNemar test requires 2x2 contingency table'}
            
            b = contingency.iloc[0, 1]
            c = contingency.iloc[1, 0]
            
            if b + c < 25:
                statistic = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
            else:
                statistic = (b - c)**2 / (b + c) if (b + c) > 0 else 0
            
            p_value = 1 - stats.chi2.cdf(statistic, 1)
            
            return {
                'test_name': "McNemar's Test",
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': 1,
                'effect_size': {'type': 'Odds ratio', 'value': float(b/c) if c > 0 else float('inf')},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Significant change' if p_value < self.alpha else 'No significant change',
                'sample_sizes': {'n': int(contingency.sum().sum()), 'discordant_pairs': int(b + c)},
                'contingency_table': contingency.to_dict(),
                'interpretation': f"McNemar's test: χ² = {statistic:.2f}, p = {p_value:.4f}. {'Significant' if p_value < self.alpha else 'No significant'} change in proportions.",
                'notes': 'Used for paired nominal data (before/after designs)'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def ks_two_sample(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
        """Kolmogorov-Smirnov two-sample test"""
        try:
            groups = df[group_col].dropna().unique()[:2]
            group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
            group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                return {'error': 'Insufficient data in one or both groups'}
            
            statistic, p_value = stats.ks_2samp(group1, group2)
            
            return {
                'test_name': 'Kolmogorov-Smirnov Two-Sample Test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'KS statistic', 'value': float(statistic)},
                'confidence_interval': {'level': 'N/A', 'interval': 'N/A'},
                'alpha': self.alpha,
                'decision': 'Distributions differ' if p_value < self.alpha else 'Distributions do not differ significantly',
                'sample_sizes': {'group1': len(group1), 'group2': len(group2)},
                'interpretation': f"KS two-sample: D = {statistic:.4f}, p = {p_value:.4f}. Distributions {'differ significantly' if p_value < self.alpha else 'do not differ significantly'}.",
                'notes': 'Tests if two samples come from same distribution'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def one_sample_proportion_ztest(self, df: pd.DataFrame, column: str, success_value: Any, test_proportion: float = 0.5) -> Dict[str, Any]:
        """One-sample Z-test for proportion"""
        try:
            data = df[column].dropna()
            successes = (data == success_value).sum()
            n = len(data)
            
            if n < 30:
                return {'error': 'Sample size too small (need n >= 30)'}
            
            p_hat = successes / n
            z_stat = (p_hat - test_proportion) / np.sqrt(test_proportion * (1 - test_proportion) / n)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return {
                'test_name': 'One-Sample Proportion Z-Test',
                'statistic': float(z_stat),
                'p_value': float(p_value),
                'df': None,
                'effect_size': {'type': 'Difference from null', 'value': float(p_hat - test_proportion)},
                'confidence_interval': {'level': f'{(1-self.alpha)*100}%', 'interval': (float(p_hat - 1.96*np.sqrt(p_hat*(1-p_hat)/n)), float(p_hat + 1.96*np.sqrt(p_hat*(1-p_hat)/n)))},
                'alpha': self.alpha,
                'decision': f'Proportion differs from {test_proportion}' if p_value < self.alpha else f'Proportion does not differ from {test_proportion}',
                'sample_sizes': {'n': n, 'successes': int(successes)},
                'sample_proportion': float(p_hat),
                'test_proportion': float(test_proportion),
                'interpretation': f"Observed proportion = {p_hat:.3f}, test value = {test_proportion}. {'Significant difference' if p_value < self.alpha else 'No significant difference'} (z = {z_stat:.2f}, p = {p_value:.4f}).",
                'notes': 'Requires large sample size for normal approximation'
            }
        except Exception as e:
            return {'error': str(e)}

    # Helper methods
    def _infer_type(self, series: pd.Series) -> str:
        """Infer if column is numeric or categorical"""
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        else:
            return 'categorical'
    
    def _check_normality(self, data: pd.Series, alpha: float = 0.05) -> bool:
        """Check normality using Shapiro-Wilk test"""
        if len(data) < 3:
            return False
        
        # Use Shapiro-Wilk test (limit to 5000 samples for performance)
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        
        try:
            _, p_value = stats.shapiro(data)
            return p_value > alpha
        except:
            return False
    
    def _interpret_ttest(self, p_value: float, cohens_d: float, group1: str, group2: str) -> str:
        """Interpret t-test results"""
        if p_value < self.alpha:
            effect_size_interpretation = (
                'large' if abs(cohens_d) >= 0.8 else
                'medium' if abs(cohens_d) >= 0.5 else
                'small'
            )
            direction = 'higher' if cohens_d > 0 else 'lower'
            return f"Groups differ significantly (p = {p_value:.4f}). {group1} is {direction} than {group2} with a {effect_size_interpretation} effect size (d = {cohens_d:.3f})."
        else:
            return f"No significant difference between groups (p = {p_value:.4f})."
    
    def _interpret_correlation(self, r: float, p_value: float, correlation_type: str = 'linear') -> str:
        """Interpret correlation results"""
        strength = (
            'very strong' if abs(r) >= 0.8 else
            'strong' if abs(r) >= 0.6 else
            'moderate' if abs(r) >= 0.4 else
            'weak' if abs(r) >= 0.2 else
            'very weak'
        )
        direction = 'positive' if r > 0 else 'negative'
        
        if p_value < self.alpha:
            return f"A {strength} {direction} {correlation_type} correlation exists (r = {r:.3f}, p = {p_value:.4f})."
        else:
            return f"No significant correlation found (r = {r:.3f}, p = {p_value:.4f})."
    
    def _generate_test_warnings(self, assumptions: Dict, test_type: str) -> List[str]:
        """Generate warnings based on assumption violations"""
        warnings = []
        
        if test_type in ['welch_ttest', 'paired_ttest']:
            if not assumptions.get('normality_group1', True) or not assumptions.get('normality_group2', True):
                warnings.append('Normality assumption violated. Consider Mann-Whitney U test.')
            if assumptions.get('sample_sizes', {}).get('group1', 0) < 30 or assumptions.get('sample_sizes', {}).get('group2', 0) < 30:
                warnings.append('Small sample size. Results should be interpreted with caution.')
        
        elif test_type == 'pearson':
            if not assumptions.get('normality_col1', True) or not assumptions.get('normality_col2', True):
                warnings.append('Normality assumption violated. Consider Spearman correlation.')
        
        elif test_type == 'anova':
            if not assumptions.get('homogeneity_of_variance', {}).get('passed', True):
                warnings.append('Homogeneity of variance assumption violated. Consider Welch ANOVA or Kruskal-Wallis test.')
        
        elif test_type == 'regression':
            if not assumptions.get('normality_of_residuals', True):
                warnings.append('Residuals are not normally distributed. Consider data transformation.')
            if not assumptions.get('homoscedasticity', {}).get('passed', True):
                warnings.append('Heteroscedasticity detected. Consider weighted least squares or robust standard errors.')
        
        return warnings
