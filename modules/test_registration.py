"""
Register all statistical tests with metadata and validation
"""
import pandas as pd
from typing import Dict, Any
from modules.test_registry import TEST_REGISTRY, TestMetadata
from modules.hypothesis_analysis import HypothesisAnalyzer

def validate_one_sample_ttest(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for one-sample t-test"""
    if 'column' not in params:
        return {'valid': False, 'error': 'Please select a numeric column'}
    if 'test_value' not in params:
        return {'valid': False, 'error': 'Please provide a test value'}
    
    column = params['column']
    if column not in df.columns:
        return {'valid': False, 'error': f'Column {column} not found in dataset'}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {'valid': False, 'error': f'Column {column} must be numeric'}
    if df[column].dropna().empty:
        return {'valid': False, 'error': f'Column {column} has no valid data'}
    
    return {'valid': True}

def validate_two_group_test(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for two-group tests (t-test, Mann-Whitney, etc.)"""
    if 'numeric_col' not in params or 'group_col' not in params:
        return {'valid': False, 'error': 'Please select both numeric and grouping columns'}
    
    numeric_col, group_col = params['numeric_col'], params['group_col']
    
    if numeric_col not in df.columns or group_col not in df.columns:
        return {'valid': False, 'error': 'Selected columns not found in dataset'}
    if not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return {'valid': False, 'error': f'{numeric_col} must be numeric'}
    
    valid_groups = df[group_col].dropna().unique()
    n_groups = len(valid_groups)
    if n_groups < 2:
        return {'valid': False, 'error': f'{group_col} must have at least 2 groups (found {n_groups})'}
    if n_groups > 2:
        return {'valid': False, 'error': f'{group_col} has {n_groups} groups. This test requires exactly 2 groups. Consider using ANOVA for 3+ groups.'}
    
    for group in valid_groups[:2]:
        group_data = df[df[group_col] == group][numeric_col].dropna()
        if len(group_data) < 2:
            return {'valid': False, 'error': f'Group "{group}" has insufficient data (need at least 2 observations)'}
    
    return {'valid': True}

def validate_anova_test(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for ANOVA/Kruskal-Wallis tests"""
    if 'numeric_col' not in params or 'group_col' not in params:
        return {'valid': False, 'error': 'Please select both numeric and grouping columns'}
    
    numeric_col, group_col = params['numeric_col'], params['group_col']
    
    if numeric_col not in df.columns or group_col not in df.columns:
        return {'valid': False, 'error': 'Selected columns not found in dataset'}
    if not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return {'valid': False, 'error': f'{numeric_col} must be numeric'}
    
    valid_groups = df[group_col].dropna().unique()
    n_groups = len(valid_groups)
    if n_groups < 2:
        return {'valid': False, 'error': f'{group_col} must have at least 2 groups'}
    
    for group in valid_groups:
        group_data = df[df[group_col] == group][numeric_col].dropna()
        if len(group_data) < 2:
            return {'valid': False, 'error': f'Group "{group}" has insufficient data'}
    
    return {'valid': True}

def validate_correlation_test(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for correlation tests"""
    if 'col1' not in params or 'col2' not in params:
        return {'valid': False, 'error': 'Please select two columns'}
    
    col1, col2 = params['col1'], params['col2']
    
    if col1 not in df.columns or col2 not in df.columns:
        return {'valid': False, 'error': 'Selected columns not found in dataset'}
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {'valid': False, 'error': 'Both columns must be numeric'}
    
    valid_data = df[[col1, col2]].dropna()
    if len(valid_data) < 3:
        return {'valid': False, 'error': 'Need at least 3 complete pairs of observations'}
    
    return {'valid': True}

def validate_paired_test(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for paired tests"""
    if 'col1' not in params or 'col2' not in params:
        return {'valid': False, 'error': 'Please select two columns for paired comparison'}
    
    col1, col2 = params['col1'], params['col2']
    
    if col1 not in df.columns or col2 not in df.columns:
        return {'valid': False, 'error': 'Selected columns not found in dataset'}
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {'valid': False, 'error': 'Both columns must be numeric for paired comparison'}
    
    valid_data = df[[col1, col2]].dropna()
    if len(valid_data) < 2:
        return {'valid': False, 'error': 'Need at least 2 complete pairs of observations'}
    
    return {'valid': True}

def validate_categorical_test(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for chi-square and Fisher's exact tests"""
    if 'col1' not in params or 'col2' not in params:
        return {'valid': False, 'error': 'Please select two categorical columns'}
    
    col1, col2 = params['col1'], params['col2']
    
    if col1 not in df.columns or col2 not in df.columns:
        return {'valid': False, 'error': 'Selected columns not found in dataset'}
    
    if df[[col1, col2]].dropna().empty:
        return {'valid': False, 'error': 'No valid data after removing missing values'}
    
    return {'valid': True}

def validate_single_column_test(df: pd.DataFrame, **params) -> Dict[str, Any]:
    """Validate inputs for single-column tests (normality, GOF, etc.)"""
    if 'column' not in params:
        return {'valid': False, 'error': 'Please select a column'}
    
    column = params['column']
    if column not in df.columns:
        return {'valid': False, 'error': f'Column {column} not found in dataset'}
    if df[column].dropna().empty:
        return {'valid': False, 'error': f'Column {column} has no valid data'}
    
    return {'valid': True}

_shared_analyzer = HypothesisAnalyzer()

def get_analyzer() -> HypothesisAnalyzer:
    """Get the shared analyzer instance"""
    return _shared_analyzer

def register_all_tests():
    """Register all statistical tests with the TEST_REGISTRY"""
    analyzer = _shared_analyzer
    
    # Parametric Tests - Comparison
    TEST_REGISTRY.register(TestMetadata(
        test_id='one_sample_ttest',
        name='One-Sample t-test',
        category='parametric',
        subcategory='comparison',
        description='Tests if a sample mean differs from a hypothesized population mean',
        assumptions=['Normal distribution', 'Independent observations'],
        input_requirements={'numeric_cols': 1, 'test_value': True},
        validator=validate_one_sample_ttest,
        executor=lambda df, **p: analyzer.one_sample_ttest(df, p['column'], p.get('test_value', 0)),
        example_use_case='Test if average customer age differs from 35 years'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='independent_ttest',
        name='Independent t-test',
        category='parametric',
        subcategory='comparison',
        description='Compares means of two independent groups (assumes equal variances)',
        assumptions=['Normal distribution', 'Equal variances', 'Independent groups'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'groups': 2},
        validator=validate_two_group_test,
        executor=lambda df, **p: analyzer.independent_ttest(df, p['numeric_col'], p['group_col']),
        example_use_case='Compare test scores between treatment and control groups'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='welch_ttest',
        name="Welch's t-test",
        category='parametric',
        subcategory='comparison',
        description='Compares means of two independent groups (robust to unequal variances)',
        assumptions=['Normal distribution', 'Independent groups'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'groups': 2},
        validator=validate_two_group_test,
        executor=lambda df, **p: analyzer.welch_ttest(df, p['numeric_col'], p['group_col']),
        example_use_case='Compare salaries between two departments with different sample sizes'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='paired_ttest',
        name='Paired t-test',
        category='parametric',
        subcategory='comparison',
        description='Compares means of two related samples (before/after, matched pairs)',
        assumptions=['Normal distribution of differences', 'Paired observations'],
        input_requirements={'numeric_cols': 2, 'paired': True},
        validator=validate_paired_test,
        executor=lambda df, **p: analyzer.paired_ttest(df, p['col1'], p['col2']),
        example_use_case='Compare blood pressure before and after treatment'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='one_way_anova',
        name='One-Way ANOVA',
        category='parametric',
        subcategory='comparison',
        description='Compares means across 3 or more independent groups',
        assumptions=['Normal distribution', 'Equal variances', 'Independent groups'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'min_groups': 2},
        validator=validate_anova_test,
        executor=lambda df, **p: analyzer.one_way_anova(df, p['numeric_col'], p['group_col']),
        example_use_case='Compare average sales across 5 different regions'
    ))
    
    # Parametric Tests - Correlation
    TEST_REGISTRY.register(TestMetadata(
        test_id='pearson_correlation',
        name='Pearson Correlation',
        category='parametric',
        subcategory='correlation',
        description='Measures linear relationship between two continuous variables',
        assumptions=['Normal distribution', 'Linear relationship', 'Homoscedasticity'],
        input_requirements={'numeric_cols': 2},
        validator=validate_correlation_test,
        executor=lambda df, **p: analyzer.pearson_correlation(df, p['col1'], p['col2']),
        example_use_case='Measure relationship between study hours and exam scores'
    ))
    
    # Parametric Tests - Variance
    TEST_REGISTRY.register(TestMetadata(
        test_id='levene_test',
        name="Levene's Test",
        category='parametric',
        subcategory='variance',
        description='Tests equality of variances across groups',
        assumptions=['Independent groups'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'min_groups': 2},
        validator=validate_anova_test,
        executor=lambda df, **p: analyzer.levene_test(df, p['numeric_col'], p['group_col']),
        example_use_case='Check if product quality variance is consistent across factories'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='bartlett_test',
        name="Bartlett's Test",
        category='parametric',
        subcategory='variance',
        description='Tests equality of variances (sensitive to normality)',
        assumptions=['Normal distribution', 'Independent groups'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'min_groups': 2},
        validator=validate_anova_test,
        executor=lambda df, **p: analyzer.bartlett_test(df, p['numeric_col'], p['group_col']),
        example_use_case='Test if measurement variance is equal across laboratories'
    ))
    
    # Parametric Tests - Goodness of Fit
    TEST_REGISTRY.register(TestMetadata(
        test_id='shapiro_wilk',
        name='Shapiro-Wilk Test',
        category='parametric',
        subcategory='goodness_of_fit',
        description='Tests if data follows a normal distribution',
        assumptions=[],
        input_requirements={'numeric_cols': 1},
        validator=validate_single_column_test,
        executor=lambda df, **p: analyzer.shapiro_wilk_test(df, p['column']),
        example_use_case='Verify if test scores are normally distributed'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='ks_test',
        name='Kolmogorov-Smirnov Test',
        category='parametric',
        subcategory='goodness_of_fit',
        description='Tests if data follows a specified distribution',
        assumptions=[],
        input_requirements={'numeric_cols': 1},
        validator=validate_single_column_test,
        executor=lambda df, **p: analyzer.ks_test(df, p['column'], p.get('distribution', 'norm')),
        example_use_case='Test if data follows uniform distribution'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='anderson_darling',
        name='Anderson-Darling Test',
        category='parametric',
        subcategory='goodness_of_fit',
        description='Tests normality (more sensitive to tails than Shapiro-Wilk)',
        assumptions=[],
        input_requirements={'numeric_cols': 1},
        validator=validate_single_column_test,
        executor=lambda df, **p: analyzer.anderson_darling_test(df, p['column']),
        example_use_case='Check normality with focus on extreme values'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='chi_square_gof',
        name='Chi-Square Goodness-of-Fit',
        category='parametric',
        subcategory='goodness_of_fit',
        description='Tests if categorical data matches expected distribution',
        assumptions=['Expected frequency ≥ 5 per category'],
        input_requirements={'categorical_cols': 1},
        validator=validate_single_column_test,
        executor=lambda df, **p: analyzer.chi_square_gof(df, p['column'], p.get('expected_freq')),
        example_use_case='Test if dice rolls are fair'
    ))
    
    # Non-Parametric Tests - Comparison
    TEST_REGISTRY.register(TestMetadata(
        test_id='mann_whitney',
        name='Mann-Whitney U Test',
        category='non_parametric',
        subcategory='comparison',
        description='Compares distributions of two independent groups (non-parametric)',
        assumptions=['Independent observations', 'Ordinal or continuous data'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'groups': 2},
        validator=validate_two_group_test,
        executor=lambda df, **p: analyzer.mann_whitney(df, p['numeric_col'], p['group_col']),
        example_use_case='Compare customer satisfaction ratings between two stores'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='wilcoxon_signed_rank',
        name='Wilcoxon Signed-Rank Test',
        category='non_parametric',
        subcategory='comparison',
        description='Compares two related samples (non-parametric paired test)',
        assumptions=['Paired observations', 'Ordinal or continuous data'],
        input_requirements={'numeric_cols': 2, 'paired': True},
        validator=validate_paired_test,
        executor=lambda df, **p: analyzer.wilcoxon_signed_rank(df, p['col1'], p['col2']),
        example_use_case='Compare pain scores before and after treatment'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='sign_test',
        name='Sign Test',
        category='non_parametric',
        subcategory='comparison',
        description='Most robust non-parametric test for paired data',
        assumptions=['Paired observations'],
        input_requirements={'numeric_cols': 2, 'paired': True},
        validator=validate_paired_test,
        executor=lambda df, **p: analyzer.sign_test(df, p['col1'], p['col2']),
        example_use_case='Determine if one brand is preferred over another'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='kruskal_wallis',
        name='Kruskal-Wallis Test',
        category='non_parametric',
        subcategory='comparison',
        description='Compares distributions across 3+ groups (non-parametric ANOVA)',
        assumptions=['Independent observations', 'Ordinal or continuous data'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'min_groups': 2},
        validator=validate_anova_test,
        executor=lambda df, **p: analyzer.kruskal_wallis(df, p['numeric_col'], p['group_col']),
        example_use_case='Compare median income across multiple job categories'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='ks_two_sample',
        name='Kolmogorov-Smirnov Two-Sample Test',
        category='non_parametric',
        subcategory='comparison',
        description='Tests if two samples come from the same distribution',
        assumptions=['Independent observations'],
        input_requirements={'numeric_cols': 1, 'categorical_cols': 1, 'groups': 2},
        validator=validate_two_group_test,
        executor=lambda df, **p: analyzer.ks_two_sample(df, p['numeric_col'], p['group_col']),
        example_use_case='Compare distributions of response times between two systems'
    ))
    
    # Non-Parametric Tests - Correlation
    TEST_REGISTRY.register(TestMetadata(
        test_id='spearman_correlation',
        name='Spearman Correlation',
        category='non_parametric',
        subcategory='correlation',
        description='Measures monotonic relationship between variables',
        assumptions=['Ordinal or continuous data'],
        input_requirements={'numeric_cols': 2},
        validator=validate_correlation_test,
        executor=lambda df, **p: analyzer.spearman_correlation(df, p['col1'], p['col2']),
        example_use_case='Measure relationship between rankings or ordinal scales'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='kendall_tau',
        name="Kendall's Tau",
        category='non_parametric',
        subcategory='correlation',
        description='Robust correlation for small samples or with outliers',
        assumptions=['Ordinal or continuous data'],
        input_requirements={'numeric_cols': 2},
        validator=validate_correlation_test,
        executor=lambda df, **p: analyzer.kendall_tau(df, p['col1'], p['col2']),
        example_use_case='Measure agreement between two raters'
    ))
    
    # Categorical Tests
    TEST_REGISTRY.register(TestMetadata(
        test_id='chi_square',
        name='Chi-Square Test of Independence',
        category='non_parametric',
        subcategory='categorical',
        description='Tests if two categorical variables are independent',
        assumptions=['Expected frequency ≥ 5 in most cells'],
        input_requirements={'categorical_cols': 2},
        validator=validate_categorical_test,
        executor=lambda df, **p: analyzer.chi_square(df, p['col1'], p['col2']),
        example_use_case='Test if gender is associated with product preference'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='fisher_exact',
        name="Fisher's Exact Test",
        category='non_parametric',
        subcategory='categorical',
        description='Tests independence in 2x2 tables (exact, for small samples)',
        assumptions=['2x2 contingency table'],
        input_requirements={'categorical_cols': 2, 'table_size': '2x2'},
        validator=validate_categorical_test,
        executor=lambda df, **p: analyzer.fisher_exact(df, p['col1'], p['col2']),
        example_use_case='Test association with small sample sizes'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='mcnemar',
        name="McNemar's Test",
        category='non_parametric',
        subcategory='categorical',
        description='Tests change in paired nominal data (before/after)',
        assumptions=['Paired observations', '2x2 table'],
        input_requirements={'categorical_cols': 2, 'paired': True},
        validator=validate_categorical_test,
        executor=lambda df, **p: analyzer.mcnemar_test(df, p['col1'], p['col2']),
        example_use_case='Test if opinion changed after intervention'
    ))
    
    # Proportion Tests
    TEST_REGISTRY.register(TestMetadata(
        test_id='two_proportion_ztest',
        name='Two-Proportion Z-Test',
        category='parametric',
        subcategory='proportion',
        description='Compares proportions between two groups',
        assumptions=['Large sample size', 'Independent groups'],
        input_requirements={'manual_input': True},
        validator=lambda df, **p: {'valid': True},
        executor=lambda df, **p: analyzer.two_proportion_ztest(p.get('successes', []), p.get('totals', [])),
        example_use_case='Compare success rates between two marketing campaigns'
    ))
    
    TEST_REGISTRY.register(TestMetadata(
        test_id='one_sample_proportion',
        name='One-Sample Proportion Z-Test',
        category='parametric',
        subcategory='proportion',
        description='Tests if a sample proportion differs from a hypothesized value',
        assumptions=['Large sample size (n ≥ 30)'],
        input_requirements={'categorical_cols': 1, 'test_proportion': True},
        validator=validate_single_column_test,
        executor=lambda df, **p: analyzer.one_sample_proportion_ztest(df, p['column'], p.get('success_value'), p.get('test_proportion', 0.5)),
        example_use_case='Test if conversion rate differs from 10%'
    ))

# Auto-register on import
register_all_tests()
