import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state, create_backup, save_cleaning_operation, undo_last_operation, redo_last_operation
from modules.data_analyzer import ColumnAnalyzer
from modules.cleaning_engine import DataCleaningEngine
from modules.visualization import DataVisualizer
from modules.ai_assistant import AIAssistant
from modules.survey_weights import SurveyWeightsManager

# Initialize session state


st.title("🧙‍♀️ Data Cleaning Wizard with Integrated Weights")

st.markdown("""
Apply comprehensive data cleaning operations with integrated survey weights. All cleaning methods 
consider survey weights when calculating impact assessments and provide both weighted and unweighted statistics.
""")

# Check if dataset is loaded
if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset.copy()
cleaning_engine = DataCleaningEngine()
visualizer = DataVisualizer()
analyzer = st.session_state.data_analyzer

# Get weights manager from session state
weights_manager = st.session_state.get('weights_manager')

# Display weights status
if weights_manager and weights_manager.weights_column:
    st.info(f"⚖️ **Survey weights active:** Using '{weights_manager.weights_column}' for weighted analysis")
else:
    st.info("📊 **Unweighted analysis:** No survey weights configured")

# Check for inter-column violations before cleaning
if 'inter_column_violations' not in st.session_state:
    with st.spinner("🔍 Checking for inter-column rule violations..."):
        violations = analyzer.detect_inter_column_violations(df)
        st.session_state.inter_column_violations = violations

# Display violations if any
violations = st.session_state.get('inter_column_violations', {})
if violations.get('total_violations', 0) > 0:
    severity_emoji = {'low': '🟢', 'moderate': '🟡', 'high': '🔴'}[violations['severity']]
    st.warning(f"{severity_emoji} **{violations['total_violations']} inter-column violations detected** - Address these during cleaning for data consistency")

# Control panel
control_cols = st.columns([2, 1, 1, 1])

with control_cols[0]:
    st.subheader("🎛️ Cleaning Controls")

with control_cols[1]:
    if st.button("↶ Undo", help="Undo last cleaning operation"):
        if undo_last_operation():
            st.success("✅ Operation undone")
            st.rerun()
        else:
            st.warning("No operations to undo")

with control_cols[2]:
    if st.button("↷ Redo", help="Redo last undone operation"):
        if redo_last_operation():
            st.success("✅ Operation redone")
            st.rerun()
        else:
            st.warning("No operations to redo")

with control_cols[3]:
    undo_available = len(st.session_state.get('undo_stack', []))
    redo_available = len(st.session_state.get('redo_stack', []))
    st.caption(f"Undo: {undo_available} | Redo: {redo_available}")

# Survey Weights Configuration Section
st.subheader("1. Survey Weights Configuration")

st.markdown("""
Configure survey weights to account for sampling design effects. All cleaning operations will 
provide both weighted and unweighted impact assessments.
""")

# Weights configuration
weights_cols = st.columns([2, 2])

with weights_cols[0]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    weights_col = st.selectbox(
        "Select weights column:",
        options=['None'] + numeric_columns,
        key="weights_selection",
        help="Choose the column containing survey design weights, or select 'None' for unweighted analysis"
    )

with weights_cols[1]:
    if weights_col != 'None':
        if st.button("Configure Weights", type="primary"):
            try:
                # Always initialize a fresh weights manager for configuration
                st.session_state.weights_manager = SurveyWeightsManager()
                
                validation_result = st.session_state.weights_manager.set_weights_column(df, weights_col)
                
                if validation_result['valid']:
                    st.success(f"✅ Weights column '{weights_col}' configured successfully!")
                    
                    if validation_result['warnings']:
                        st.warning("⚠️ Warnings:")
                        for warning in validation_result['warnings']:
                            st.write(f"• {warning}")
                else:
                    st.error("❌ Invalid weights column:")
                    for error in validation_result['errors']:
                        st.write(f"• {error}")
                        
            except Exception as e:
                st.error(f"❌ Error configuring weights: {str(e)}")
    else:
        # Clear weights configuration if 'None' selected
        if 'weights_manager' in st.session_state:
            st.session_state.weights_manager = None
        st.info("ℹ️ Proceeding with unweighted analysis")

# Display current weights status
weights_manager = st.session_state.get('weights_manager')
if weights_manager and weights_manager.weights_column:
    weights_data = df[weights_manager.weights_column].dropna()
    
    # Show weights diagnostics
    diagnostics_cols = st.columns(4)
    with diagnostics_cols[0]:
        st.metric("Mean Weight", f"{weights_data.mean():.3f}")
    with diagnostics_cols[1]:
        st.metric("Weight Range", f"{weights_data.min():.2f} - {weights_data.max():.2f}")
    with diagnostics_cols[2]:
        # Calculate design effect estimate
        deff = (weights_data.sum()**2) / (len(weights_data) * (weights_data**2).sum())
        st.metric("Design Effect", f"{deff:.2f}")
    with diagnostics_cols[3]:
        effective_n = len(weights_data) / deff
        st.metric("Effective N", f"{effective_n:.0f}")

# Column selection and method selection
st.subheader("2. Select Column and Cleaning Method")

selection_cols = st.columns([2, 2])

with selection_cols[0]:
    selected_column = st.selectbox(
        "Choose column to clean:",
        options=[''] + list(df.columns),
        help="Select the column you want to clean"
    )

with selection_cols[1]:
    if selected_column:
        column_type = st.session_state.column_types.get(selected_column, 'unknown')
        st.info(f"**Column Type:** {column_type.title()}")
        
        # Show basic stats
        series = df[selected_column]
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / len(series)) * 100
        
        if missing_count > 0:
            st.warning(f"⚠️ {missing_count} missing values ({missing_pct:.1f}%)")
        
        if pd.api.types.is_numeric_dtype(series):
            outlier_info = ""
            try:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    outlier_info = f"⚡ {len(outliers)} potential outliers detected"
                    st.warning(outlier_info)
            except:
                pass

if not selected_column:
    st.info("👆 Please select a column to start cleaning.")
    st.stop()

# Get analysis if available
analysis = st.session_state.column_analysis.get(selected_column, {})
if not analysis:
    st.info("ℹ️ No analysis available for this column. Running quick analysis...")
    analyzer = ColumnAnalyzer()
    try:
        analysis = analyzer.analyze_column(df, selected_column)
        st.session_state.column_analysis[selected_column] = analysis
    except Exception as e:
        st.error(f"Error analyzing column: {str(e)}")
        analysis = {}

# Method selection based on available cleaning methods
st.subheader("2. Choose Cleaning Method")

method_type = st.selectbox(
    "Select cleaning category:",
    options=['missing_values', 'outliers', 'data_quality'],
    format_func=lambda x: {
        'missing_values': '❌ Missing Values',
        'outliers': '⚡ Outliers',
        'data_quality': '🔧 Data Quality'
    }[x]
)

# Get available methods for the selected type
available_methods = {
    'missing_values': {
        'mean_imputation': '📊 Mean Imputation (numeric only)',
        'median_imputation': '📊 Median Imputation (numeric only)',
        'mode_imputation': '📊 Mode Imputation (categorical)',
        'forward_fill': '➡️ Forward Fill',
        'backward_fill': '⬅️ Backward Fill',
        'knn_imputation': '🎯 KNN Imputation (numeric only)',
        'interpolation': '📈 Interpolation (numeric only)',
        'missing_category': '📁 Create Missing Category',
        'regression_imputation': '🔬 Regression Imputation (numeric only)'
    },
    'outliers': {
        'iqr_removal': '📦 IQR Method Removal',
        'zscore_removal': '📊 Z-Score Removal',
        'winsorization': '✂️ Winsorization',
        'log_transformation': '📈 Log Transformation',
        'cap_outliers': '🧢 Cap Outliers',
        'isolation_forest': '🌲 Isolation Forest'
    },
    'data_quality': {
        'type_standardization': '🔄 Type Standardization',
        'remove_duplicates': '🔄 Remove Duplicates',
        'trim_whitespace': '✂️ Trim Whitespace',
        'standardize_case': '🔤 Standardize Case'
    }
}

method_name = st.selectbox(
    "Select specific method:",
    options=list(available_methods[method_type].keys()),
    format_func=lambda x: available_methods[method_type][x]
)

# Method parameters
st.subheader("3. Method Parameters")

parameters = {}

# Parameter inputs based on method
if method_name == 'knn_imputation':
    parameters['n_neighbors'] = st.slider("Number of neighbors", 3, 10, 5)
elif method_name == 'interpolation':
    parameters['method'] = st.selectbox("Interpolation method", ['linear', 'polynomial', 'spline'])
elif method_name == 'missing_category':
    parameters['category_name'] = st.text_input("Category name for missing values", 'Missing')
elif method_name == 'winsorization':
    col1, col2 = st.columns(2)
    with col1:
        parameters['lower_percentile'] = st.slider("Lower percentile", 0.1, 10.0, 5.0)
    with col2:
        parameters['upper_percentile'] = st.slider("Upper percentile", 90.0, 99.9, 95.0)
elif method_name == 'iqr_removal':
    parameters['multiplier'] = st.slider("IQR multiplier", 1.0, 3.0, 1.5)
elif method_name == 'zscore_removal':
    parameters['threshold'] = st.slider("Z-score threshold", 2.0, 4.0, 3.0)
elif method_name == 'cap_outliers':
    parameters['method'] = st.selectbox("Capping method", ['iqr', 'percentile'])
    if parameters['method'] == 'iqr':
        parameters['multiplier'] = st.slider("IQR multiplier", 1.0, 3.0, 1.5)
elif method_name == 'type_standardization':
    parameters['target_type'] = st.selectbox("Target type", ['numeric', 'string', 'datetime'])
elif method_name == 'standardize_case':
    parameters['case_type'] = st.selectbox("Case type", ['lower', 'upper', 'title'])
elif method_name == 'isolation_forest':
    parameters['contamination'] = st.slider("Contamination rate", 0.01, 0.2, 0.1)

# Preview section
st.subheader("4. Preview and Apply")

preview_cols = st.columns([1, 1])

with preview_cols[0]:
    if st.button("👁️ Preview Changes", type="secondary", width='stretch'):
        try:
            # Apply method to get preview
            cleaned_series, metadata = cleaning_engine.apply_cleaning_method(
                df, selected_column, method_type, method_name, parameters
            )
            
            if metadata['success']:
                # Store preview results
                st.session_state.preview_results = {
                    'original': df[selected_column].copy(),
                    'cleaned': cleaned_series,
                    'metadata': metadata,
                    'column': selected_column,
                    'method_type': method_type,
                    'method_name': method_name,
                    'parameters': parameters
                }
                st.success("✅ Preview generated!")
            else:
                st.error(f"❌ Preview failed: {metadata.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error generating preview: {str(e)}")

with preview_cols[1]:
    if st.button("✅ Apply Changes", type="primary", width='stretch'):
        if 'preview_results' not in st.session_state:
            st.warning("⚠️ Please generate a preview first")
        else:
            # Confirm before applying
            if st.session_state.get('confirm_apply', False):
                # Create backup
                create_backup()
                
                # Apply changes
                preview = st.session_state.preview_results
                st.session_state.dataset[selected_column] = preview['cleaned']
                
                # Save operation to history
                operation = {
                    'column': selected_column,
                    'method_type': method_type,
                    'method_name': method_name,
                    'parameters': parameters,
                    'result': 'Applied successfully',
                    'metadata': preview['metadata']
                }
                save_cleaning_operation(operation)
                
                # Clear preview and confirmation
                del st.session_state.preview_results
                st.session_state.confirm_apply = False
                
                st.success("✅ Changes applied successfully!")
                st.rerun()
            else:
                st.session_state.confirm_apply = True
                st.warning("⚠️ Click again to confirm applying changes")

# Show preview results
if 'preview_results' in st.session_state:
    preview = st.session_state.preview_results
    
    st.subheader("5. Preview Results")
    
    # Impact statistics
    metadata = preview['metadata']
    impact_stats = metadata.get('impact_stats', {})
    
    if impact_stats:
        impact_cols = st.columns(4)
        
        with impact_cols[0]:
            st.metric("Rows Affected", impact_stats.get('rows_affected', 0))
        with impact_cols[1]:
            st.metric("% Changed", f"{impact_stats.get('percentage_changed', 0):.2f}%")
        with impact_cols[2]:
            st.metric("Missing Before", impact_stats.get('missing_before', 0))
        with impact_cols[3]:
            st.metric("Missing After", impact_stats.get('missing_after', 0))
    
    # Before/After comparison
    if pd.api.types.is_numeric_dtype(preview['original']):
        # Statistical comparison for numeric data
        stats_cols = st.columns(2)
        
        with stats_cols[0]:
            st.markdown("**Before Statistics:**")
            orig_stats = preview['original'].describe()
            st.dataframe(orig_stats.to_frame('Original'), width='stretch')
        
        with stats_cols[1]:
            st.markdown("**After Statistics:**")
            clean_stats = preview['cleaned'].describe()
            st.dataframe(clean_stats.to_frame('Cleaned'), width='stretch')
    
    # Visualization comparison
    st.markdown("**Before/After Comparison:**")
    comparison_fig = visualizer.plot_before_after_comparison(
        preview['original'], 
        preview['cleaned'], 
        selected_column, 
        method_name
    )
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Sample data comparison
    st.markdown("**Sample Data Changes:**")
    
    # Show rows that changed
    original_series = preview['original']
    cleaned_series = preview['cleaned']
    
    changed_mask = (original_series != cleaned_series) | (original_series.isnull() != cleaned_series.isnull())
    changed_indices = changed_mask[changed_mask].index[:20]  # Show first 20 changes
    
    if len(changed_indices) > 0:
        comparison_data = pd.DataFrame({
            'Index': changed_indices,
            'Before': [original_series.loc[i] if pd.notna(original_series.loc[i]) else 'NaN' for i in changed_indices],
            'After': [cleaned_series.loc[i] if pd.notna(cleaned_series.loc[i]) else 'NaN' for i in changed_indices]
        })
        
        st.dataframe(comparison_data, width='stretch')
        
        if len(changed_indices) == 20:
            remaining = changed_mask.sum() - 20
            st.caption(f"... and {remaining} more changes")
    else:
        st.info("No changes detected in the data")

# AI Guidance Section
st.subheader("6. 🤖 AI Guidance")

ai_guidance_cols = st.columns([3, 1])

with ai_guidance_cols[0]:
    # Check if we have a selected quick question
    default_question = st.session_state.get('selected_quick_question', '')
    ai_question = st.text_area(
        "Ask AI about this cleaning method:",
        value=default_question,
        placeholder="e.g., 'Will this method preserve the statistical properties of my data?' or 'What are the risks of applying this method?'",
        key="cleaning_ai_question"
    )
    
    # Clear the quick question after it's been loaded
    if 'selected_quick_question' in st.session_state:
        del st.session_state.selected_quick_question

with ai_guidance_cols[1]:
    st.markdown("**Quick Questions:**")
    
    quick_questions = [
        "Is this the best method for this column?",
        "What are the risks?",
        "How will this affect my analysis?",
        "Are there better alternatives?"
    ]
    
    for i, question in enumerate(quick_questions):
        if st.button(f"❓ {question}", key=f"quick_q_{i}", width='stretch'):
            # Store in a different session state key to avoid widget conflicts
            st.session_state.selected_quick_question = question
            st.rerun()

if st.button("🤖 Get AI Guidance") and ai_question.strip():
    assistant = AIAssistant()
    
    # Set context
    dataset_info = {
        'shape': df.shape,
        'columns': len(df.columns),
        'missing_summary': df.isnull().sum().to_dict(),
        'column_types': st.session_state.column_types
    }
    
    assistant.set_context(dataset_info, analysis)
    
    with st.spinner("🤖 AI is analyzing..."):
        if 'preview_results' in st.session_state:
            # If we have preview results, ask about the specific method and impact
            detailed_question = f"""I'm planning to apply {method_name} to column '{selected_column}'. 
            The preview shows {st.session_state.preview_results['metadata']['impact_stats'].get('rows_affected', 0)} rows will be affected.
            
            User question: {ai_question}"""
        else:
            # General question about the method
            detailed_question = f"""I'm considering applying {method_name} to column '{selected_column}'.
            
            User question: {ai_question}"""
        
        response = assistant.ask_question(detailed_question, selected_column)
        
        st.markdown("**🤖 AI Response:**")
        st.markdown(response)

# Cleaning History
st.subheader("7. 📝 Cleaning History")

if st.session_state.cleaning_history:
    # Show history for current column
    column_history = st.session_state.cleaning_history.get(selected_column, [])
    
    if column_history:
        st.markdown(f"**Operations applied to '{selected_column}':**")
        
        for i, operation in enumerate(reversed(column_history)):  # Show most recent first
            with st.expander(f"Operation {len(column_history)-i}: {operation.get('method_name', 'Unknown')} - {operation.get('timestamp', '')[:19]}"):
                st.write(f"**Method:** {operation.get('method_name', 'Unknown')}")
                st.write(f"**Type:** {operation.get('method_type', 'Unknown')}")
                st.write(f"**Result:** {operation.get('result', 'Unknown')}")
                
                if operation.get('parameters'):
                    st.write("**Parameters:**")
                    for key, value in operation['parameters'].items():
                        st.write(f"  - {key}: {value}")
                
                if operation.get('impact_stats'):
                    impact = operation['impact_stats']
                    st.write(f"**Impact:** {impact.get('rows_affected', 0)} rows affected ({impact.get('percentage_changed', 0):.2f}%)")
    else:
        st.info(f"No cleaning operations applied to '{selected_column}' yet.")
    
    # Overall history summary
    total_operations = sum(len(ops) for ops in st.session_state.cleaning_history.values())
    columns_cleaned = len(st.session_state.cleaning_history)
    
    st.markdown(f"**Overall Progress:** {total_operations} operations across {columns_cleaned} columns")
    
else:
    st.info("No cleaning operations performed yet.")

# Sidebar with column status
with st.sidebar:
    st.markdown("### 🧹 Cleaning Status")
    
    # Show cleaning progress
    cleaned_columns = len(st.session_state.cleaning_history)
    total_columns = len(df.columns)
    progress = cleaned_columns / total_columns if total_columns > 0 else 0
    
    st.progress(progress)
    st.write(f"{cleaned_columns}/{total_columns} columns cleaned")
    
    # Column quality overview
    st.markdown("### 📊 Column Quality")
    
    # Collect columns with quality scores
    columns_with_quality = []
    columns_without_quality = []
    
    for col in df.columns:
        col_analysis = st.session_state.column_analysis.get(col, {})
        if col_analysis:
            quality_score = col_analysis.get('data_quality', {}).get('score', 0)
            columns_with_quality.append((col, quality_score))
        else:
            columns_without_quality.append(col)
    
    # Sort columns by quality score (ascending order - worst first)
    columns_with_quality.sort(key=lambda x: x[1])
    
    # Display sorted columns (first 10)
    display_count = 0
    max_display = 10
    
    for col, quality_score in columns_with_quality[:max_display]:
        quality_color = "green" if quality_score >= 80 else "orange" if quality_score >= 60 else "red"
        cleaned_indicator = "🧹" if col in st.session_state.cleaning_history else "⏳"
        
        st.markdown(f"""
        **{col[:15]}{'...' if len(col) > 15 else ''}** {cleaned_indicator}  
        <span style="color: {quality_color}">{quality_score}% Quality</span>
        """, unsafe_allow_html=True)
        display_count += 1
    
    # Show unanalyzed columns if space remains
    remaining_slots = max_display - display_count
    for col in columns_without_quality[:remaining_slots]:
        st.write(f"**{col[:15]}{'...' if len(col) > 15 else ''}** ❓ Not analyzed")
        display_count += 1
    
    total_remaining = len(columns_with_quality) + len(columns_without_quality) - display_count
    if total_remaining > 0:
        st.caption(f"... and {total_remaining} more columns")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔍 Analyze All Columns", width='stretch'):
        st.switch_page("pages/2_Column_Analysis.py")
    
    if st.button("📊 Generate Report", width='stretch'):
        st.switch_page("pages/7_Reports.py")
    
    if st.button("🤖 AI Assistant", width='stretch'):
        st.switch_page("pages/6_AI_Assistant.py")

st.divider()

st.subheader("✅ Data Validation & Quality Check")

if st.button("🔍 Run Pre-Export Validation", type="primary"):
    with st.spinner("Validating dataset quality..."):
        validation_results = cleaning_engine.validate_dataset_for_export(df)
        st.session_state.validation_results = validation_results
        st.rerun()

if 'validation_results' in st.session_state and st.session_state.validation_results:
    val_results = st.session_state.validation_results
    
    if val_results['overall_passed']:
        st.success("✅ Dataset passed validation! Ready for export.")
    else:
        st.error("❌ Dataset has critical issues that should be addressed before export.")
    
    val_cols = st.columns(3)
    with val_cols[0]:
        st.metric("Critical Issues", val_results['total_issues'])
    with val_cols[1]:
        st.metric("Warnings", val_results['total_warnings'])
    with val_cols[2]:
        columns_with_issues = sum(1 for v in val_results['column_validations'].values() if not v['passed'])
        st.metric("Columns with Issues", columns_with_issues)
    
    if val_results['critical_issues']:
        st.markdown("#### 🔴 Critical Issues:")
        for issue in val_results['critical_issues']:
            st.error(issue)
    
    if val_results['recommendations']:
        st.markdown("#### 💡 Recommendations:")
        for rec in val_results['recommendations']:
            st.info(rec)
    
    with st.expander("📋 Detailed Column Validation"):
        for col, col_val in val_results['column_validations'].items():
            status_icon = "✅" if col_val['passed'] else "❌"
            st.markdown(f"**{status_icon} {col}**")
            
            if col_val['issues']:
                for issue in col_val['issues']:
                    st.error(f"  • {issue}")
            
            if col_val['warnings']:
                for warning in col_val['warnings']:
                    st.warning(f"  • {warning}")
            
            if not col_val['issues'] and not col_val['warnings']:
                st.success("  • No issues detected")
            
            st.divider()

# Navigation hints
st.markdown("---")
st.markdown("**Tips:**")
st.markdown("- 👁️ Always preview changes before applying them")
st.markdown("- ✅ Run validation checks before exporting data")
st.markdown("- 🤖 Use AI guidance for method selection and validation")
st.markdown("- ↶↷ Use undo/redo to experiment safely")
st.markdown("- 📊 Monitor the impact statistics to understand changes")
