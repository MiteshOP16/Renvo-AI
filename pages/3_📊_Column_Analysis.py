import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.utils import initialize_session_state, get_column_summary, format_number
from modules.data_analyzer import ColumnAnalyzer
from modules.visualization import DataVisualizer
from modules.ai_assistant import AIAssistant



st.title("🔍 Individual Column Analysis")

# Check if dataset is loaded
if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset

# Check for inter-column violations when page loads
if 'inter_column_violations' not in st.session_state:
    with st.spinner("🔄 Running inter-column validation checks..."):
        analyzer = st.session_state.data_analyzer
        violations = analyzer.detect_inter_column_violations(df)
        st.session_state.inter_column_violations = violations
        
        if violations['total_violations'] > 0:
            severity_color = {'low': '🟢', 'moderate': '🟡', 'high': '🔴'}[violations['severity']]
            st.warning(f"{severity_color} **Inter-column violations detected:** {violations['total_violations']} violations found ({violations['severity']} severity)")

# Display inter-column violations summary if any
if st.session_state.get('inter_column_violations', {}).get('total_violations', 0) > 0:
    violations = st.session_state.inter_column_violations
    with st.expander(f"⚠️ Inter-Column Violations Summary ({violations['total_violations']} violations)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Violations", violations['total_violations'])
        with col2:
            severity_emoji = {'low': '🟢', 'moderate': '🟡', 'high': '🔴'}[violations['severity']]
            st.metric("Severity", f"{severity_emoji} {violations['severity'].title()}")
        
        if violations['violation_types']:
            st.subheader("Violation Types Found:")
            for vtype in violations['violation_types']:
                st.write(f"• {vtype}")
        
        if violations.get('rule_checks'):
            st.subheader("Detailed Rule Checks:")
            for rule in violations['rule_checks']:
                st.write(f"**{rule['rule_type'].replace('_', ' ').title()}:** {rule['description']} ({rule['violations']} violations)")
                st.caption(f"Columns involved: {', '.join(rule['columns'])}")
analyzer = ColumnAnalyzer()
visualizer = DataVisualizer()

st.markdown("""
Analyze each column individually to understand data quality, patterns, and cleaning requirements. 
Each column receives tailored analysis based on its characteristics and data type.
""")

# Column selection
st.subheader("1. Select Column for Analysis")

col_select, col_info = st.columns([2, 2])

with col_select:
    selected_column = st.selectbox(
        "Choose a column to analyze:",
        options=df.columns.tolist(),
        help="Select a column to view detailed analysis and cleaning recommendations"
    )

with col_info:
    if selected_column:
        column_type = st.session_state.column_types.get(selected_column, 'unknown')
        st.info(f"**Column Type:** {column_type.title()}")
        st.write(get_column_summary(df, selected_column))

if not selected_column:
    st.stop()

# Analysis controls
analysis_controls = st.columns([2, 1, 1])

with analysis_controls[0]:
    force_refresh = st.checkbox("Force refresh analysis", help="Recalculate analysis even if cached")

with analysis_controls[1]:
    if st.button("🔍 Analyze Column", type="primary"):
        with st.spinner(f"Analyzing column '{selected_column}'..."):
            try:
                analysis_result = analyzer.analyze_column(df, selected_column, force_refresh=force_refresh)
                st.session_state.column_analysis[selected_column] = analysis_result
                st.success("✅ Analysis completed!")
            except Exception as e:
                st.error(f"❌ Error analyzing column: {str(e)}")

with analysis_controls[2]:
    if st.button("📊 Quick Overview", help="Show summary of all columns"):
        st.session_state.show_overview = True

# Show overview if requested
if st.session_state.get('show_overview', False):
    st.subheader("📊 All Columns Overview")
    
    # Create overview visualization
    overview_fig = visualizer.plot_column_overview(df)
    st.plotly_chart(overview_fig, use_container_width=True)
    
    # Missing patterns heatmap
    missing_fig = visualizer.plot_missing_patterns(df)
    st.plotly_chart(missing_fig, use_container_width=True)
    
    if st.button("❌ Close Overview"):
        st.session_state.show_overview = False
        st.rerun()

# Display analysis results
if selected_column in st.session_state.column_analysis:
    analysis = st.session_state.column_analysis[selected_column]
    
    st.subheader(f"2. Detailed Analysis: {selected_column}")
    
    # Basic Information Tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Basic Info", 
        "❌ Missing Data", 
        "⚡ Outliers", 
        "📈 Distribution", 
        "🚨 Rule Violations",
        "🎯 Recommendations"
    ])
    
    with tab1:
        st.subheader("Basic Information")
        
        basic_info = analysis['basic_info']
        
        # Metrics in columns
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Total Values", f"{basic_info['count']:,}")
        with metric_cols[1]:
            st.metric("Missing Values", f"{basic_info['missing_count']:,}")
        with metric_cols[2]:
            st.metric("Missing %", f"{basic_info['missing_percentage']:.2f}%")
        with metric_cols[3]:
            st.metric("Unique Values", f"{basic_info['unique_count']:,}")
        
        # Additional metrics for numeric columns
        if 'mean' in basic_info:
            numeric_cols = st.columns(4)
            with numeric_cols[0]:
                st.metric("Mean", format_number(basic_info['mean']))
            with numeric_cols[1]:
                st.metric("Median", format_number(basic_info['median']))
            with numeric_cols[2]:
                st.metric("Std Dev", format_number(basic_info['std']))
            with numeric_cols[3]:
                st.metric("Range", f"{format_number(basic_info['min'])} - {format_number(basic_info['max'])}")
        
        # Data Quality Score
        quality = analysis['data_quality']
        quality_color = "green" if quality['score'] >= 80 else "orange" if quality['score'] >= 60 else "red"
        
        st.markdown("### Data Quality Assessment")
        st.markdown(f"""
        **Quality Score:** <span style="color: {quality_color}; font-size: 1.2em; font-weight: bold;">
        {quality['score']}/100 (Grade: {quality['grade']})</span>
        """, unsafe_allow_html=True)
        
        if quality['issues']:
            st.markdown("**Issues Identified:**")
            for issue in quality['issues']:
                st.markdown(f"- ⚠️ {issue}")
        else:
            st.success("✅ No significant quality issues detected")
    
    with tab2:
        st.subheader("Missing Data Analysis")
        
        missing_analysis = analysis['missing_analysis']
        
        if missing_analysis['total_missing'] == 0:
            st.success("✅ No missing values in this column!")
        else:
            # Missing data metrics
            missing_cols = st.columns(3)
            
            with missing_cols[0]:
                st.metric("Total Missing", missing_analysis['total_missing'])
            with missing_cols[1]:
                st.metric("Missing %", f"{missing_analysis['percentage']:.2f}%")
            with missing_cols[2]:
                st.metric("Pattern Type", missing_analysis['pattern_type'].replace('_', ' ').title())
            
            # Pattern analysis
            st.markdown("### Pattern Analysis")
            
            pattern_descriptions = {
                'sporadic': '🟢 **Sporadic**: Missing values are randomly scattered (<5% missing)',
                'random': '🟡 **Random**: Missing values appear randomly distributed',
                'systematic_blocks': '🟠 **Systematic Blocks**: Large consecutive blocks of missing data',
                'front_loaded': '🔵 **Front Loaded**: Most missing values at the beginning',
                'tail_loaded': '🟣 **Tail Loaded**: Most missing values at the end'
            }
            
            pattern = missing_analysis['pattern_type']
            if pattern in pattern_descriptions:
                st.markdown(pattern_descriptions[pattern])
            
            # Consecutive missing information
            if missing_analysis.get('max_consecutive', 0) > 0:
                st.write(f"**Maximum consecutive missing values:** {missing_analysis['max_consecutive']}")
            
            # Missing data visualization
            if missing_analysis['total_missing'] > 0:
                st.markdown("### Missing Data Pattern Visualization")
                
                # Create missing data plot
                series = df[selected_column]
                missing_mask = series.isnull()
                
                fig = go.Figure()
                
                # Plot missing vs present
                fig.add_trace(go.Scatter(
                    x=list(range(len(series))),
                    y=missing_mask.astype(int),
                    mode='markers',
                    name='Missing (1) vs Present (0)',
                    marker=dict(
                        color=missing_mask.astype(int),
                        colorscale=[[0, 'blue'], [1, 'red']],
                        size=4
                    )
                ))
                
                fig.update_layout(
                    title=f"Missing Data Pattern - {selected_column}",
                    xaxis_title="Row Index",
                    yaxis_title="Missing (1) / Present (0)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Outlier Detection")
        
        outlier_analysis = analysis['outlier_analysis']
        
        # Check if outlier detection was attempted and has valid results
        method_results = outlier_analysis.get('method_results', {})
        has_valid_results = method_results and any(
            result.get('outlier_count', 0) > 0 or 'outlier_values' in result 
            for result in method_results.values()
        )
        
        if not method_results:
            # No method results at all - not applicable for this column type
            summary = outlier_analysis.get('summary', {})
            reason = summary.get('analysis', 'Outlier detection not applicable for this column type')
            st.info(f"ℹ️ {reason}")
        else:
            # Outlier detection was performed - show results even if no outliers found
            summary = outlier_analysis['summary']
            
            # Check if any outliers were actually found (excluding note entries)
            total_outliers_found = sum(
                result.get('outlier_count', 0) for result in method_results.values() 
                if 'note' not in result
            )
            
            if total_outliers_found == 0:
                st.success("✅ No outliers detected using any method")
                st.write("Outlier detection was performed but no significant outliers were found.")
            else:
                # Summary metrics
                severity_colors = {'high': 'red', 'moderate': 'orange', 'low': 'green'}
                severity = summary.get('severity', 'low')
                
                st.markdown(f"""
                **Outlier Severity:** <span style="color: {severity_colors[severity]}; font-weight: bold;">
                {severity.upper()}</span>
                """, unsafe_allow_html=True)
            
            # Always show method comparison table
            st.markdown("### Detection Methods Comparison")
            
            method_data = []
            for method_key, results in method_results.items():
                if 'note' in results:
                    # Handle note entries differently
                    method_data.append({
                        'Method': results['method'],
                        'Outliers Found': 'N/A (small sample)',
                        'Percentage': 'N/A'
                    })
                else:
                    method_data.append({
                        'Method': results['method'],
                        'Outliers Found': results['outlier_count'],
                        'Percentage': f"{results['outlier_percentage']:.2f}%"
                    })
            
            method_df = pd.DataFrame(method_data)
            st.dataframe(method_df, use_container_width=True)
            
            # Detailed results for each method
            for method_key, results in method_results.items():
                with st.expander(f"📊 {results['method']} Details"):
                    # Handle special note entries for small samples
                    if 'note' in results:
                        st.info(results['note'])
                    else:
                        st.write(f"**Outliers found:** {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
                        
                        if 'lower_bound' in results and 'upper_bound' in results:
                            st.write(f"**Bounds:** {format_number(results['lower_bound'])} to {format_number(results['upper_bound'])}")
                        
                        if 'threshold' in results:
                            st.write(f"**Threshold:** {results['threshold']}")
                        
                        if results.get('outlier_values'):
                            st.write("**Sample outlier values:**")
                            outlier_sample = results['outlier_values'][:10]  # Show first 10
                            st.write(", ".join([format_number(v) for v in outlier_sample]))
                        elif results['outlier_count'] == 0:
                            st.write("**No outlier values found by this method**")
            
            # Outlier visualization (only if outliers were found)
            if total_outliers_found > 0:
                st.markdown("### Outlier Visualization")
                try:
                    outlier_fig = visualizer.plot_outliers(df[selected_column], selected_column, outlier_analysis)
                    st.plotly_chart(outlier_fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not generate outlier visualization")
    
    with tab4:
        st.subheader("Distribution Analysis")
        
        distribution_analysis = analysis['distribution_analysis']
        
        # Distribution plot with enhanced visualization
        dist_fig = visualizer.plot_column_distribution(df[selected_column], selected_column)
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Add explanation banner
        st.info("📊 **Understanding Distribution:** The distribution shows how values are spread across the data range. This helps identify patterns, clusters, and potential data quality issues.")
        
        # Distribution characteristics
        if distribution_analysis['type'] == 'numeric':
            st.markdown("### 📈 Distribution Characteristics Explained")
            
            char_cols = st.columns(3)
            
            with char_cols[0]:
                skew_val = distribution_analysis['skewness']
                if abs(skew_val) < 0.5:
                    skew_desc = "Approximately Normal"
                    skew_explain = "Data is balanced around the mean"
                    skew_icon = "✅"
                elif abs(skew_val) < 1:
                    skew_desc = "Moderately Skewed"
                    skew_explain = "Some asymmetry in the data"
                    skew_icon = "⚠️"
                else:
                    skew_desc = "Highly Skewed"
                    skew_explain = "Data heavily concentrated on one side"
                    skew_icon = "🔴"
                
                st.metric("Skewness", f"{skew_val:.3f}")
                st.markdown(f"**{skew_icon} {skew_desc}**")
                st.caption(skew_explain)
                
                # Direction explanation
                if skew_val > 0.1:
                    st.caption("➡️ Right-skewed: tail extends to the right")
                elif skew_val < -0.1:
                    st.caption("⬅️ Left-skewed: tail extends to the left")
            
            with char_cols[1]:
                kurt_val = distribution_analysis['kurtosis']
                if kurt_val < -0.5:
                    kurt_desc = "Platykurtic"
                    kurt_explain = "Flatter distribution with lighter tails"
                    kurt_icon = "📊"
                elif kurt_val > 0.5:
                    kurt_desc = "Leptokurtic"
                    kurt_explain = "Peaked distribution with heavier tails"
                    kurt_icon = "📈"
                else:
                    kurt_desc = "Mesokurtic"
                    kurt_explain = "Normal-like peakedness"
                    kurt_icon = "✅"
                
                st.metric("Kurtosis", f"{kurt_val:.3f}")
                st.markdown(f"**{kurt_icon} {kurt_desc}**")
                st.caption(kurt_explain)
            
            with char_cols[2]:
                normality = distribution_analysis['normality_test']
                is_normal = normality['is_normal']
                
                st.metric("Normality Test", "✅ Normal" if is_normal else "❌ Not Normal")
                st.caption(f"p-value: {normality['shapiro_p']:.4f}")
                
                if is_normal:
                    st.caption("Data follows a normal distribution")
                else:
                    st.caption("Data deviates from normal distribution")
            
            # Quartile information with explanation
            st.markdown("### 📊 Quartile Analysis")
            
            quartile_cols = st.columns(5)
            basic_info = analysis['basic_info']
            
            with quartile_cols[0]:
                st.metric("Minimum", format_number(basic_info['min']))
                st.caption("Lowest value")
            
            with quartile_cols[1]:
                st.metric("Q1 (25%)", format_number(basic_info.get('q1', 0)))
                st.caption("25% of data below")
            
            with quartile_cols[2]:
                st.metric("Median (50%)", format_number(basic_info['median']))
                st.caption("Middle value")
            
            with quartile_cols[3]:
                st.metric("Q3 (75%)", format_number(basic_info.get('q3', 0)))
                st.caption("75% of data below")
            
            with quartile_cols[4]:
                st.metric("Maximum", format_number(basic_info['max']))
                st.caption("Highest value")
            
            # Interpretation guidance
            with st.expander("💡 How to Interpret These Statistics"):
                st.markdown("""
                **Skewness:**
                - **Near 0 (-0.5 to 0.5)**: Symmetric distribution, data is balanced
                - **Positive (> 0.5)**: Right-skewed, most values on the left, with a long tail to the right
                - **Negative (< -0.5)**: Left-skewed, most values on the right, with a long tail to the left
                
                **Kurtosis:**
                - **Near 0 (-0.5 to 0.5)**: Normal bell-curve shape
                - **Positive (> 0.5)**: More peaked with heavier tails (more outliers likely)
                - **Negative (< -0.5)**: Flatter with lighter tails (fewer outliers)
                
                **Normality Test:**
                - If p-value > 0.05: Data likely follows a normal distribution
                - If p-value ≤ 0.05: Data significantly deviates from normal distribution
                - Many statistical methods assume normality, so this is important to check
                
                **Quartiles (Q1, Median, Q3):**
                - **Q1**: 25% of values are below this point
                - **Median**: The middle value - 50% above, 50% below
                - **Q3**: 75% of values are below this point
                - **IQR (Q3-Q1)**: Measures spread of the middle 50% of data
                """)
        
        elif distribution_analysis['type'] == 'categorical':
            st.markdown("### 📊 Category Distribution Explained")
            
            freq_dist = distribution_analysis['frequency_distribution']
            if freq_dist:
                freq_data = list(freq_dist.items())
                freq_df = pd.DataFrame(freq_data, columns=['Category', 'Count'])
                freq_df['Percentage'] = (freq_df['Count'] / freq_df['Count'].sum() * 100).round(2)
                freq_df = freq_df.sort_values('Count', ascending=False)
                
                st.dataframe(freq_df, use_container_width=True, hide_index=True)
                
                # Category insights
                insights_cols = st.columns(3)
                
                with insights_cols[0]:
                    st.metric("Unique Categories", len(freq_df))
                    st.caption("Number of distinct values")
                
                with insights_cols[1]:
                    most_common = freq_df.iloc[0]
                    st.metric("Most Common", f"{most_common['Count']:,}")
                    st.caption(f"Category: {most_common['Category']}")
                
                with insights_cols[2]:
                    least_common = freq_df.iloc[-1]
                    st.metric("Least Common", f"{least_common['Count']:,}")
                    st.caption(f"Category: {least_common['Category']}")
            
            if 'entropy' in distribution_analysis:
                entropy_val = distribution_analysis['entropy']
                max_entropy = np.log(len(freq_dist)) if len(freq_dist) > 0 else 1
                entropy_ratio = entropy_val / max_entropy if max_entropy > 0 else 0
                
                st.metric("Entropy Score", f"{entropy_val:.3f}")
                st.caption(f"Uniformity: {entropy_ratio*100:.1f}%")
                
                if entropy_ratio > 0.9:
                    st.info("✅ Categories are evenly distributed")
                elif entropy_ratio > 0.7:
                    st.info("⚠️ Moderately balanced distribution")
                else:
                    st.warning("⚠️ Imbalanced distribution - some categories dominate")
                
                with st.expander("💡 Understanding Entropy"):
                    st.markdown("""
                    **Entropy** measures how evenly distributed categories are:
                    - **High entropy (>90%)**: All categories appear with similar frequencies
                    - **Medium entropy (70-90%)**: Some imbalance between categories
                    - **Low entropy (<70%)**: Strong concentration in few categories
                    
                    **Why it matters:**
                    - Imbalanced categories can bias analysis
                    - Machine learning models may struggle with rare categories
                    - May indicate data collection issues
                    """)
    
    with tab5:
        st.subheader("Rule-Based Violations")
        
        rule_violations = analysis.get('rule_violations', {})
        
        if rule_violations.get('total_violations', 0) == 0:
            st.success("✅ No rule violations detected in this column!")
        else:
            # Violation summary
            severity = rule_violations.get('severity', 'low')
            severity_colors = {'high': 'red', 'moderate': 'orange', 'low': 'yellow'}
            severity_icons = {'high': '🔴', 'moderate': '🟡', 'low': '🟡'}
            
            violation_cols = st.columns(3)
            
            with violation_cols[0]:
                st.metric("Total Violations", rule_violations['total_violations'])
            with violation_cols[1]:
                st.metric("Violation Types", len(rule_violations.get('violation_types', [])))
            with violation_cols[2]:
                st.markdown(f"""
                **Severity:** {severity_icons[severity]} <span style="color: {severity_colors[severity]}; font-weight: bold;">
                {severity.upper()}</span>
                """, unsafe_allow_html=True)
            
            # Display violation types
            if rule_violations.get('violation_types'):
                st.markdown("### Detected Violations")
                for violation_type in rule_violations['violation_types']:
                    st.markdown(f"- ⚠️ {violation_type}")
            
            # Detailed violation information
            details = rule_violations.get('details', {})
            if details:
                st.markdown("### Violation Details")
                
                for violation_key, violation_info in details.items():
                    with st.expander(f"📊 {violation_info.get('rule', violation_key)}"):
                        st.write(f"**Count:** {violation_info.get('count', 0)} violations")
                        st.write(f"**Rule:** {violation_info.get('rule', 'No rule specified')}")
                        
                        if violation_info.get('invalid_values'):
                            st.write("**Sample invalid values:**")
                            invalid_values = violation_info['invalid_values']
                            for i, value in enumerate(invalid_values[:5], 1):
                                st.write(f"{i}. `{value}`")
                            
                            if len(invalid_values) > 5:
                                st.write(f"... and {len(invalid_values) - 5} more")
            
            # Recommendations for fixing violations
            st.markdown("### 🛠️ Fixing Rule Violations")
            st.info("""
            **Recommended Actions:**
            - Review the invalid values to understand the pattern
            - Consider if these are data entry errors or legitimate edge cases
            - Use the Cleaning Wizard to apply appropriate transformations
            - Consult with domain experts for unusual cases
            """)
    
    with tab6:
        st.subheader("Cleaning Recommendations")
        
        recommendations = analysis['cleaning_recommendations']
        
        if not recommendations:
            st.info("ℹ️ No specific cleaning recommendations for this column")
        else:
            st.markdown(f"**{len(recommendations)} recommendations generated** based on column-specific analysis:")
            
            for i, rec in enumerate(recommendations, 1):
                priority_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                priority_icon = priority_colors.get(rec['priority'], '⚪')
                
                with st.expander(f"{priority_icon} {i}. {rec['description']} (Score: {rec['applicability_score']}/100)"):
                    st.markdown(f"**Method:** `{rec['method']}`")
                    st.markdown(f"**Type:** {rec['type'].replace('_', ' ').title()}")
                    st.markdown(f"**Priority:** {rec['priority'].upper()}")
                    
                    if rec['pros']:
                        st.markdown("**Pros:**")
                        for pro in rec['pros']:
                            st.markdown(f"- ✅ {pro}")
                    
                    if rec['cons']:
                        st.markdown("**Cons:**")
                        for con in rec['cons']:
                            st.markdown(f"- ⚠️ {con}")
                    
                    # Quick apply button
                    if st.button(f"Apply {rec['method']}", key=f"apply_{selected_column}_{i}"):
                        st.info("💡 Use the Cleaning Wizard page to apply cleaning methods with full control and preview.")
        
        # AI Assistant integration
        st.markdown("### 🤖 Get AI Guidance")
        
        ai_cols = st.columns([2, 1])
        
        with ai_cols[0]:
            ai_question = st.text_input(
                "Ask the AI about this column:",
                placeholder="e.g., 'Why do you recommend KNN over mean imputation for this column?'",
                key=f"ai_question_{selected_column}"
            )
        
        with ai_cols[1]:
            if st.button("Ask AI", key=f"ask_ai_{selected_column}"):
                if ai_question:
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
                        response = assistant.ask_question(ai_question, selected_column)
                        st.markdown("**AI Response:**")
                        st.markdown(response)
                else:
                    st.warning("Please enter a question first.")

else:
    st.info("👆 Select a column and click 'Analyze Column' to see detailed analysis.")

# Navigation hints
st.markdown("---")
st.markdown("**Next Steps:**")
st.markdown("- 🧹 Use **Cleaning Wizard** to apply cleaning methods")
st.markdown("- 🤖 Visit **AI Assistant** for detailed guidance and explanations")
st.markdown("- 📊 Generate **Reports** to document your analysis")

# Quick actions sidebar
with st.sidebar:
    st.markdown("### 🔍 Quick Actions")
    
    if selected_column and selected_column in st.session_state.column_analysis:
        analysis = st.session_state.column_analysis[selected_column]
        
        # Quality score
        quality_score = analysis['data_quality']['score']
        quality_color = "green" if quality_score >= 80 else "orange" if quality_score >= 60 else "red"
        
        st.markdown(f"""
        **{selected_column}**  
        Quality: <span style="color: {quality_color}">**{quality_score}/100**</span>
        """, unsafe_allow_html=True)
        
        # Quick stats
        missing_pct = analysis['basic_info']['missing_percentage']
        if missing_pct > 0:
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        if 'method_results' in analysis['outlier_analysis']:
            outlier_count = sum([r['outlier_count'] for r in analysis['outlier_analysis']['method_results'].values()])
            if outlier_count > 0:
                st.metric("Outliers Detected", outlier_count)
        
        # Quick navigation
        if st.button("🧹 Clean This Column", width='stretch'):
            st.switch_page("pages/3_Cleaning_Wizard.py")
        
        if st.button("🤖 Ask AI About This Column", width='stretch'):
            st.switch_page("pages/6_AI_Assistant.py")
    
    # Analysis progress
    st.markdown("### 📊 Analysis Progress")
    analyzed_count = len(st.session_state.column_analysis)
    total_count = len(df.columns)
    progress = analyzed_count / total_count if total_count > 0 else 0
    
    st.progress(progress)
    st.write(f"{analyzed_count}/{total_count} columns analyzed")
    
    if st.button("🔍 Analyze All Columns", width='stretch'):
        analyzer = ColumnAnalyzer()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, col in enumerate(df.columns):
            if col not in st.session_state.column_analysis:
                status_text.text(f"Analyzing: {col}")
                try:
                    analysis = analyzer.analyze_column(df, col)
                    st.session_state.column_analysis[col] = analysis
                except Exception as e:
                    st.error(f"Error analyzing {col}: {str(e)}")
                
            progress_bar.progress((i + 1) / len(df.columns))
        
        status_text.text("✅ All columns analyzed!")
        st.success("Analysis complete!")
        st.rerun()
