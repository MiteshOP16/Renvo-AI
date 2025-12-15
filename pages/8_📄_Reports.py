import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state
from modules.report_generator import ReportGenerator
from modules.visualization import DataVisualizer
import io
import base64
from datetime import datetime

# Initialize session state


st.title("📊 Comprehensive Report Generation")

st.markdown("""
Generate standardized reports in PDF or HTML format that summarize the data cleaning process. 
Reports include weighted/unweighted summaries, workflow logs, violation analysis, and visualizations 
formatted according to official statistical release requirements.
""")

# Check if dataset is loaded
if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset
original_df = st.session_state.original_dataset

# Initialize report generator with weights manager
weights_manager = st.session_state.get('weights_manager')
if weights_manager:
    st.session_state.report_generator.weights_manager = weights_manager

report_generator = st.session_state.report_generator
visualizer = DataVisualizer()

st.subheader("📥 Data Export Options")

export_cols = st.columns([2, 1, 1])

with export_cols[0]:
    st.markdown("Export your cleaned dataset in various formats")

with export_cols[1]:
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📊 Download Cleaned CSV",
        data=csv_data,
        file_name=f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with export_cols[2]:
    if original_df is not None:
        csv_original = original_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Original CSV",
            data=csv_original,
            file_name=f"original_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()

st.subheader("📋 Cleaned Dataset Preview")

preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Current Data", "Summary Statistics", "Data Comparison"])

with preview_tab1:
    st.markdown("**Current Cleaned Dataset:**")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing first 100 rows of {len(df)} total rows")

with preview_tab2:
    st.markdown("**Dataset Statistics:**")
    
    stats_cols = st.columns(4)
    with stats_cols[0]:
        st.metric("Total Rows", f"{len(df):,}")
    with stats_cols[1]:
        st.metric("Total Columns", len(df.columns))
    with stats_cols[2]:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with stats_cols[3]:
        cleaned_count = len(st.session_state.get('cleaning_history', {}))
        st.metric("Columns Cleaned", cleaned_count)
    
    st.markdown("**Numeric Columns Summary:**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("No numeric columns in dataset")

with preview_tab3:
    if original_df is not None:
        st.markdown("**Changes from Original Dataset:**")
        
        comparison_data = []
        for col in df.columns:
            if col in original_df.columns:
                orig_missing = original_df[col].isnull().sum()
                curr_missing = df[col].isnull().sum()
                
                comparison_data.append({
                    'Column': col,
                    'Original Missing': orig_missing,
                    'Current Missing': curr_missing,
                    'Change': curr_missing - orig_missing,
                    'Cleaned': '✅' if col in st.session_state.get('cleaning_history', {}) else ''
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.info("Original dataset not available for comparison")

st.divider()

# Report generation controls
st.subheader("📄 Report Configuration")

config_cols = st.columns([2, 2])

with config_cols[0]:
    report_types = st.multiselect(
        "Select report types to generate:",
        options=['executive_summary', 'detailed_analysis', 'methodology', 'audit_trail'],
        default=['executive_summary', 'detailed_analysis'],
        format_func=lambda x: {
            'executive_summary': '📋 Executive Summary',
            'detailed_analysis': '🔍 Detailed Analysis',
            'methodology': '📚 Methodology Documentation',
            'audit_trail': '📝 Complete Audit Trail'
        }[x]
    )

with config_cols[1]:
    export_format = st.selectbox(
        "Export format:",
        options=['pdf', 'markdown', 'html', 'json'],
        format_func=lambda x: {
            'pdf': '📄 PDF Document',
            'markdown': '📝 Markdown',
            'html': '🌐 HTML',
            'json': '📊 JSON'
        }[x]
    )

# Report generation
st.subheader("2. Generate Reports")

generate_cols = st.columns([2, 1, 1])

with generate_cols[0]:
    if st.button("📊 Generate Reports", type="primary", width='stretch'):
        if not report_types:
            st.error("Please select at least one report type.")
        else:
            with st.spinner("Generating reports..."):
                try:
                    # Generate comprehensive reports with weights integration
                    weights_manager = st.session_state.get('weights_manager')
                    violations = st.session_state.get('inter_column_violations', {})
                    
                    reports = report_generator.generate_complete_report(
                        df, 
                        original_df,
                        st.session_state.column_analysis, 
                        st.session_state.cleaning_history,
                        weights_manager=weights_manager,
                        violations=violations
                    )
                    
                    # Filter to requested types
                    filtered_reports = {k: v for k, v in reports.items() if k in report_types}
                    
                    st.session_state.generated_reports = filtered_reports
                    st.session_state.report_timestamp = datetime.now()
                    
                    st.success(f"✅ Successfully generated {len(filtered_reports)} report(s)!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating reports: {str(e)}")

with generate_cols[1]:
    include_visualizations = st.checkbox("Include visualizations", value=True)

with generate_cols[2]:
    if st.session_state.get('generated_reports'):
        timestamp = st.session_state.get('report_timestamp', datetime.now())
        filename = f"cleaning_report_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        if export_format == 'pdf':
            try:
                # Generate PDF with all data
                anomaly_results = st.session_state.get('anomaly_results', {})
                saved_visualizations = st.session_state.get('saved_visualizations', [])
                
                pdf_bytes = report_generator.export_to_pdf(
                    reports=st.session_state.generated_reports,
                    df=df,
                    analysis_results=st.session_state.column_analysis,
                    cleaning_history=st.session_state.cleaning_history,
                    anomaly_results=anomaly_results,
                    saved_visualizations=saved_visualizations,
                    title="Data Cleaning Report"
                )
                
                st.download_button(
                    "💾 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                
                # Show what's included
                pdf_includes = []
                if st.session_state.column_analysis:
                    pdf_includes.append(f"✅ {len(st.session_state.column_analysis)} column analyses")
                if anomaly_results:
                    pdf_includes.append(f"✅ {len(anomaly_results)} anomaly detections")
                if saved_visualizations:
                    pdf_includes.append(f"✅ {len(saved_visualizations)} visualizations")
                if st.session_state.cleaning_history:
                    pdf_includes.append(f"✅ {len(st.session_state.cleaning_history)} cleaning operations")
                
                if pdf_includes:
                    st.caption("**Included in PDF:** " + " | ".join(pdf_includes))
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                st.exception(e)
        elif export_format == 'html':
            html_content = report_generator.export_to_html(
                st.session_state.generated_reports, 
                "Data Cleaning Report"
            )
            st.download_button(
                "💾 Download Report",
                data=html_content,
                file_name=f"{filename}.html",
                mime="text/html",
                use_container_width=True
            )
        elif export_format == 'json':
            json_content = report_generator.export_to_json(
                st.session_state.generated_reports,
                {
                    'dataset_shape': df.shape,
                    'columns_analyzed': len(st.session_state.column_analysis),
                    'columns_cleaned': len(st.session_state.cleaning_history)
                }
            )
            st.download_button(
                "💾 Download Report",
                data=json_content,
                file_name=f"{filename}.json",
                mime="application/json",
                use_container_width=True
            )
        else:  # markdown
            # Combine all markdown reports
            combined_content = ""
            for report_type, content in st.session_state.generated_reports.items():
                combined_content += f"# {report_type.replace('_', ' ').title()}\n\n"
                combined_content += content + "\n\n---\n\n"
            
            st.download_button(
                "💾 Download Report",
                data=combined_content,
                file_name=f"{filename}.md",
                mime="text/markdown",
                use_container_width=True
            )

# Display generated reports
if st.session_state.get('generated_reports'):
    st.subheader("3. Report Preview")
    
    # Create tabs for each report type
    report_tabs = st.tabs([
        name.replace('_', ' ').title() for name in st.session_state.generated_reports.keys()
    ])
    
    for tab, (report_type, content) in zip(report_tabs, st.session_state.generated_reports.items()):
        with tab:
            st.markdown(content)
    
    # Visualizations section
    if include_visualizations:
        st.subheader("4. Supporting Visualizations")
        
        viz_tabs = st.tabs([
            "📊 Dataset Overview", 
            "❌ Missing Patterns", 
            "📈 Data Quality", 
            "🔗 Correlations"
        ])
        
        with viz_tabs[0]:
            st.markdown("### Dataset Overview")
            overview_fig = visualizer.plot_column_overview(df)
            st.plotly_chart(overview_fig, use_container_width=True)
            
            # Basic statistics table
            st.markdown("### Basic Dataset Statistics")
            
            stats_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Columns Analyzed', 'Columns Cleaned'],
                'Value': [
                    f"{len(df):,}",
                    len(df.columns),
                    f"{df.isnull().sum().sum():,}",
                    len(st.session_state.column_analysis),
                    len(st.session_state.cleaning_history)
                ]
            }
            
            if original_df is not None:
                original_missing = original_df.isnull().sum().sum()
                current_missing = df.isnull().sum().sum()
                missing_reduction = original_missing - current_missing
                
                stats_data['Metric'].append('Missing Values Reduced')
                stats_data['Value'].append(f"{missing_reduction:,}")
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width='stretch', hide_index=True)
        
        with viz_tabs[1]:
            st.markdown("### Missing Data Patterns")
            missing_fig = visualizer.plot_missing_patterns(df)
            st.plotly_chart(missing_fig, use_container_width=True)
            
            # Missing data summary
            missing_summary = df.isnull().sum().sort_values(ascending=False)
            missing_summary = missing_summary[missing_summary > 0]
            
            if len(missing_summary) > 0:
                st.markdown("### Columns with Missing Data")
                missing_df = pd.DataFrame({
                    'Column': missing_summary.index,
                    'Missing Count': missing_summary.values,
                    'Missing Percentage': (missing_summary.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, width='stretch', hide_index=True)
            else:
                st.success("✅ No missing data in the current dataset!")
        
        with viz_tabs[2]:
            st.markdown("### Data Quality Scores")
            
            # Quality scores visualization
            if st.session_state.column_analysis:
                quality_data = []
                for col, analysis in st.session_state.column_analysis.items():
                    quality_score = analysis.get('data_quality', {}).get('score', 0)
                    quality_grade = analysis.get('data_quality', {}).get('grade', 'F')
                    
                    quality_data.append({
                        'Column': col,
                        'Quality Score': quality_score,
                        'Grade': quality_grade,
                        'Status': '🧹 Cleaned' if col in st.session_state.cleaning_history else '⏳ Pending'
                    })
                
                if quality_data:
                    quality_df = pd.DataFrame(quality_data)
                    
                    # Sort by quality score
                    quality_df = quality_df.sort_values('Quality Score', ascending=False)
                    
                    st.dataframe(quality_df, width='stretch', hide_index=True)
                    
                    # Quality distribution
                    avg_quality = quality_df['Quality Score'].mean()
                    st.metric("Average Quality Score", f"{avg_quality:.1f}/100")
                    
                    grade_counts = quality_df['Grade'].value_counts()
                    st.write("**Grade Distribution:**")
                    for grade in ['A', 'B', 'C', 'D', 'F']:
                        if grade in grade_counts:
                            st.write(f"- Grade {grade}: {grade_counts[grade]} columns")
            else:
                st.info("No quality analysis available. Please analyze columns first.")
        
        with viz_tabs[3]:
            st.markdown("### Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_fig = visualizer.plot_correlation_matrix(df)
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                # High correlation pairs
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # Avoid duplicates
                            corr_val = corr_matrix.loc[col1, col2]
                            if abs(corr_val) > 0.7:  # High correlation threshold
                                high_corr_pairs.append({
                                    'Column 1': col1,
                                    'Column 2': col2,
                                    'Correlation': round(corr_val, 3)
                                })
                
                if high_corr_pairs:
                    st.markdown("### High Correlation Pairs (|r| > 0.7)")
                    corr_df = pd.DataFrame(high_corr_pairs)
                    st.dataframe(corr_df, width='stretch', hide_index=True)
                else:
                    st.info("No high correlation pairs found.")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
    
    st.divider()
    
    st.subheader("5. Custom Visualizations")
    
    saved_visualizations = st.session_state.get('saved_visualizations', [])
    
    if saved_visualizations:
        st.success(f"✅ {len(saved_visualizations)} custom visualization(s) included in report")
        
        for idx, viz in enumerate(saved_visualizations):
            st.markdown(f"#### {viz['name']}")
            st.plotly_chart(viz['fig'], use_container_width=True)
            st.caption(f"Type: {viz['type'].title()} | Columns: {', '.join(viz['columns'])}")
            st.divider()
    else:
        st.info("No custom visualizations saved. Visit the Visualization page to create and save charts.")
    
    st.divider()

# Cleaning summary
st.subheader("6. Cleaning Operations Summary")

if st.session_state.cleaning_history:
    # Operations by column
    operations_summary = []
    total_operations = 0
    
    for column, operations in st.session_state.cleaning_history.items():
        total_operations += len(operations)
        
        # Get most recent operation
        if operations:
            last_op = operations[-1]
            operations_summary.append({
                'Column': column,
                'Operations Applied': len(operations),
                'Last Method': last_op.get('method_name', 'Unknown'),
                'Last Applied': last_op.get('timestamp', '')[:19] if last_op.get('timestamp') else 'Unknown'
            })
    
    if operations_summary:
        st.markdown(f"### Summary: {total_operations} operations across {len(operations_summary)} columns")
        
        operations_df = pd.DataFrame(operations_summary)
        st.dataframe(operations_df, width='stretch', hide_index=True)
        
        # Operations timeline
        st.markdown("### Operations Timeline")
        
        all_operations = []
        for column, operations in st.session_state.cleaning_history.items():
            for op in operations:
                if op.get('timestamp'):
                    all_operations.append({
                        'Timestamp': op['timestamp'][:19],
                        'Column': column,
                        'Method': op.get('method_name', 'Unknown'),
                        'Type': op.get('method_type', 'Unknown')
                    })
        
        if all_operations:
            # Sort by timestamp
            timeline_df = pd.DataFrame(all_operations)
            timeline_df = timeline_df.sort_values('Timestamp', ascending=False)
            st.dataframe(timeline_df.head(20), width='stretch', hide_index=True)
            
            if len(timeline_df) > 20:
                st.caption(f"Showing 20 most recent operations out of {len(timeline_df)} total")

else:
    st.info("No cleaning operations performed yet.")

# Report statistics
st.subheader("6. Report Statistics")

stats_cols = st.columns(4)

with stats_cols[0]:
    analyzed_pct = (len(st.session_state.column_analysis) / len(df.columns) * 100) if len(df.columns) > 0 else 0
    st.metric("Analysis Progress", f"{analyzed_pct:.0f}%")

with stats_cols[1]:
    cleaned_pct = (len(st.session_state.cleaning_history) / len(df.columns) * 100) if len(df.columns) > 0 else 0
    st.metric("Cleaning Progress", f"{cleaned_pct:.0f}%")

with stats_cols[2]:
    if original_df is not None:
        original_missing = original_df.isnull().sum().sum()
        current_missing = df.isnull().sum().sum()
        if original_missing > 0:
            missing_reduction_pct = ((original_missing - current_missing) / original_missing * 100)
            st.metric("Missing Data Reduced", f"{missing_reduction_pct:.0f}%")
        else:
            st.metric("Missing Data Reduced", "0%")
    else:
        st.metric("Missing Data Reduced", "N/A")

with stats_cols[3]:
    if st.session_state.column_analysis:
        avg_quality = np.mean([
            analysis.get('data_quality', {}).get('score', 0) 
            for analysis in st.session_state.column_analysis.values()
        ])
        st.metric("Avg Quality Score", f"{avg_quality:.0f}/100")
    else:
        st.metric("Avg Quality Score", "N/A")

# Recommendations and next steps
st.subheader("7. 📋 Recommendations & Next Steps")

recommendations = []

# Analysis recommendations
unanalyzed_cols = set(df.columns) - set(st.session_state.column_analysis.keys())
if unanalyzed_cols:
    recommendations.append({
        'type': 'analysis',
        'priority': 'high',
        'title': f'Analyze {len(unanalyzed_cols)} remaining columns',
        'description': f'Columns not yet analyzed: {", ".join(list(unanalyzed_cols)[:5])}{"..." if len(unanalyzed_cols) > 5 else ""}'
    })

# Cleaning recommendations
uncleaned_cols = set(df.columns) - set(st.session_state.cleaning_history.keys())
if uncleaned_cols:
    recommendations.append({
        'type': 'cleaning',
        'priority': 'medium',
        'title': f'Consider cleaning {len(uncleaned_cols)} columns',
        'description': f'Columns that might benefit from cleaning: {", ".join(list(uncleaned_cols)[:5])}{"..." if len(uncleaned_cols) > 5 else ""}'
    })

# Quality recommendations
if st.session_state.column_analysis:
    low_quality_cols = [
        col for col, analysis in st.session_state.column_analysis.items()
        if analysis.get('data_quality', {}).get('score', 0) < 70
    ]
    
    if low_quality_cols:
        recommendations.append({
            'type': 'quality',
            'priority': 'high',
            'title': f'Address {len(low_quality_cols)} low-quality columns',
            'description': f'Columns with quality score < 70: {", ".join(low_quality_cols[:5])}{"..." if len(low_quality_cols) > 5 else ""}'
        })

# Validation recommendations
if st.session_state.cleaning_history:
    recommendations.append({
        'type': 'validation',
        'priority': 'medium',
        'title': 'Validate cleaning results',
        'description': 'Review cleaned data and verify that changes align with your analysis goals'
    })

if recommendations:
    for rec in recommendations:
        priority_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        priority_icon = priority_colors.get(rec['priority'], '⚪')
        
        with st.expander(f"{priority_icon} {rec['title']} ({rec['priority']} priority)"):
            st.write(rec['description'])
            
            if rec['type'] == 'analysis':
                if st.button("🔍 Go to Column Analysis", key=f"rec_analysis_{rec['title'][:10]}"):
                    st.switch_page("pages/2_Column_Analysis.py")
            elif rec['type'] == 'cleaning':
                if st.button("🧹 Go to Cleaning Wizard", key=f"rec_cleaning_{rec['title'][:10]}"):
                    st.switch_page("pages/3_Cleaning_Wizard.py")
            elif rec['type'] == 'quality':
                if st.button("🔍 Analyze Quality Issues", key=f"rec_quality_{rec['title'][:10]}"):
                    st.switch_page("pages/2_Column_Analysis.py")
else:
    st.success("🎉 Great job! Your dataset appears to be well-analyzed and cleaned.")

# Sidebar with quick stats
with st.sidebar:
    st.markdown("### 📊 Report Summary")
    
    # Dataset overview
    st.metric("Dataset Size", f"{len(df):,} × {len(df.columns)}")
    
    # Progress metrics
    progress_data = {
        'Analysis': len(st.session_state.column_analysis) / len(df.columns) if len(df.columns) > 0 else 0,
        'Cleaning': len(st.session_state.cleaning_history) / len(df.columns) if len(df.columns) > 0 else 0
    }
    
    for metric, value in progress_data.items():
        st.progress(value)
        st.write(f"{metric}: {value*100:.0f}%")
    
    # Quality overview
    if st.session_state.column_analysis:
        quality_scores = [
            analysis.get('data_quality', {}).get('score', 0)
            for analysis in st.session_state.column_analysis.values()
        ]
        
        avg_quality = np.mean(quality_scores)
        quality_color = "green" if avg_quality >= 80 else "orange" if avg_quality >= 60 else "red"
        
        st.markdown(f"""
        **Average Quality Score:**  
        <span style="color: {quality_color}; font-size: 1.2em; font-weight: bold;">
        {avg_quality:.0f}/100</span>
        """, unsafe_allow_html=True)
    
    # Report status
    st.markdown("### 📋 Report Status")
    
    if st.session_state.get('generated_reports'):
        st.success("✅ Reports Generated")
        report_count = len(st.session_state.generated_reports)
        st.write(f"**Reports:** {report_count}")
        
        timestamp = st.session_state.get('report_timestamp')
        if timestamp:
            st.write(f"**Generated:** {timestamp.strftime('%H:%M:%S')}")
    else:
        st.info("📋 No reports generated yet")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("📤 Export All Data", width='stretch'):
        csv_data = df.to_csv(index=False)
        st.download_button(
            "💾 Download CSV",
            data=csv_data,
            file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    if st.button("🔍 Quick Analysis", width='stretch'):
        st.switch_page("pages/2_Column_Analysis.py")
    
    if st.button("🤖 AI Assistant", width='stretch'):
        st.switch_page("pages/6_AI_Assistant.py")

# Footer
st.markdown("---")
st.markdown("**📝 Report Generation Complete**")
st.markdown("Your data cleaning reports provide comprehensive documentation of all analysis and cleaning operations performed on your dataset. These reports ensure reproducibility and provide audit trails for statistical agencies.")
