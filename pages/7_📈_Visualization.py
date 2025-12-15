import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.utils import initialize_session_state
from modules.visualization import DataVisualizer
import io
import base64



st.title("📊 Interactive Data Visualization")

st.markdown("""
Create custom visualizations to explore your data. All visualizations automatically reflect your cleaned data 
and can be saved as static images for inclusion in PDF reports.
""")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset
original_df = st.session_state.original_dataset
visualizer = DataVisualizer()

if 'saved_visualizations' not in st.session_state:
    st.session_state.saved_visualizations = []

# ===== CUSTOM VISUALIZATION BUILDER =====
st.subheader("🎨 Custom Visualization Builder")

st.markdown("""
Select columns and chart type to create custom visualizations. All charts update in real-time to reflect 
your cleaned data and can be saved as static images for PDF reports.
""")

# Column selection and chart type
col1, col2 = st.columns([3, 1])

with col1:
    # Get column types for filtering
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()
    
    selected_columns = st.multiselect(
        "📋 Select column(s) for visualization:",
        options=all_cols,
        help="Choose one or more columns. Available chart types will update based on your selection."
    )

with col2:
    viz_type = st.selectbox(
        "📊 Chart Type:",
        options=['bar', 'line', 'scatter', 'box', 'violin', 'histogram', 'kde', 'qq', 'pie', 'heatmap', 'correlation'],
        format_func=lambda x: {
            'bar': '📊 Bar Chart',
            'line': '📈 Line Chart',
            'scatter': '🔵 Scatter Plot',
            'box': '📦 Box Plot',
            'violin': '🎻 Violin Plot',
            'histogram': '📊 Histogram',
            'kde': '📈 KDE Plot',
            'qq': '📉 Q-Q Plot',
            'pie': '🥧 Pie Chart',
            'heatmap': '🔥 Heatmap',
            'correlation': '🔗 Correlation Matrix'
        }[x]
    )

# Chart configuration
if selected_columns:
    with st.expander("⚙️ Chart Configuration", expanded=True):
        config_cols = st.columns(3)
        
        with config_cols[0]:
            chart_title = st.text_input(
                "Chart Title:",
                value=f"{viz_type.title()} - {', '.join(selected_columns[:2])}{' ...' if len(selected_columns) > 2 else ''}"
            )
        
        with config_cols[1]:
            chart_height = st.slider("Chart Height (px):", 300, 800, 500, 50)
        
        with config_cols[2]:
            show_legend = st.checkbox("Show Legend", value=True)
    
    # Generate visualization button
    if st.button("🎨 Generate Visualization", type="primary", use_container_width=True):
        try:
            fig = None
            error_message = None
            
            # Bar Chart
            if viz_type == 'bar':
                if len(selected_columns) == 1:
                    col = selected_columns[0]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # For numeric columns, show distribution
                        value_counts = df[col].value_counts().head(20).sort_index()
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   labels={'x': col, 'y': 'Count'}, title=chart_title)
                    else:
                        # For categorical columns
                        value_counts = df[col].value_counts().head(20)
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   labels={'x': col, 'y': 'Count'}, title=chart_title)
                        fig.update_xaxes(tickangle=-45)
                elif len(selected_columns) == 2:
                    x_col, y_col = selected_columns[0], selected_columns[1]
                    if pd.api.types.is_numeric_dtype(df[y_col]):
                        fig = px.bar(df, x=x_col, y=y_col, title=chart_title)
                    else:
                        error_message = "For 2-column bar charts, the second column should be numeric."
                else:
                    error_message = "Bar charts work best with 1-2 columns."
            
            # Line Chart
            elif viz_type == 'line':
                if len(selected_columns) >= 1:
                    fig = go.Figure()
                    for col in selected_columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig.add_trace(go.Scatter(
                                y=df[col].dropna(),
                                mode='lines',
                                name=col
                            ))
                    fig.update_layout(title=chart_title, xaxis_title="Index", yaxis_title="Value")
                else:
                    error_message = "Please select at least one numeric column for line charts."
            
            # Scatter Plot
            elif viz_type == 'scatter':
                if len(selected_columns) >= 2:
                    x_col, y_col = selected_columns[0], selected_columns[1]
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        color_col = selected_columns[2] if len(selected_columns) > 2 else None
                        size_col = selected_columns[3] if len(selected_columns) > 3 else None
                        
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                                       title=chart_title, hover_data=selected_columns[:4])
                    else:
                        error_message = "First two columns must be numeric for scatter plots."
                else:
                    error_message = "Scatter plots require at least 2 numeric columns."
            
            # Box Plot
            elif viz_type == 'box':
                fig = go.Figure()
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig.add_trace(go.Box(
                            y=df[col].dropna(),
                            name=col,
                            boxmean='sd'
                        ))
                if fig.data:
                    fig.update_layout(title=chart_title, yaxis_title="Value", showlegend=show_legend)
                else:
                    error_message = "No numeric columns selected for box plot."
            
            # Violin Plot
            elif viz_type == 'violin':
                fig = go.Figure()
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig.add_trace(go.Violin(
                            y=df[col].dropna(),
                            name=col,
                            box_visible=True,
                            meanline_visible=True
                        ))
                if fig.data:
                    fig.update_layout(title=chart_title, yaxis_title="Value", showlegend=show_legend)
                else:
                    error_message = "No numeric columns selected for violin plot."
            
            # Histogram
            elif viz_type == 'histogram':
                fig = go.Figure()
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig.add_trace(go.Histogram(
                            x=df[col].dropna(),
                            name=col,
                            opacity=0.7
                        ))
                if fig.data:
                    fig.update_layout(title=chart_title, barmode='overlay', showlegend=show_legend)
                else:
                    error_message = "No numeric columns selected for histogram."
            
            # KDE Plot (Kernel Density Estimate)
            elif viz_type == 'kde':
                from scipy import stats
                fig = go.Figure()
                colors = ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A']
                for idx, col in enumerate(selected_columns):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        data = df[col].dropna()
                        if len(data) > 1:
                            # Calculate KDE with Scott's bandwidth (default, good for most cases)
                            kde = stats.gaussian_kde(data, bw_method='scott')
                            
                            # Extend range beyond data for better visualization
                            data_range = data.max() - data.min()
                            x_range = np.linspace(data.min() - 0.1 * data_range, 
                                                data.max() + 0.1 * data_range, 300)
                            y_kde = kde(x_range)
                            
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=y_kde,
                                name=col,
                                mode='lines',
                                fill='tozeroy',
                                opacity=0.6,
                                line=dict(color=colors[idx % len(colors)], width=2)
                            ))
                if fig.data:
                    fig.update_layout(
                        title=chart_title, 
                        xaxis_title="Value", 
                        yaxis_title="Density",
                        showlegend=show_legend,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                else:
                    error_message = "No numeric columns with sufficient data for KDE plot."
            
            # Q-Q Plot (Quantile-Quantile)
            elif viz_type == 'qq':
                from scipy import stats
                if len(selected_columns) == 1:
                    col = selected_columns[0]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        data = df[col].dropna()
                        if len(data) > 1:
                            # Calculate Q-Q plot points using scipy.stats.probplot
                            qq_data = stats.probplot(data, dist="norm")
                            theoretical_quantiles = qq_data[0][0]
                            sample_quantiles = qq_data[0][1]
                            slope = qq_data[1][0]
                            intercept = qq_data[1][1]
                            
                            # Create figure
                            fig = go.Figure()
                            
                            # Add data points
                            fig.add_trace(go.Scatter(
                                x=theoretical_quantiles,
                                y=sample_quantiles,
                                mode='markers',
                                name='Sample Data',
                                marker=dict(
                                    color='#19d3f3',
                                    size=6,
                                    opacity=0.7,
                                    line=dict(width=0.5, color='DarkSlateGrey')
                                )
                            ))
                            
                            # Add fitted reference line (best fit)
                            x_line = np.array([theoretical_quantiles[0], theoretical_quantiles[-1]])
                            y_line = slope * x_line + intercept
                            fig.add_trace(go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode='lines',
                                name='Best Fit Line',
                                line=dict(color='#636efa', width=2)
                            ))
                            
                            # Add 45-degree line for perfect normal distribution
                            fig.add_trace(go.Scatter(
                                x=x_line,
                                y=x_line,
                                mode='lines',
                                name='Perfect Normal',
                                line=dict(color='red', width=1, dash='dash'),
                                opacity=0.5
                            ))
                            
                            fig.update_layout(
                                title=chart_title,
                                xaxis_title="Theoretical Quantiles (Normal Distribution)",
                                yaxis_title="Sample Quantiles",
                                showlegend=show_legend,
                                template='plotly_white',
                                hovermode='closest'
                            )
                            
                            # Add annotation explaining the plot
                            fig.add_annotation(
                                text="Points close to the line suggest normal distribution",
                                xref="paper", yref="paper",
                                x=0.02, y=0.98,
                                showarrow=False,
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="gray",
                                borderwidth=1,
                                font=dict(size=10)
                            )
                        else:
                            error_message = "Insufficient data for Q-Q plot."
                    else:
                        error_message = "Q-Q plot requires a numeric column."
                else:
                    error_message = "Q-Q plot works with exactly 1 numeric column."
            
            # Pie Chart
            elif viz_type == 'pie':
                if len(selected_columns) == 1:
                    col = selected_columns[0]
                    value_counts = df[col].value_counts().head(15)
                    fig = go.Figure(data=[go.Pie(
                        labels=value_counts.index,
                        values=value_counts.values,
                        textinfo='label+percent',
                        hole=0.3
                    )])
                    fig.update_layout(title=chart_title)
                else:
                    error_message = "Pie charts work with exactly 1 column."
            
            # Heatmap
            elif viz_type == 'heatmap':
                if len(selected_columns) >= 2:
                    numeric_selected = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
                    if len(numeric_selected) >= 2:
                        corr_matrix = df[numeric_selected].corr()
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu',
                            zmid=0,
                            text=corr_matrix.values.round(2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            colorbar=dict(title="Correlation")
                        ))
                        fig.update_layout(title=chart_title)
                    else:
                        error_message = "Need at least 2 numeric columns for heatmap."
                else:
                    error_message = "Heatmap requires at least 2 columns."
            
            # Correlation Matrix
            elif viz_type == 'correlation':
                numeric_selected = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_selected) >= 2:
                    fig = visualizer.plot_correlation_matrix(df[numeric_selected])
                    fig.update_layout(title=chart_title)
                else:
                    error_message = "Need at least 2 numeric columns for correlation matrix."
            
            # Display result
            if error_message:
                st.error(f"❌ {error_message}")
            elif fig:
                # Update figure layout
                fig.update_layout(height=chart_height, showlegend=show_legend)
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data quality indicator
                if original_df is not None:
                    orig_missing = sum(original_df[col].isnull().sum() for col in selected_columns if col in original_df.columns)
                    curr_missing = sum(df[col].isnull().sum() for col in selected_columns)
                    
                    if orig_missing != curr_missing:
                        st.success(f"✅ Chart reflects cleaned data (reduced {orig_missing - curr_missing} missing values)")
                    else:
                        st.info("ℹ️ Chart shows current dataset state")
                
                # Save to session state for later
                st.session_state['current_visualization'] = {
                    'fig': fig,
                    'title': chart_title,
                    'columns': selected_columns,
                    'type': viz_type
                }
                
                # Save button
                save_cols = st.columns([1, 1, 2])
                
                with save_cols[0]:
                    if st.button("💾 Save to Report", use_container_width=True):
                        try:
                            # Convert to static image
                            img_bytes = fig.to_image(format="png", width=1200, height=chart_height, scale=2)
                            img_b64 = base64.b64encode(img_bytes).decode()
                            
                            # Save with metadata
                            st.session_state.saved_visualizations.append({
                                'name': chart_title,
                                'type': viz_type,
                                'columns': selected_columns,
                                'title': chart_title,
                                'fig': fig,
                                'img_b64': img_b64,
                                'img_bytes': img_bytes
                            })
                            
                            st.success(f"✅ Visualization saved! ({len(st.session_state.saved_visualizations)} total)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving visualization: {str(e)}")
                
                with save_cols[1]:
                    # Download button for individual chart
                    try:
                        img_bytes = fig.to_image(format="png", width=1200, height=chart_height, scale=2)
                        st.download_button(
                            label="📥 Download PNG",
                            data=img_bytes,
                            file_name=f"{chart_title.replace(' ', '_').lower()}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.caption(f"Download unavailable: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error generating visualization: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Select columns above to start creating visualizations")

st.divider()

# ===== SAVED VISUALIZATIONS =====
st.subheader("💾 Saved Visualizations for PDF Reports")

if st.session_state.saved_visualizations:
    st.success(f"📊 {len(st.session_state.saved_visualizations)} visualization(s) saved and ready for PDF export")
    
    # Display saved visualizations
    for idx, viz in enumerate(st.session_state.saved_visualizations):
        with st.expander(f"📈 {idx + 1}. {viz['name']}", expanded=False):
            col_display, col_actions = st.columns([4, 1])
            
            with col_display:
                st.plotly_chart(viz['fig'], use_container_width=True)
                st.caption(f"**Type:** {viz['type'].title()} | **Columns:** {', '.join(viz['columns'])}")
            
            with col_actions:
                if st.button("🗑️ Remove", key=f"remove_viz_{idx}", use_container_width=True):
                    st.session_state.saved_visualizations.pop(idx)
                    st.rerun()
                
                # Download individual visualization
                try:
                    st.download_button(
                        label="📥 Download",
                        data=viz['img_bytes'],
                        file_name=f"{viz['name'].replace(' ', '_').lower()}.png",
                        mime="image/png",
                        key=f"download_viz_{idx}",
                        use_container_width=True
                    )
                except:
                    st.caption("Download unavailable")
    
    # Bulk actions
    st.markdown("---")
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("🗑️ Clear All Visualizations", use_container_width=True):
            st.session_state.saved_visualizations = []
            st.success("✅ All visualizations cleared")
            st.rerun()
    
    with action_cols[1]:
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
            st.switch_page("pages/7_Reports.py")
else:
    st.info("No visualizations saved yet. Create and save visualizations above to include them in your PDF reports.")

st.divider()

# ===== QUICK INSIGHTS =====
st.subheader("💡 Data Quality Dashboard")

st.markdown("Quick overview of your dataset's current state:")

# Overview metrics
overview_cols = st.columns(4)

with overview_cols[0]:
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    st.metric("Missing Data", f"{missing_pct:.1f}%", 
             delta=f"{total_missing:,} cells",
             delta_color="inverse")

with overview_cols[1]:
    analyzed_count = len(st.session_state.column_analysis)
    total_cols = len(df.columns)
    st.metric("Analyzed Columns", f"{analyzed_count}/{total_cols}")

with overview_cols[2]:
    cleaned_count = len(st.session_state.cleaning_history)
    st.metric("Cleaned Columns", f"{cleaned_count}/{total_cols}")

with overview_cols[3]:
    if st.session_state.column_analysis:
        avg_quality = sum(
            analysis.get('data_quality', {}).get('score', 0) 
            for analysis in st.session_state.column_analysis.values()
        ) / len(st.session_state.column_analysis)
        st.metric("Avg Quality", f"{avg_quality:.0f}/100")
    else:
        st.metric("Avg Quality", "N/A")

# Quick visualization tips
with st.expander("💡 Visualization Tips"):
    st.markdown("""
    **Choosing the Right Chart Type:**
    
    - **Bar Chart**: Best for comparing categories or showing distribution of single variables
    - **Line Chart**: Ideal for showing trends over time or sequences
    - **Scatter Plot**: Perfect for exploring relationships between two numeric variables
    - **Box Plot**: Great for comparing distributions and identifying outliers
    - **Violin Plot**: Shows distribution density along with outlier information
    - **Histogram**: Excellent for understanding the frequency distribution of numeric data
    - **KDE Plot**: Visualizes the probability density function of numeric data with smooth curves
    - **Q-Q Plot**: Assesses if data follows a normal distribution by comparing sample quantiles with theoretical quantiles
    - **Pie Chart**: Use for showing proportions (works best with <10 categories)
    - **Heatmap/Correlation**: Visualize relationships between multiple numeric variables
    
    **Multi-Column Selection:**
    - 1 column: Bar, line, histogram, KDE, Q-Q, pie, box, violin
    - 2 columns: Scatter, bar (x vs y), heatmap
    - 3+ columns: Scatter (with color/size), heatmap, correlation matrix
    
    **Real-Time Updates:**
    - All visualizations automatically use your current cleaned dataset
    - Changes from the Cleaning Wizard are immediately reflected
    - Compare with original data to see cleaning impact
    
    **PDF Report Integration:**
    - Save visualizations as static PNG images
    - Saved charts are automatically included in PDF reports
    - High resolution (1200px width) for professional quality
    """)

st.divider()

# Navigation
st.markdown("### 📋 Next Steps")

nav_cols = st.columns(3)

with nav_cols[0]:
    if st.button("🔍 Analyze More Columns", use_container_width=True):
        st.switch_page("pages/2_Column_Analysis.py")

with nav_cols[1]:
    if st.button("🧹 Clean Data", use_container_width=True):
        st.switch_page("pages/3_Cleaning_Wizard.py")

with nav_cols[2]:
    if st.button("📊 Generate PDF Report", use_container_width=True, type="primary"):
        st.switch_page("pages/7_Reports.py")
