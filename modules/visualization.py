import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import plotly.figure_factory as ff

# Design System Colors
COLOR_PRIMARY = "#2563EB"  # Blue
COLOR_SECONDARY = "#7C3AED"  # Purple
COLOR_SUCCESS = "#10B981"  # Green
COLOR_WARNING = "#F59E0B"  # Amber
COLOR_ERROR = "#EF4444"  # Red
COLOR_INFO = "#3B82F6"  # Light Blue
COLOR_LIGHT = "#DBEAFE"  # Light Blue
COLOR_NEUTRAL = "#6B7280"  # Gray

# Color palette for multi-series charts
COLOR_PALETTE = [
    COLOR_PRIMARY,      # Blue
    COLOR_SUCCESS,      # Green
    COLOR_WARNING,      # Amber
    COLOR_SECONDARY,    # Purple
    COLOR_INFO,        # Light Blue
    COLOR_ERROR,       # Red
    "#EC4899",         # Pink
    "#14B8A6",         # Teal
]

class DataVisualizer:
    """Comprehensive visualization module for data cleaning assistant"""
    
    def __init__(self):
        self.color_palette = COLOR_PALETTE
        self.plot_config = {
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawrect', 'eraseshape']
        }
        self._corr_cache = {}
    
    def plot_missing_patterns(self, df: pd.DataFrame, max_cols: int = 50) -> go.Figure:
        """Create heatmap of missing value patterns - optimized for large datasets"""
        # Limit columns for visualization performance
        if len(df.columns) > max_cols:
            cols_to_show = df.columns[:max_cols].tolist()
            df_viz = df[cols_to_show]
            title_suffix = f" (showing first {max_cols} columns)"
        else:
            df_viz = df
            title_suffix = ""
        
        # Sample rows if dataset is very large (>10000 rows)
        if len(df_viz) > 10000:
            # Sample every nth row to reduce memory usage
            sample_rate = len(df_viz) // 5000
            df_viz = df_viz.iloc[::sample_rate]
            title_suffix += f" (sampled {len(df_viz)} rows)"
        
        # Create missing value matrix - optimized
        missing_matrix = df_viz.isnull().astype(np.int8)  # Use int8 instead of int for memory
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.values.T,
            y=missing_matrix.columns,
            x=missing_matrix.index,
            colorscale=[[0, COLOR_SUCCESS], [1, COLOR_ERROR]],
            showscale=True,
            colorbar=dict(title="Missing Values", tickvals=[0, 1], ticktext=["Present", "Missing"])
        ))
        
        fig.update_layout(
            title=f"Missing Value Pattern{title_suffix}",
            xaxis_title="Row Index",
            yaxis_title="Columns",
            height=max(400, min(len(df_viz.columns) * 20, 800)),  # Cap height
            xaxis=dict(showticklabels=False) if len(df_viz) > 1000 else {},
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="rgba(243, 244, 246, 1)",
            paper_bgcolor="white"
        )
        
        return fig
    
    def plot_column_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create overview plot of all columns with basic statistics"""
        stats = []
        for col in df.columns:
            col_stats = {
                'column': col,
                'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
                'unique_pct': (df[col].nunique() / len(df)) * 100,
                'dtype': str(df[col].dtype)
            }
            stats.append(col_stats)
        
        stats_df = pd.DataFrame(stats)
        
        fig = go.Figure()
        
        # Missing percentage bars
        fig.add_trace(go.Bar(
            name='Missing %',
            x=stats_df['column'],
            y=stats_df['missing_pct'],
            yaxis='y',
            marker_color=COLOR_ERROR,
            opacity=0.8
        ))
        
        # Unique percentage line
        fig.add_trace(go.Scatter(
            name='Unique %',
            x=stats_df['column'],
            y=stats_df['unique_pct'],
            yaxis='y2',
            mode='lines+markers',
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=8, color=COLOR_PRIMARY)
        ))
        
        fig.update_layout(
            title="Column Overview: Missing Values and Uniqueness",
            xaxis_title="Columns",
            yaxis=dict(title="Missing Percentage", side="left", color=COLOR_ERROR),
            yaxis2=dict(title="Unique Percentage", side="right", overlaying="y", color=COLOR_PRIMARY),
            height=500,
            xaxis=dict(tickangle=45),
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="rgba(243, 244, 246, 1)",
            paper_bgcolor="white"
        )
        
        return fig
    
    def plot_column_distribution(self, series: pd.Series, column_name: str) -> go.Figure:
        """Plot distribution for a specific column"""
        if pd.api.types.is_numeric_dtype(series):
            return self._plot_numeric_distribution(series, column_name)
        else:
            return self._plot_categorical_distribution(series, column_name)
    
    def _plot_numeric_distribution(self, series: pd.Series, column_name: str) -> go.Figure:
        """Plot numeric column distribution with statistics"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", 
                             x=0.5, y=0.5, 
                             xref="paper", yref="paper",
                             showarrow=False, font_size=20)
            fig.update_layout(title=f"{column_name} - No Data")
            return fig
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=non_null_series,
            name="Distribution",
            nbinsx=min(50, int(len(non_null_series) / 10)),
            marker_color=COLOR_PRIMARY,
            marker_line_width=1,
            marker_line_color="white",
            opacity=0.8
        ))
        
        # Add statistics lines
        mean_val = non_null_series.mean()
        median_val = non_null_series.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color=COLOR_ERROR, line_width=2,
                     annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dash", line_color=COLOR_SUCCESS, line_width=2,
                     annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")
        
        # Add quartiles
        q25 = non_null_series.quantile(0.25)
        q75 = non_null_series.quantile(0.75)
        fig.add_vline(x=q25, line_dash="dot", line_color=COLOR_WARNING, line_width=1.5, opacity=0.7)
        fig.add_vline(x=q75, line_dash="dot", line_color=COLOR_WARNING, line_width=1.5, opacity=0.7)
        
        fig.update_layout(
            title=f"{column_name} - Distribution",
            xaxis_title=column_name,
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="rgba(243, 244, 246, 1)",
            paper_bgcolor="white"
        )
        
        return fig
    
    def _plot_categorical_distribution(self, series: pd.Series, column_name: str) -> go.Figure:
        """Plot categorical column distribution"""
        value_counts = series.value_counts().head(20)  # Top 20 categories
        
        if len(value_counts) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", 
                             x=0.5, y=0.5, 
                             xref="paper", yref="paper",
                             showarrow=False, font_size=20)
            fig.update_layout(title=f"{column_name} - No Data")
            return fig
        
        # Use color gradient for categorical data
        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(value_counts))]
        
        fig = go.Figure(data=[
            go.Bar(x=value_counts.index, y=value_counts.values,
                   marker_color=colors,
                   marker_line_width=1,
                   marker_line_color="white")
        ])
        
        fig.update_layout(
            title=f"{column_name} - Top Categories",
            xaxis_title="Categories",
            yaxis_title="Count",
            height=400,
            xaxis=dict(tickangle=45),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="rgba(243, 244, 246, 1)",
            paper_bgcolor="white"
        )
        
        return fig
    
    def plot_outliers(self, series: pd.Series, column_name: str, outlier_results: Dict[str, Any]) -> go.Figure:
        """Plot outlier detection results"""
        if not pd.api.types.is_numeric_dtype(series):
            fig = go.Figure()
            fig.add_annotation(text="Outlier detection only available for numeric columns", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font_size=16)
            fig.update_layout(title=f"{column_name} - Not Numeric")
            return fig
        
        non_null_series = series.dropna()
        
        if len(non_null_series) < 10:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for outlier detection", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font_size=16)
            fig.update_layout(title=f"{column_name} - Insufficient Data")
            return fig
        
        fig = go.Figure()
        
        # Box plot
        fig.add_trace(go.Box(
            y=non_null_series,
            name="Data",
            boxpoints="outliers",
            marker_color=COLOR_PRIMARY,
            line_color=COLOR_PRIMARY,
            fillcolor=COLOR_LIGHT
        ))
        
        # Add outlier detection results if available
        if 'method_results' in outlier_results:
            iqr_results = outlier_results['method_results'].get('iqr', {})
            if iqr_results:
                # Highlight IQR outliers
                outlier_values = iqr_results.get('outlier_values', [])
                if outlier_values:
                    fig.add_trace(go.Scatter(
                        y=outlier_values,
                        x=['Outliers'] * len(outlier_values),
                        mode='markers',
                        name='IQR Outliers',
                        marker=dict(color=COLOR_ERROR, size=10, symbol='x', line=dict(width=2, color=COLOR_ERROR))
                    ))
        
        fig.update_layout(
            title=f"{column_name} - Outlier Detection",
            yaxis_title=column_name,
            height=400,
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="rgba(243, 244, 246, 1)",
            paper_bgcolor="white"
        )
        
        return fig
    
    def plot_before_after_comparison(self, before_series: pd.Series, after_series: pd.Series, 
                                   column_name: str, operation: str) -> go.Figure:
        """Create before/after comparison visualization"""
        fig = go.Figure()
        
        if pd.api.types.is_numeric_dtype(before_series):
            # Numeric comparison - histograms
            fig.add_trace(go.Histogram(
                x=before_series.dropna(),
                name="Before",
                opacity=0.7,
                marker_color=COLOR_ERROR,
                marker_line_width=1,
                marker_line_color="white",
                nbinsx=30
            ))
            
            fig.add_trace(go.Histogram(
                x=after_series.dropna(),
                name="After",
                opacity=0.7,
                marker_color=COLOR_SUCCESS,
                marker_line_width=1,
                marker_line_color="white",
                nbinsx=30
            ))
            
            fig.update_layout(
                title=f"{column_name} - Before/After {operation}",
                xaxis_title=column_name,
                yaxis_title="Frequency",
                barmode='overlay'
            )
        else:
            # Categorical comparison - bar charts
            before_counts = before_series.value_counts().head(10)
            after_counts = after_series.value_counts().head(10)
            
            # Align categories
            all_categories = list(set(before_counts.index) | set(after_counts.index))
            
            before_aligned = [before_counts.get(cat, 0) for cat in all_categories]
            after_aligned = [after_counts.get(cat, 0) for cat in all_categories]
            
            fig.add_trace(go.Bar(
                x=all_categories,
                y=before_aligned,
                name="Before",
                marker_color=COLOR_ERROR,
                marker_line_width=1,
                marker_line_color="white",
                opacity=0.8
            ))
            
            fig.add_trace(go.Bar(
                x=all_categories,
                y=after_aligned,
                name="After",
                marker_color=COLOR_SUCCESS,
                marker_line_width=1,
                marker_line_color="white",
                opacity=0.8
            ))
            
            fig.update_layout(
                title=f"{column_name} - Before/After {operation}",
                xaxis_title="Categories",
                yaxis_title="Count",
                barmode='group',
                xaxis=dict(tickangle=45)
            )
        
        fig.update_layout(
            height=400, 
            legend=dict(x=0.01, y=0.99),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="rgba(243, 244, 246, 1)",
            paper_bgcolor="white"
        )
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, max_cols: int = 20) -> go.Figure:
        """Create correlation matrix for numeric columns - optimized with caching"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns for correlation analysis", 
                             x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font_size=16)
            fig.update_layout(title="Correlation Matrix - No Numeric Data")
            return fig
        
        # Limit columns for visualization performance
        if len(numeric_df.columns) > max_cols:
            numeric_df = numeric_df.iloc[:, :max_cols]
        
        # Create cache key based on columns and data hash
        cols_tuple = tuple(numeric_df.columns)
        cache_key = (cols_tuple, len(numeric_df))
        
        # Check cache first
        if cache_key in self._corr_cache:
            corr_matrix = self._corr_cache[cache_key]
        else:
            # Calculate correlation matrix (optimized with numpy)
            corr_matrix = numeric_df.corr(method='pearson')
            self._corr_cache[cache_key] = corr_matrix
        
        # Create heatmap with professional colors
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0, COLOR_ERROR],      # Red for negative correlation
                [0.5, "white"],         # White for no correlation
                [1, COLOR_SUCCESS]      # Green for positive correlation
            ],
            zmid=0,
            colorbar=dict(title="Correlation", tickvals=[-1, 0, 1], ticktext=["-1.0", "0.0", "1.0"])
        ))
        
        fig.update_layout(
            title="Correlation Matrix (Numeric Columns)",
            height=600,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        return fig
    
    def create_summary_dashboard(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[go.Figure]:
        """Create a comprehensive summary dashboard"""
        figures = []
        
        # 1. Dataset Overview
        overview_fig = self.plot_column_overview(df)
        figures.append(overview_fig)
        
        # 2. Missing Pattern
        missing_fig = self.plot_missing_patterns(df)
        figures.append(missing_fig)
        
        # 3. Data Quality Summary
        quality_scores = []
        columns = []
        
        for col, analysis in analysis_results.items():
            if 'data_quality' in analysis:
                quality_scores.append(analysis['data_quality'].get('score', 0))
                columns.append(col)
        
        if quality_scores:
            # Use color gradient based on quality scores
            colors = [
                COLOR_SUCCESS if score >= 80 else COLOR_WARNING if score >= 60 else COLOR_ERROR
                for score in quality_scores
            ]
            
            quality_fig = go.Figure(data=[
                go.Bar(x=columns, y=quality_scores, marker_color=colors,
                       marker_line_width=1, marker_line_color="white")
            ])
            
            quality_fig.update_layout(
                title="Data Quality Scores by Column",
                xaxis_title="Columns",
                yaxis_title="Quality Score",
                height=400,
                xaxis=dict(tickangle=45),
                font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif", size=12),
                plot_bgcolor="rgba(243, 244, 246, 1)",
                paper_bgcolor="white"
            )
            
            figures.append(quality_fig)
        
        return figures
