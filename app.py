import streamlit as st
import pandas as pd
import numpy as np
import traceback



def main():
    from modules.utils import initialize_session_state, detect_column_types, create_backup
    from modules.data_analyzer import ColumnAnalyzer
    from modules.design_system import apply_global_styles
    import io
    
    # Apply global styling
    apply_global_styles()
    
    initialize_session_state()
    st.title("Renvo AI - Intelligent Data Cleaning Assistant")
    
    st.markdown("""
    Welcome to the Survey Data Cleaning Assistant - an AI-powered tool designed specifically for statistical agencies.
    This application analyzes each column individually and provides context-specific cleaning recommendations.
    
    ### Key Features:
    - **Individual Column Analysis**: Each column is analyzed separately with tailored recommendations
    - **AI-Powered Assistance**: Context-aware guidance using advanced language models
    - **Multiple Cleaning Strategies**: Various methods for handling missing values, outliers, and inconsistencies
    - **Comprehensive Audit Trail**: Track all cleaning operations with undo/redo functionality
    - **Statistical Rigor**: Maintain methodological consistency for survey data
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the pages in the sidebar to navigate through the application:")
    
    st.sidebar.markdown("""
    **Data Cleaning:**
    - Anomaly Detection - Detect and fix data issues
    - Data Transformation - Merge/split columns, expand JSON
    - Column Analysis - Detailed column analysis  
    - Cleaning Wizard - Apply cleaning methods
    
    **Data Analysis:**
    - Hypothesis Testing - Statistical hypothesis tests
    - Data Balancer - Balance datasets for ML
    
    **Data Visualization:**
    - Visualization - Create interactive charts
    - Reports - Generate cleaning reports
    
    **AI:**
    - AI Assistant - Get expert advice
    """)
    
    st.divider()
    
    # ========== DATA UPLOAD SECTION ==========
    st.header("📊 Data Upload & Configuration")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your survey dataset. Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Store in session state
            if st.session_state.dataset is None or not df.equals(st.session_state.dataset):
                st.session_state.dataset = df.copy()
                st.session_state.original_dataset = df.copy()
                
                # Auto-detect column types
                st.session_state.column_types = detect_column_types(df)
                
                # Clear previous analysis
                st.session_state.column_analysis = {}
                st.session_state.cleaning_history = {}
                st.session_state.undo_stack = []
                st.session_state.redo_stack = []
                
                st.info("🔍 Column types automatically detected. You can review and modify them below.")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.stop()
    
    # Display current dataset info if available
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        st.divider()
        st.subheader("📋 Dataset Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data preview
        st.subheader("🔍 Data Preview")
        
        preview_options = st.columns([3, 1])
        with preview_options[0]:
            preview_rows = st.slider("Number of rows to preview", min_value=5, max_value=min(100, len(df)), value=10)
        with preview_options[1]:
            show_info = st.checkbox("Show column info", value=False)
        
        if show_info:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Missing Count': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique Values': df.nunique()
            })
            
            st.dataframe(col_info, use_container_width=True)
        
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        st.divider()
        
        # Column type configuration
        st.subheader("⚙️ Column Type Configuration")
        
        st.markdown("""
        **Important:** Correct column types are crucial for appropriate cleaning recommendations. 
        Review the auto-detected types and adjust as needed.
        """)
        
        # Available column types
        type_options = [
            'continuous', 'integer', 'ordinal', 'categorical', 'binary', 
            'text', 'datetime', 'empty', 'unknown'
        ]
        
        type_descriptions = {
            'continuous': 'Continuous numeric data (e.g., age, income, measurements)',
            'integer': 'Integer numeric data (e.g., count of items, number of children)',
            'ordinal': 'Ordered categories (e.g., education level, satisfaction rating)',
            'categorical': 'Unordered categories (e.g., gender, region, occupation)',
            'binary': 'Two-category variables (e.g., yes/no, male/female)',
            'text': 'Free text data (e.g., comments, descriptions)',
            'datetime': 'Date and time information',
            'empty': 'Columns with no data',
            'unknown': 'Unable to determine type automatically'
        }
        
        # Display type legend
        with st.expander("📖 Column Type Guide"):
            for type_name, description in type_descriptions.items():
                st.write(f"**{type_name.title()}:** {description}")
        
        # Column type editor
        st.write("Review and adjust column types:")
        
        # Create columns for the editor
        cols = st.columns([3, 2, 2, 1])
        cols[0].write("**Column Name**")
        cols[1].write("**Detected Type**")
        cols[2].write("**Assigned Type**")
        cols[3].write("**Sample Values**")
        
        updated_types = {}
        
        for i, col in enumerate(df.columns):
            with st.container():
                editor_cols = st.columns([3, 2, 2, 1])
                
                with editor_cols[0]:
                    st.write(col)
                
                with editor_cols[1]:
                    detected_type = st.session_state.column_types.get(col, 'unknown')
                    st.write(f"`{detected_type}`")
                
                with editor_cols[2]:
                    current_type = st.session_state.column_types.get(col, 'unknown')
                    column_type_options = type_options.copy()
                    if current_type and current_type not in column_type_options:
                        column_type_options.append(current_type)
                    try:
                        type_index = column_type_options.index(current_type)
                    except ValueError:
                        type_index = column_type_options.index('unknown') if 'unknown' in column_type_options else 0
                    selected_type = st.selectbox(
                        f"Type for {col}",
                        column_type_options,
                        index=type_index,
                        key=f"type_{col}",
                        label_visibility="collapsed"
                    )
                    updated_types[col] = selected_type
                
                with editor_cols[3]:
                    sample_values = df[col].dropna().head(3).tolist()
                    sample_text = ", ".join([str(v)[:20] + "..." if len(str(v)) > 20 else str(v) for v in sample_values])
                    st.write(f"`{sample_text}`")
        
        # Update button
        col_update, col_analyze = st.columns([1, 1])
        
        with col_update:
            if st.button("💾 Update Column Types", type="primary", use_container_width=True):
                st.session_state.column_types = updated_types
                st.success("✅ Column types updated successfully!")
                st.rerun()
        
        with col_analyze:
            if st.button("🔍 Start Column Analysis", use_container_width=True):
                if any(updated_types.values()):
                    st.session_state.column_types = updated_types
                    
                    # Initialize analyzer and run basic analysis
                    analyzer = ColumnAnalyzer()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, col in enumerate(df.columns):
                        status_text.text(f"Analyzing column: {col}")
                        try:
                            analysis = analyzer.analyze_column(df, col)
                            st.session_state.column_analysis[col] = analysis
                        except Exception as e:
                            st.warning(f"⚠️ Error analyzing column {col}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(df.columns))
                    
                    status_text.text("Analysis complete!")
                    st.success("🎉 Column analysis completed! Navigate to the Column Analysis page to view results.")
                    
                    # Auto-navigate suggestion
                    st.info("💡 **Next Step:** Go to the **Column Analysis** page to review detailed analysis results for each column.")
                else:
                    st.error("Please configure column types before starting analysis.")
        
        st.divider()
        
        # Configuration export/import
        st.subheader("💾 Configuration Management")
        
        config_cols = st.columns([1, 1])
        
        with config_cols[0]:
            if st.button("📤 Export Configuration", use_container_width=True):
                from modules.utils import export_configuration
                config_json = export_configuration()
                st.download_button(
                    label="💾 Download Configuration",
                    data=config_json,
                    file_name=f"data_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with config_cols[1]:
            config_file = st.file_uploader(
                "📥 Import Configuration",
                type=['json'],
                help="Upload a previously exported configuration file"
            )
            
            if config_file is not None:
                try:
                    config_content = config_file.read().decode('utf-8')
                    from modules.utils import import_configuration
                    
                    if import_configuration(config_content):
                        st.success("✅ Configuration imported successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to import configuration")
                except Exception as e:
                    st.error(f"❌ Error importing configuration: {str(e)}")
        
        # Data quality warning
        if df.isnull().sum().sum() > len(df) * len(df.columns) * 0.2:
            st.warning("⚠️ **High Missing Data Rate:** This dataset has more than 20% missing values. Consider reviewing data collection processes.")
        
        if len(df) > 50000:
            st.info("ℹ️ **Large Dataset:** Processing may take longer for datasets with more than 50,000 rows. Consider using sampling for initial exploration.")
    
    else:
        st.info("👆 Please upload a dataset to get started with the data cleaning process.")
        
        # Help section for users without data
        with st.expander("📚 Getting Started Guide"):
            st.markdown("""
            ### How to Use This Data Cleaning Assistant
            
            1. **Upload Your Data**: Use the file uploader above to select your CSV or Excel file
            2. **Review Column Types**: The system will automatically detect column types, but you should verify them
            3. **Configure Settings**: Adjust any column types that weren't detected correctly
            4. **Start Analysis**: Click "Start Column Analysis" to begin the cleaning process
            
            ### Supported File Formats
            - **CSV files** (.csv): Comma-separated values
            - **Excel files** (.xlsx, .xls): Microsoft Excel formats
            
            ### Best Practices
            - Ensure your data has column headers in the first row
            - Remove any summary rows or metadata from the top of your file
            - Keep file sizes reasonable (under 100MB for best performance)
            - Have a clear understanding of what each column represents
            
            ### Column Type Importance
            Correctly identifying column types is crucial because:
            - **Continuous**: Gets outlier detection, normality tests, advanced imputation
            - **Categorical**: Gets frequency analysis, mode imputation, consistency checks
            - **Ordinal**: Gets order-preserving operations, median imputation
            - **Binary**: Gets specialized binary analysis and imputation
            - **Text**: Gets text cleaning, standardization, pattern analysis
            """)


if __name__ == "__main__":
    main()
