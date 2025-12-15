import streamlit as st
import pandas as pd
from modules.utils import initialize_session_state, create_backup, save_cleaning_operation
from datetime import datetime


def coerce_column_dtype(df: pd.DataFrame, column: str, expected_type: str):
    """
    Convert column to proper dtype based on expected_type while preserving nulls.
    
    Returns:
        tuple: (modified dataframe, conversion_applied boolean)
    """
    try:
        # Work on a copy of the series to avoid partial mutations
        original_series = df[column].copy()
        
        if expected_type in ['integer', 'float', 'numeric']:
            # Numeric conversion with errors='coerce' preserves NaN
            numeric_series = pd.to_numeric(original_series, errors='coerce')
            
            # Check if values are integers (safely handling NaN)
            non_null_values = numeric_series.dropna()
            is_integer_type = expected_type == 'integer' or (
                expected_type == 'numeric' and 
                len(non_null_values) > 0 and 
                all(float(x).is_integer() for x in non_null_values)
            )
            
            if is_integer_type:
                df[column] = numeric_series.astype('Int64')  # Nullable integer
            else:
                df[column] = numeric_series.astype('Float64')  # Nullable float
            return df, True
            
        elif expected_type == 'datetime':
            converted = pd.to_datetime(original_series, errors='coerce', utc=False)
            df[column] = converted
            return df, True
            
        elif expected_type == 'categorical':
            # Convert to string then categorical, preserving NA
            str_series = original_series.astype('string')
            categories = sorted(str_series.dropna().unique())
            df[column] = str_series.astype(pd.CategoricalDtype(categories=categories, ordered=False))
            return df, True
            
        elif expected_type == 'binary':
            # Normalize to True/False, preserve nulls, use nullable boolean dtype
            def normalize_binary(val):
                if pd.isna(val):
                    return pd.NA
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, str):
                    val_lower = str(val).lower().strip()
                    if val_lower in ['true', '1', 'yes', 'y', 't']:
                        return True
                    elif val_lower in ['false', '0', 'no', 'n', 'f']:
                        return False
                return pd.NA
            
            converted = original_series.apply(normalize_binary).astype('boolean')
            df[column] = converted
            return df, True
            
        elif expected_type == 'text':
            # Use pandas nullable string dtype to preserve NA
            converted = original_series.astype('string')
            df[column] = converted
            return df, True
            
        else:
            # Unknown/unsupported type - skip conversion
            return df, False
            
    except Exception as e:
        # If conversion fails, return unchanged (don't mutate df)
        return df, False

st.title("🔍 Data Type Anomaly Detection & Duplicate Removal")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

st.markdown("""
This section helps you detect and fix data quality issues:
- **Type Anomalies**: Values that don't match their declared data type (e.g., text in numeric columns)
- **Duplicate Rows**: Identical or similar rows that should be removed
- **Invalid Formats**: Date formats, unexpected values in binary/categorical columns
""")

df = st.session_state.dataset
column_types = st.session_state.column_types

# Tabs for different functionalities
tab1, tab2 = st.tabs(["🔍 Type Anomalies", "🗑️ Duplicate Removal"])

# ========== TAB 1: TYPE ANOMALIES ==========
with tab1:
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Select Column to Check")
        
        selected_column = st.selectbox(
            "Choose a column to analyze for anomalies",
            options=list(df.columns),
            help="Select which column you want to check for data type mismatches"
        )
    
    with col2:
        st.subheader("Column Information")
        if selected_column:
            expected_type = column_types.get(selected_column, 'unknown')
            st.metric("Expected Type", expected_type.title())
            st.metric("Total Values", len(df[selected_column]))
            st.metric("Null Values", df[selected_column].isnull().sum())
    
    st.divider()
    
    if selected_column:
        detector = st.session_state.anomaly_detector
        
        col_scan, col_refresh = st.columns([3, 1])
        with col_scan:
            st.subheader("2. Scan for Anomalies")
        with col_refresh:
            if st.button("🔄 Re-scan Column", use_container_width=True):
                if selected_column in st.session_state.anomaly_results:
                    del st.session_state.anomaly_results[selected_column]
                st.rerun()
        
        if selected_column not in st.session_state.anomaly_results:
            with st.spinner(f"Scanning {selected_column} for anomalies..."):
                expected_type = column_types.get(selected_column, 'unknown')
                anomaly_data = detector.detect_column_anomalies(df, selected_column, expected_type)
                st.session_state.anomaly_results[selected_column] = anomaly_data
                st.session_state.anomaly_last_updated = datetime.now().isoformat()
        
        anomaly_data = st.session_state.anomaly_results[selected_column]
        
        if anomaly_data['anomaly_count'] == 0:
            st.success(f"✅ {anomaly_data['summary']}")
        else:
            st.warning(f"⚠️ {anomaly_data['summary']}")
            
            st.divider()
            
            st.subheader("3. Review Anomalies")
            
            anomalies_df = pd.DataFrame(anomaly_data['anomalies'])
            
            st.dataframe(
                anomalies_df,
                use_container_width=True,
                column_config={
                    "row_index": st.column_config.NumberColumn("Row Index", help="Row number in dataset"),
                    "value": st.column_config.TextColumn("Anomalous Value", help="The problematic value"),
                    "reason": st.column_config.TextColumn("Reason", help="Why this value is anomalous")
                }
            )
            
            st.download_button(
                label="📥 Download Anomalies as CSV",
                data=anomalies_df.to_csv(index=False),
                file_name=f"anomalies_{selected_column}.csv",
                mime="text/csv"
            )
            
            st.divider()
            
            st.subheader("4. Fix Anomalies")
            
            fix_method = st.radio(
                "Choose how to handle these anomalies:",
                options=["Set All Anomalous Cells to Null", "Replace Values Individually"],
                help="Either set all anomalous cell values to null or replace specific values"
            )
            
            if fix_method == "Set All Anomalous Cells to Null":
                st.warning(f"⚠️ This will set **{anomaly_data['anomaly_count']} cell values** to null (NaN) in column '{selected_column}'. The rows will remain in the dataset.")
                
                col_confirm, col_cancel = st.columns(2)
                
                with col_confirm:
                    if st.button("🗑️ Confirm: Set Anomalous Cells to Null", type="primary", use_container_width=True):
                        create_backup()
                        
                        anomaly_indices = [a['row_index'] for a in anomaly_data['anomalies']]
                        cleaned_df, summary = detector.remove_anomalies(df, selected_column, anomaly_indices)
                        
                        # Apply column type conversion after fixing anomalies (setting to null)
                        expected_type = column_types.get(selected_column, 'unknown')
                        cleaned_df, conversion_applied = coerce_column_dtype(cleaned_df, selected_column, expected_type)
                        
                        st.session_state.dataset = cleaned_df
                        
                        save_cleaning_operation({
                            'column': selected_column,
                            'operation': 'remove_anomalies',
                            'details': summary,
                            'rows_affected': summary.get('cells_nullified', 0)
                        })
                        
                        del st.session_state.anomaly_results[selected_column]
                        
                        if conversion_applied:
                            st.success(f"✅ Successfully set {summary.get('cells_nullified', 0)} cell values to null and converted column type to {expected_type}!")
                        else:
                            st.success(f"✅ Successfully set {summary.get('cells_nullified', 0)} cell values to null!")
                        st.rerun()
                
                with col_cancel:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.info("Operation cancelled.")
            
            else:
                st.info("Select specific rows to replace values individually.")
                
                for idx, anomaly in enumerate(anomaly_data['anomalies']):
                    with st.expander(f"Row {anomaly['row_index']}: {anomaly['value']} - {anomaly['reason']}"):
                        col_show, col_replace = st.columns([2, 1])
                        
                        with col_show:
                            row_data = df.loc[anomaly['row_index']].to_dict()
                            st.json(row_data)
                        
                        with col_replace:
                            new_value = st.text_input(
                                "New value",
                                key=f"replace_{selected_column}_{anomaly['row_index']}",
                                placeholder="Enter replacement value"
                            )
                            
                            if st.button(f"Replace", key=f"btn_{selected_column}_{anomaly['row_index']}"):
                                if new_value:
                                    create_backup()
                                    
                                    modified_df, summary = detector.replace_anomaly(
                                        df.copy(),
                                        anomaly['row_index'],
                                        selected_column,
                                        new_value
                                    )
                                    
                                    # Apply column type conversion after fixing anomalies
                                    expected_type = column_types.get(selected_column, 'unknown')
                                    modified_df, conversion_applied = coerce_column_dtype(modified_df, selected_column, expected_type)
                                    
                                    st.session_state.dataset = modified_df
                                    
                                    save_cleaning_operation({
                                        'column': selected_column,
                                        'operation': 'replace_anomaly',
                                        'details': summary
                                    })
                                    
                                    del st.session_state.anomaly_results[selected_column]
                                    
                                    if conversion_applied:
                                        st.success(f"✅ Replaced value at row {anomaly['row_index']} and converted column type to {expected_type}")
                                    else:
                                        st.success(f"✅ Replaced value at row {anomaly['row_index']}")
                                    st.rerun()
                                else:
                                    st.warning("Please enter a replacement value")
                
                st.divider()
                
                st.subheader("Batch Replace")
                st.markdown("Replace multiple anomalies with the same value:")
                
                batch_value = st.text_input(
                    "Enter value to replace all anomalies in this column",
                    key=f"batch_{selected_column}",
                    placeholder="e.g., 0, NULL, Unknown"
                )
                
                if st.button(f"Replace All {anomaly_data['anomaly_count']} Anomalies with '{batch_value}'", 
                            disabled=not batch_value,
                            type="primary"):
                    create_backup()
                    
                    replacements = {a['row_index']: batch_value for a in anomaly_data['anomalies']}
                    modified_df, summary = detector.batch_replace_anomalies(
                        df.copy(),
                        selected_column,
                        replacements
                    )
                    
                    # Apply column type conversion after fixing anomalies
                    expected_type = column_types.get(selected_column, 'unknown')
                    modified_df, conversion_applied = coerce_column_dtype(modified_df, selected_column, expected_type)
                    
                    st.session_state.dataset = modified_df
                    
                    save_cleaning_operation({
                        'column': selected_column,
                        'operation': 'batch_replace_anomalies',
                        'details': summary
                    })
                    
                    del st.session_state.anomaly_results[selected_column]
                    
                    if conversion_applied:
                        st.success(f"✅ Replaced {summary['replacements_count']} anomalies and converted column type to {expected_type}!")
                    else:
                        st.success(f"✅ Replaced {summary['replacements_count']} anomalies!")
                    st.rerun()
    
    st.divider()
    
    with st.expander("📊 Scan All Columns for Anomalies", expanded=False):
        st.markdown("Scan all columns at once to get a complete overview of data type issues.")
        
        if st.button("🔍 Scan All Columns", use_container_width=True):
            with st.spinner("Scanning all columns for anomalies..."):
                all_anomalies = detector.detect_all_anomalies(df, column_types)
                
                if not all_anomalies:
                    st.success("✅ No anomalies detected in any column!")
                else:
                    st.warning(f"⚠️ Found anomalies in {len(all_anomalies)} column(s)")
                    
                    summary_data = []
                    for col, data in all_anomalies.items():
                        summary_data.append({
                            'Column': col,
                            'Expected Type': data['expected_type'],
                            'Anomaly Count': data['anomaly_count'],
                            'Percentage': f"{data['anomaly_percentage']:.2f}%"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    st.download_button(
                        label="📥 Download Full Anomaly Report",
                        data=summary_df.to_csv(index=False),
                        file_name="all_anomalies_summary.csv",
                        mime="text/csv"
                    )

# ========== TAB 2: DUPLICATE REMOVAL ==========
with tab2:
    st.divider()
    
    st.subheader("1. Detect Complete Duplicate Rows")
    
    st.info("""
    **Note:** This feature removes **complete duplicate rows** where all values match.
    - If no columns are selected: Entire rows must be identical to be considered duplicates
    - If specific columns are selected: Only those columns must match for rows to be duplicates
    """)
    
    # Detect duplicates
    total_duplicates = df.duplicated().sum()
    duplicate_rows = df[df.duplicated(keep=False)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Complete Duplicate Rows", f"{total_duplicates:,}")
    with col3:
        if len(df) > 0:
            dup_percentage = (total_duplicates / len(df)) * 100
            st.metric("Duplicate %", f"{dup_percentage:.2f}%")
        else:
            st.metric("Duplicate %", "0.00%")
    
    st.divider()
    
    # Duplicate detection options
    st.subheader("2. Configure Duplicate Detection")
    
    col_option1, col_option2 = st.columns(2)
    
    with col_option1:
        duplicate_subset = st.multiselect(
            "Select columns to check for duplicates (leave empty to check ALL columns)",
            options=list(df.columns),
            default=None,
            help="Choose specific columns to identify duplicates. If empty, all columns in the row must match for duplicates."
        )
    
    with col_option2:
        keep_option = st.selectbox(
            "Which duplicate to keep?",
            options=["first", "last", "none"],
            help="first: Keep first occurrence, last: Keep last occurrence, none: Remove all duplicates"
        )
    
    # Re-detect duplicates based on selected columns
    if duplicate_subset:
        total_duplicates_subset = df.duplicated(subset=duplicate_subset, keep=False).sum()
        st.warning(f"🔍 Found **{total_duplicates_subset:,}** rows where these columns are identical: {', '.join(duplicate_subset)}")
        st.caption(f"⚠️ **Complete rows** will be removed if the selected columns match, even if other columns differ.")
    else:
        st.success(f"🔍 Checking for **complete duplicate rows** (all {len(df.columns)} columns must match)")
        st.caption(f"✓ Only rows that are 100% identical across all columns will be removed.")
    
    st.divider()
    
    # Show duplicate rows
    if total_duplicates > 0 or (duplicate_subset and total_duplicates_subset > 0):
        st.subheader("3. Preview Duplicates")
        
        if duplicate_subset:
            preview_duplicates = df[df.duplicated(subset=duplicate_subset, keep=False)]
        else:
            preview_duplicates = df[df.duplicated(keep=False)]
        
        st.dataframe(
            preview_duplicates.head(100),
            use_container_width=True,
            height=300
        )
        
        if len(preview_duplicates) > 100:
            st.info(f"Showing first 100 of {len(preview_duplicates)} duplicate rows")
        
        st.download_button(
            label="📥 Download All Duplicates as CSV",
            data=preview_duplicates.to_csv(index=False),
            file_name=f"duplicates_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.divider()
        
        # Remove duplicates
        st.subheader("4. Remove Duplicates")
        
        if duplicate_subset:
            rows_to_remove = df.duplicated(subset=duplicate_subset, keep=keep_option).sum()
        else:
            rows_to_remove = df.duplicated(keep=keep_option).sum()
        
        st.warning(f"⚠️ This will remove **{rows_to_remove:,} rows** from your dataset (keeping '{keep_option}' occurrence)")
        
        col_remove, col_cancel = st.columns(2)
        
        with col_remove:
            if st.button("🗑️ Remove Duplicates", type="primary", use_container_width=True):
                create_backup()
                
                # Remove duplicates
                if duplicate_subset:
                    cleaned_df = df.drop_duplicates(subset=duplicate_subset, keep=keep_option)
                else:
                    cleaned_df = df.drop_duplicates(keep=keep_option)
                
                rows_removed = len(df) - len(cleaned_df)
                
                st.session_state.dataset = cleaned_df
                
                save_cleaning_operation({
                    'column': 'dataset',
                    'operation': 'remove_duplicates',
                    'details': {
                        'rows_removed': rows_removed,
                        'keep_option': keep_option,
                        'subset_columns': duplicate_subset if duplicate_subset else 'all columns'
                    },
                    'rows_affected': rows_removed
                })
                
                st.success(f"✅ Successfully removed {rows_removed:,} duplicate rows!")
                st.balloons()
                st.rerun()
        
        with col_cancel:
            if st.button("❌ Cancel", use_container_width=True):
                st.info("Operation cancelled.")
    
    else:
        st.success("✅ No duplicate rows found in your dataset!")
        st.info("Your data is clean and ready for analysis.")

st.divider()
st.info("💡 **Tip**: After fixing anomalies and removing duplicates, proceed to Column Analysis to examine data quality and patterns.")
