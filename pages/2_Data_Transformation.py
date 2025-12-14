import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state, create_backup
from modules.data_transformer import DataTransformer

initialize_session_state()

st.title("Data Transformation")

if st.session_state.dataset is None:
    st.warning("No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset
transformer = DataTransformer()

st.markdown("""
Transform your data by merging or splitting columns, and handle nested JSON/dictionary data.
All operations support various data types including dates, times, and complex structures.
""")

transformation_tabs = st.tabs([
    "Merge / Split Columns",
    "Expand JSON Data", 
    "Data Type Conversion"
])

with transformation_tabs[0]:
    st.header("Merge / Split Columns")
    
    operation_mode = st.toggle("Split Mode", value=False, help="Toggle between Merge and Split operations")
    
    if not operation_mode:
        st.subheader("Merge Columns")
        st.markdown("Combine multiple columns into a single column with custom separators.")
        
        available_columns = df.columns.tolist()
        merge_columns = st.multiselect(
            "Select columns to merge",
            options=available_columns,
            help="Select 2 or more columns to merge together"
        )
        
        if len(merge_columns) >= 2:
            is_datetime_merge = st.checkbox(
                "Date/Time merge",
                help="Enable for intelligent date/time component merging (year, month, day, hour, etc.)"
            )
            
            st.markdown("#### Separator Configuration")
            
            if len(merge_columns) == 2:
                separator = st.text_input(
                    "Separator between columns",
                    value="-",
                    help="Character(s) to use between merged values"
                )
                separators = [separator]
            else:
                st.markdown("Define a separator for each pair of adjacent columns:")
                separators = []
                cols = st.columns(len(merge_columns) - 1)
                for i, col in enumerate(cols):
                    with col:
                        sep = st.text_input(
                            f"Between '{merge_columns[i]}' and '{merge_columns[i+1]}'",
                            value="-",
                            key=f"sep_{i}"
                        )
                        separators.append(sep)
            
            new_column_name = st.text_input(
                "New column name",
                value=f"{'_'.join(merge_columns[:2])}_merged",
                help="Name for the new merged column"
            )
            
            handle_missing = st.selectbox(
                "Handle missing values",
                options=['skip', 'empty', 'null_string', 'fail'],
                format_func=lambda x: {
                    'skip': 'Skip missing values',
                    'empty': 'Replace with empty string',
                    'null_string': 'Replace with "NULL"',
                    'fail': 'Mark entire row as null if any value missing'
                }[x]
            )
            
            datetime_format = None
            if is_datetime_merge:
                st.markdown("#### DateTime Output Format")
                format_option = st.selectbox(
                    "Output format",
                    options=['auto', 'custom'],
                    help="Choose automatic formatting or specify a custom format"
                )
                if format_option == 'custom':
                    datetime_format = st.text_input(
                        "Custom datetime format",
                        value="%Y-%m-%d %H:%M:%S",
                        help="Use strftime format codes (e.g., %Y-%m-%d)"
                    )
            
            validation = transformer.validate_merge_columns(df, merge_columns, is_datetime_merge)
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    st.warning(warning)
            
            if validation['errors']:
                for error in validation['errors']:
                    st.error(error)
            
            st.markdown("#### Preview")
            preview_df = df[merge_columns].head(5).copy()
            st.dataframe(preview_df, use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Apply Merge", type="primary", disabled=not validation['valid']):
                    create_backup()
                    
                    result_df, operation_info = transformer.merge_columns(
                        df=df,
                        columns=merge_columns,
                        separators=separators,
                        new_column_name=new_column_name,
                        handle_missing=handle_missing,
                        is_datetime_merge=is_datetime_merge,
                        datetime_format=datetime_format
                    )
                    
                    if operation_info['success']:
                        st.session_state.dataset = result_df
                        st.success(f"Successfully merged columns into '{new_column_name}'")
                        st.markdown(f"**Rows affected:** {operation_info['rows_affected']}")
                        st.markdown(f"**Null values:** {operation_info['null_count']}")
                        st.rerun()
                    else:
                        st.error(f"Merge failed: {operation_info['error']}")
            
            with col2:
                if st.button("Preview Result"):
                    result_df, operation_info = transformer.merge_columns(
                        df=df.head(10),
                        columns=merge_columns,
                        separators=separators,
                        new_column_name=new_column_name,
                        handle_missing=handle_missing,
                        is_datetime_merge=is_datetime_merge,
                        datetime_format=datetime_format
                    )
                    if operation_info['success']:
                        st.dataframe(result_df[[*merge_columns, new_column_name]], use_container_width=True)
                    else:
                        st.error(f"Preview failed: {operation_info['error']}")
        
        elif len(merge_columns) == 1:
            st.info("Select at least 2 columns to merge.")
        else:
            st.info("Select columns from the dropdown above to begin merging.")
    
    else:
        st.subheader("Split Column")
        st.markdown("Split a single column into multiple columns based on a separator.")
        
        available_columns = df.columns.tolist()
        split_column = st.selectbox(
            "Select column to split",
            options=available_columns,
            help="Choose the column you want to split"
        )
        
        if split_column:
            is_datetime_split = st.checkbox(
                "DateTime split",
                help="Split a datetime column into components (year, month, day, etc.)"
            )
            
            if is_datetime_split:
                st.markdown("#### DateTime Components to Extract")
                datetime_components = st.multiselect(
                    "Select components",
                    options=['year', 'month', 'day', 'hour', 'minute', 'second', 
                             'weekday', 'week', 'quarter', 'dayofyear', 'date', 'time'],
                    default=['year', 'month', 'day'],
                    help="Choose which datetime components to extract"
                )
                separator = ""
            else:
                separator = st.text_input(
                    "Separator",
                    value="-",
                    help="Character(s) to split on (leave empty for whitespace)"
                )
                datetime_components = None
            
            new_column_prefix = st.text_input(
                "New column prefix",
                value=f"{split_column}",
                help="Prefix for the new split columns"
            )
            
            max_splits = st.number_input(
                "Maximum splits",
                min_value=-1,
                max_value=100,
                value=-1,
                help="-1 for unlimited splits, or specify a maximum number"
            )
            
            if not is_datetime_split:
                validation = transformer.validate_split_column(df, split_column, separator)
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(warning)
                
                if validation['errors']:
                    for error in validation['errors']:
                        st.error(error)
                
                if validation['estimated_columns'] > 0:
                    st.info(f"Estimated number of new columns: {validation['estimated_columns']}")
            
            st.markdown("#### Sample Data")
            sample_data = df[split_column].dropna().head(5)
            st.write(sample_data.tolist())
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Apply Split", type="primary"):
                    create_backup()
                    
                    result_df, operation_info = transformer.split_column(
                        df=df,
                        column=split_column,
                        separator=separator,
                        new_column_prefix=new_column_prefix,
                        max_splits=max_splits,
                        is_datetime_split=is_datetime_split,
                        datetime_components=datetime_components
                    )
                    
                    if operation_info['success']:
                        st.session_state.dataset = result_df
                        st.success(f"Successfully split column into: {operation_info['new_columns']}")
                        st.rerun()
                    else:
                        st.error(f"Split failed: {operation_info['error']}")
            
            with col2:
                if st.button("Preview Split"):
                    result_df, operation_info = transformer.split_column(
                        df=df.head(10),
                        column=split_column,
                        separator=separator,
                        new_column_prefix=new_column_prefix,
                        max_splits=max_splits,
                        is_datetime_split=is_datetime_split,
                        datetime_components=datetime_components
                    )
                    if operation_info['success']:
                        preview_cols = [split_column] + operation_info['new_columns']
                        st.dataframe(result_df[preview_cols], use_container_width=True)
                    else:
                        st.error(f"Preview failed: {operation_info['error']}")

with transformation_tabs[1]:
    st.header("Expand JSON Data")
    st.markdown("""
    Detect and expand columns containing nested JSON or dictionary-like data.
    Extract specific keys into new columns or explode arrays into multiple rows.
    """)
    
    expand_tabs = st.tabs(["Expand JSON to Columns", "Convert Columns to JSON"])
    
    with expand_tabs[0]:
        st.subheader("Expand JSON/Dictionary Columns")
        
        if st.button("Detect JSON Columns", type="secondary"):
            with st.spinner("Scanning for JSON/dictionary columns..."):
                json_columns = transformer.detect_json_columns(df)
                st.session_state.detected_json_columns = json_columns
        
        if 'detected_json_columns' in st.session_state and st.session_state.detected_json_columns:
            json_columns = st.session_state.detected_json_columns
            
            st.success(f"Found {len(json_columns)} column(s) with JSON/dictionary data")
            
            for i, col_info in enumerate(json_columns):
                with st.expander(f"Column: {col_info['column']} ({col_info['json_percentage']:.1f}% JSON)"):
                    st.markdown(f"""
                    - **Type:** {'Nested Array' if col_info['is_nested'] else 'Array' if col_info['is_array'] else 'Object'}
                    - **Available Keys:** {', '.join(col_info['keys'][:20])}{'...' if len(col_info['keys']) > 20 else ''}
                    """)
                    
                    st.markdown("##### Sample Data")
                    sample = df[col_info['column']].dropna().head(3)
                    for idx, val in enumerate(sample):
                        st.code(str(val)[:500] + ('...' if len(str(val)) > 500 else ''), language='json')
            
            st.divider()
            
            st.subheader("Extract Data")
            
            json_col_names = [c['column'] for c in json_columns]
            selected_json_col = st.selectbox(
                "Select JSON column to expand",
                options=json_col_names
            )
            
            if selected_json_col:
                col_info = next(c for c in json_columns if c['column'] == selected_json_col)
                
                keys_to_extract = st.multiselect(
                    "Select keys to extract",
                    options=col_info['keys'],
                    default=col_info['keys'][:3] if len(col_info['keys']) >= 3 else col_info['keys'],
                    help="Choose which keys from the JSON to create as new columns"
                )
                
                explode_arrays = st.checkbox(
                    "Explode arrays into separate rows",
                    value=False,
                    help="If checked, each array element becomes a separate row (increases row count)"
                )
                
                if explode_arrays:
                    st.warning("Exploding arrays will create multiple rows per original record. This may significantly increase your dataset size.")
                
                column_prefix = st.text_input(
                    "Column prefix",
                    value=selected_json_col,
                    help="Prefix for the new extracted columns"
                )
                
                if keys_to_extract:
                    st.markdown("##### Preview New Columns")
                    st.write([f"{column_prefix}_{key}" for key in keys_to_extract])
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Apply Expansion", type="primary"):
                            create_backup()
                            
                            result_df, operation_info = transformer.expand_json_column(
                                df=df,
                                column=selected_json_col,
                                keys_to_extract=keys_to_extract,
                                explode_arrays=explode_arrays,
                                prefix=column_prefix
                            )
                            
                            if operation_info['success']:
                                st.session_state.dataset = result_df
                                st.success(f"Successfully expanded JSON column")
                                st.markdown(f"**New columns:** {operation_info['new_columns']}")
                                if explode_arrays:
                                    st.markdown(f"**New row count:** {operation_info['rows_after']}")
                                st.rerun()
                            else:
                                st.error(f"Expansion failed: {operation_info['error']}")
                    
                    with col2:
                        if st.button("Preview Expansion"):
                            result_df, operation_info = transformer.expand_json_column(
                                df=df.head(10),
                                column=selected_json_col,
                                keys_to_extract=keys_to_extract,
                                explode_arrays=explode_arrays,
                                prefix=column_prefix
                            )
                            if operation_info['success']:
                                preview_cols = [selected_json_col] + operation_info['new_columns']
                                st.dataframe(result_df[preview_cols], use_container_width=True)
                            else:
                                st.error(f"Preview failed: {operation_info['error']}")
        
        elif 'detected_json_columns' in st.session_state:
            st.info("No JSON/dictionary columns detected in the dataset.")
        else:
            st.info("Click 'Detect JSON Columns' to scan your dataset for nested data.")
    
    with expand_tabs[1]:
        st.subheader("Convert Columns to JSON/Dictionary")
        st.markdown("Combine multiple columns back into a JSON/dictionary format.")
        
        available_columns = df.columns.tolist()
        columns_to_combine = st.multiselect(
            "Select columns to combine into JSON",
            options=available_columns,
            help="Select columns that will become keys in the JSON object"
        )
        
        if columns_to_combine:
            new_json_column = st.text_input(
                "New JSON column name",
                value="combined_json",
                help="Name for the new JSON column"
            )
            
            as_array = st.checkbox(
                "Wrap as array",
                value=False,
                help="Wrap each JSON object in an array"
            )
            
            group_by_col = st.selectbox(
                "Group by column (optional)",
                options=['None'] + [c for c in available_columns if c not in columns_to_combine],
                help="If selected, rows with the same value in this column will be grouped into arrays"
            )
            group_by = None if group_by_col == 'None' else group_by_col
            
            st.markdown("##### Preview")
            preview_data = df[columns_to_combine].head(3).to_dict('records')
            for record in preview_data:
                st.json(record)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Convert to JSON", type="primary"):
                    create_backup()
                    
                    result_df, operation_info = transformer.columns_to_json(
                        df=df,
                        columns=columns_to_combine,
                        new_column_name=new_json_column,
                        group_by=group_by,
                        as_array=as_array
                    )
                    
                    if operation_info['success']:
                        st.session_state.dataset = result_df
                        st.success(f"Successfully created JSON column '{new_json_column}'")
                        st.rerun()
                    else:
                        st.error(f"Conversion failed: {operation_info['error']}")
            
            with col2:
                if st.button("Preview JSON"):
                    result_df, operation_info = transformer.columns_to_json(
                        df=df.head(5),
                        columns=columns_to_combine,
                        new_column_name=new_json_column,
                        group_by=group_by,
                        as_array=as_array
                    )
                    if operation_info['success']:
                        st.dataframe(result_df[[*columns_to_combine, new_json_column]], use_container_width=True)
                    else:
                        st.error(f"Preview failed: {operation_info['error']}")

with transformation_tabs[2]:
    st.header("Data Type Conversion")
    st.markdown("""
    Convert columns to different data types with validation and error handling.
    
    **Supported Types:** Integer, Float, String, Boolean, DateTime, Date, Time, 
    Categorical, List, Dictionary
    """)
    
    type_info_expander = st.expander("Data Type Reference", expanded=False)
    with type_info_expander:
        st.markdown("""
        | Type | Description | Example |
        |------|-------------|---------|
        | **Integer** | Whole numbers (supports nullable) | 1, 42, -7 |
        | **Float** | Decimal numbers | 3.14, -0.5, 100.0 |
        | **String** | Text data | "hello", "2024-01-15" |
        | **Boolean** | True/False values | True, False, 1, 0, "yes", "no" |
        | **DateTime** | Date and time combined | 2024-01-15 14:30:00 |
        | **Date** | Date only | 2024-01-15 |
        | **Time** | Time only | 14:30:00 |
        | **Categorical** | Fixed set of categories | "A", "B", "C" |
        | **List** | Array of values | [1, 2, 3], ["a", "b"] |
        | **Dictionary** | Key-value pairs | {"name": "John", "age": 30} |
        """)
    
    available_columns = df.columns.tolist()
    convert_column = st.selectbox(
        "Select column to convert",
        options=available_columns,
        key="convert_col_select"
    )
    
    if convert_column:
        current_dtype = transformer.detect_column_dtype(df[convert_column])
        st.info(f"Current detected type: **{current_dtype}**")
        
        sample_values = df[convert_column].dropna().head(5).tolist()
        st.markdown("**Sample values:**")
        st.write(sample_values)
        
        target_dtype = st.selectbox(
            "Convert to type",
            options=['integer', 'float', 'string', 'boolean', 'datetime', 
                     'date', 'time', 'categorical', 'list', 'dictionary'],
            help="Select the target data type"
        )
        
        datetime_format = None
        if target_dtype in ['datetime', 'date', 'time']:
            use_custom_format = st.checkbox("Use custom datetime format")
            if use_custom_format:
                datetime_format = st.text_input(
                    "DateTime format",
                    value="%Y-%m-%d %H:%M:%S" if target_dtype == 'datetime' else "%Y-%m-%d",
                    help="strftime format string (e.g., %Y-%m-%d for dates)"
                )
        
        error_handling = st.selectbox(
            "Error handling",
            options=['coerce', 'raise'],
            format_func=lambda x: {
                'coerce': 'Convert errors to null (recommended)',
                'raise': 'Stop on first error'
            }[x]
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Apply Conversion", type="primary"):
                create_backup()
                
                result_df, operation_info = transformer.convert_column_dtype(
                    df=df,
                    column=convert_column,
                    target_dtype=target_dtype,
                    datetime_format=datetime_format,
                    errors=error_handling
                )
                
                if operation_info['success']:
                    st.session_state.dataset = result_df
                    st.success(f"Successfully converted column to {target_dtype}")
                    st.markdown(f"**Successful conversions:** {operation_info['successful_conversions']}")
                    if operation_info['failed_conversions'] > 0:
                        st.warning(f"**Failed conversions:** {operation_info['failed_conversions']} (set to null)")
                    st.rerun()
                else:
                    st.error(f"Conversion failed: {operation_info['error']}")
        
        with col2:
            if st.button("Preview Conversion"):
                result_df, operation_info = transformer.convert_column_dtype(
                    df=df.head(10),
                    column=convert_column,
                    target_dtype=target_dtype,
                    datetime_format=datetime_format,
                    errors=error_handling
                )
                if operation_info['success']:
                    comparison_df = pd.DataFrame({
                        'Original': df[convert_column].head(10),
                        'Converted': result_df[convert_column]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.error(f"Preview failed: {operation_info['error']}")

st.divider()
st.subheader("Current Dataset Status")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Rows", f"{len(df):,}")
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
with col4:
    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

if st.checkbox("Show current dataset preview"):
    st.dataframe(df.head(20), use_container_width=True)

if st.checkbox("Show column data types"):
    dtype_info = []
    for col in df.columns:
        detected = transformer.detect_column_dtype(df[col])
        dtype_info.append({
            'Column': col,
            'Pandas dtype': str(df[col].dtype),
            'Detected type': detected,
            'Non-null count': df[col].notna().sum(),
            'Null count': df[col].isna().sum()
        })
    st.dataframe(pd.DataFrame(dtype_info), use_container_width=True)
