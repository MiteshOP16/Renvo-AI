import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.utils import initialize_session_state
from modules.data_balancer import DataBalancer
import io

st.title("Data Balancer")

st.markdown("""
Balance your dataset for machine learning by handling class imbalance in your target variable.
Choose from multiple sampling techniques to create a balanced dataset suitable for model training.
""")

initialize_session_state()

if st.session_state.dataset is None:
    st.warning("No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset

balancer = DataBalancer()

if 'balanced_data' not in st.session_state:
    st.session_state.balanced_data = None
if 'balancing_result' not in st.session_state:
    st.session_state.balancing_result = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'use_split_data' not in st.session_state:
    st.session_state.use_split_data = False
if 'split_performed' not in st.session_state:
    st.session_state.split_performed = False

st.divider()

st.subheader("Step 1: Select Columns")

st.markdown("""
**Select feature columns** (must be numeric) and a **target column** (the class you want to balance).
""")

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
all_cols = df.columns.tolist()

col1, col2 = st.columns(2)

with col1:
    feature_cols = st.multiselect(
        "Select feature columns (numeric only):",
        options=numeric_cols,
        help="These columns will be used as input features. Only numeric columns can be selected.",
        key='feature_selection'
    )

with col2:
    target_col = st.selectbox(
        "Select target column (class to balance):",
        options=[''] + all_cols,
        help="This column contains the class labels you want to balance. Can be numeric or categorical.",
        key='target_selection'
    )

if feature_cols and target_col and target_col != '':
    validation = balancer.validate_data(df, feature_cols, target_col)
    
    if not validation['valid']:
        st.divider()
        st.error("**Data Validation Failed - Please clean your data first:**")
        for error in validation['errors']:
            st.error(f"- {error}")
        if validation['warnings']:
            st.warning("**Warnings:**")
            for warning in validation['warnings']:
                st.warning(f"- {warning}")
        st.info("Please use the Cleaning Wizard to handle missing values and encode categorical variables before balancing.")
        st.stop()
    else:
        if validation['warnings']:
            st.warning("**Warnings:**")
            for warning in validation['warnings']:
                st.warning(f"- {warning}")

if target_col and target_col != '':
    st.divider()
    st.subheader("Current Class Distribution")
    
    dist = df[target_col].value_counts().sort_index()
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.markdown("**Class Counts:**")
        dist_df = pd.DataFrame({
            'Class': dist.index.astype(str),
            'Count': dist.values,
            'Percentage': (dist.values / len(df) * 100).round(2)
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    with col_dist2:
        fig = go.Figure(data=[
            go.Bar(
                x=dist.index.astype(str),
                y=dist.values,
                text=dist.values,
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        fig.update_layout(
            title="Class Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    imbalance_ratio = dist.max() / dist.min() if dist.min() > 0 else float('inf')
    if imbalance_ratio > 1.5:
        st.warning(f"Dataset is imbalanced. Ratio: {imbalance_ratio:.2f}:1 (majority:minority)")
    else:
        st.success(f"Dataset is relatively balanced. Ratio: {imbalance_ratio:.2f}:1")

if feature_cols and target_col and target_col != '':
    st.divider()
    st.subheader("Step 2: Choose Data Usage")
    
    st.markdown("""
    Choose whether to use the whole dataset or split it into training and test sets.
    **Recommended:** Use split data to keep a held-out test set for model evaluation.
    """)
    
    data_option = st.radio(
        "Select data usage option:",
        options=["Use Whole Data", "Use Split Data"],
        key="data_usage_option",
        horizontal=True
    )
    
    st.session_state.use_split_data = (data_option == "Use Split Data")
    
    if st.session_state.use_split_data:
        st.markdown("---")
        st.markdown("**Configure Train/Test Split:**")
        
        col_split1, col_split2 = st.columns([2, 1])
        
        with col_split1:
            test_percentage = st.slider(
                "Test set percentage:",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data to hold out for testing. The rest will be used for training and balancing."
            )
        
        with col_split2:
            train_percentage = 100 - test_percentage
            st.metric("Train Set", f"{train_percentage}%")
            st.metric("Test Set", f"{test_percentage}%")
        
        if st.button("Perform Stratified Split", type="primary", use_container_width=True):
            with st.spinner("Performing stratified split..."):
                split_result = balancer.stratified_split(
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    test_size=test_percentage / 100,
                    random_state=42
                )
                
                if split_result['success']:
                    st.session_state.train_data = split_result['train_data']
                    st.session_state.test_data = split_result['test_data']
                    st.session_state.split_performed = True
                    st.success(f"Split completed! Train: {split_result['train_size']} rows, Test: {split_result['test_size']} rows")
                    st.rerun()
                else:
                    st.error(f"Split failed: {split_result['error']}")
        
        if st.session_state.split_performed and st.session_state.train_data is not None:
            st.markdown("---")
            st.markdown("**Split Results:**")
            
            col_train, col_test = st.columns(2)
            
            with col_train:
                st.markdown("**Training Set Distribution:**")
                train_dist = st.session_state.train_data[target_col].value_counts().sort_index()
                train_dist_df = pd.DataFrame({
                    'Class': train_dist.index.astype(str),
                    'Count': train_dist.values,
                    'Percentage': (train_dist.values / len(st.session_state.train_data) * 100).round(2)
                })
                st.dataframe(train_dist_df, use_container_width=True, hide_index=True)
            
            with col_test:
                st.markdown("**Test Set Distribution:**")
                test_dist = st.session_state.test_data[target_col].value_counts().sort_index()
                test_dist_df = pd.DataFrame({
                    'Class': test_dist.index.astype(str),
                    'Count': test_dist.values,
                    'Percentage': (test_dist.values / len(st.session_state.test_data) * 100).round(2)
                })
                st.dataframe(test_dist_df, use_container_width=True, hide_index=True)
            
            st.info(f"Training data ({len(st.session_state.train_data)} rows) will be used for balancing. Test data ({len(st.session_state.test_data)} rows) will remain unchanged.")

st.divider()
st.subheader("Step 3: Choose Balancing Method")

methods_dict = balancer.get_available_methods()

tab_os, tab_us, tab_hybrid = st.tabs([
    "Oversampling", 
    "Undersampling", 
    "Hybrid"
])

with tab_os:
    st.markdown("""
    **Oversampling** increases the number of minority class samples.
    
    - **Random Oversampling**: Randomly duplicates minority class samples
    - **SMOTE**: Creates synthetic samples using k-nearest neighbors
    """)
    
    for method in methods_dict['Oversampling']:
        if st.button(f"Select {method}", key=f"btn_os_{method}", use_container_width=True):
            st.session_state.selected_method = method

with tab_us:
    st.markdown("""
    **Undersampling** reduces the number of majority class samples.
    
    - **Random Undersampling**: Randomly removes majority class samples
    - **Tomek Links**: Removes borderline majority samples
    - **NearMiss-1/2/3**: Selects majority samples based on distance to minority samples
    - **ENN**: Removes samples whose class differs from majority of neighbors
    - **CNN**: Finds consistent subset of majority class
    - **OSS**: Combines Tomek Links and CNN
    - **Cluster Centroids**: Replaces majority samples with cluster centroids
    - **NCR**: Cleans data by removing noisy samples
    """)
    
    for method in methods_dict['Undersampling']:
        if st.button(f"Select {method}", key=f"btn_us_{method}", use_container_width=True):
            st.session_state.selected_method = method

with tab_hybrid:
    st.markdown("""
    **Hybrid Methods** combine oversampling and undersampling.
    
    - **SMOTE + Tomek Links**: First applies SMOTE, then removes Tomek links
    - **SMOTE + ENN**: First applies SMOTE, then cleans with ENN
    """)
    
    for method in methods_dict['Hybrid']:
        if st.button(f"Select {method}", key=f"btn_hybrid_{method}", use_container_width=True):
            st.session_state.selected_method = method

if 'selected_method' in st.session_state:
    st.success(f"Selected method: **{st.session_state.selected_method}**")

st.divider()
st.subheader("Step 4: Apply Balancing")

can_apply = feature_cols and target_col and target_col != '' and 'selected_method' in st.session_state
if st.session_state.use_split_data:
    can_apply = can_apply and st.session_state.split_performed and st.session_state.train_data is not None

col_apply1, col_apply2 = st.columns([2, 1])

with col_apply1:
    apply_disabled = not can_apply
    if st.session_state.use_split_data and not st.session_state.split_performed:
        st.warning("Please perform the stratified split first before applying balancing.")
    
    if st.button("Apply Balancing", type="primary", use_container_width=True, disabled=apply_disabled):
        if st.session_state.use_split_data:
            data_to_balance = st.session_state.train_data
        else:
            data_to_balance = df
        
        with st.spinner(f"Applying {st.session_state.selected_method}..."):
            result = balancer.balance_data(
                df=data_to_balance,
                feature_cols=feature_cols,
                target_col=target_col,
                method=st.session_state.selected_method,
                random_state=42
            )
            
            if result['success']:
                st.session_state.balanced_data = result['balanced_data']
                st.session_state.balancing_result = result
                st.success(f"Balancing completed successfully using {result['method']}!")
                st.rerun()
            else:
                st.error(f"Balancing failed: {result['error']}")

with col_apply2:
    if st.button("Reset All", use_container_width=True):
        st.session_state.balanced_data = None
        st.session_state.balancing_result = None
        st.session_state.train_data = None
        st.session_state.test_data = None
        st.session_state.split_performed = False
        if 'selected_method' in st.session_state:
            del st.session_state.selected_method
        st.success("Reset successful")
        st.rerun()

if st.session_state.balancing_result:
    st.divider()
    st.subheader("Step 5: Results")
    
    result = st.session_state.balancing_result
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric("Original Size", f"{result['original_size']:,} rows")
    
    with col_metrics2:
        st.metric("Balanced Size", f"{result['balanced_size']:,} rows")
    
    with col_metrics3:
        size_change = ((result['balanced_size'] - result['original_size']) / result['original_size'] * 100)
        st.metric("Size Change", f"{size_change:+.1f}%")
    
    st.divider()
    st.markdown("### Before vs After Class Distribution")
    
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.markdown("**Before Balancing:**")
        before_df = pd.DataFrame({
            'Class': result['original_distribution'].index.astype(str),
            'Count': result['original_distribution'].values,
            'Percentage': (result['original_distribution'].values / result['original_size'] * 100).round(2)
        })
        st.dataframe(before_df, use_container_width=True, hide_index=True)
        
        fig_before = go.Figure(data=[
            go.Bar(
                x=result['original_distribution'].index.astype(str),
                y=result['original_distribution'].values,
                text=result['original_distribution'].values,
                textposition='auto',
                marker_color='lightcoral',
                name='Before'
            )
        ])
        fig_before.update_layout(
            title="Original Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_before, use_container_width=True)
    
    with col_after:
        st.markdown("**After Balancing:**")
        after_df = pd.DataFrame({
            'Class': result['balanced_distribution'].index.astype(str),
            'Count': result['balanced_distribution'].values,
            'Percentage': (result['balanced_distribution'].values / result['balanced_size'] * 100).round(2)
        })
        st.dataframe(after_df, use_container_width=True, hide_index=True)
        
        fig_after = go.Figure(data=[
            go.Bar(
                x=result['balanced_distribution'].index.astype(str),
                y=result['balanced_distribution'].values,
                text=result['balanced_distribution'].values,
                textposition='auto',
                marker_color='lightgreen',
                name='After'
            )
        ])
        fig_after.update_layout(
            title="Balanced Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_after, use_container_width=True)
    
    st.divider()
    st.subheader("Step 6: Download Data")
    
    st.warning("""
    **Important Warning:**
    
    Balanced data will have a different number of rows than your cleaned dataset. 
    Use balanced columns directly for model training.
    """)
    
    st.markdown("### Download Balanced Training Data")
    
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        csv_buffer = io.StringIO()
        st.session_state.balanced_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Balanced Train (CSV)",
            data=csv_data,
            file_name=f"balanced_train_{result['method'].replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        excel_buffer = io.BytesIO()
        st.session_state.balanced_data.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="Download Balanced Train (Excel)",
            data=excel_data,
            file_name=f"balanced_train_{result['method'].replace(' ', '_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_download3:
        if st.button("Preview Balanced Data", use_container_width=True):
            st.session_state.show_balanced_preview = not st.session_state.get('show_balanced_preview', False)
    
    if st.session_state.get('show_balanced_preview', False):
        st.divider()
        st.markdown("### Balanced Data Preview")
        st.dataframe(st.session_state.balanced_data.head(100), use_container_width=True)
        st.caption(f"Showing first 100 rows of {len(st.session_state.balanced_data)} total rows")

if st.session_state.use_split_data and st.session_state.test_data is not None:
    st.divider()
    st.subheader("Test Data (Unchanged)")
    
    st.info(f"""
    The test set contains {len(st.session_state.test_data)} rows and has been kept unchanged.
    Use this data to evaluate your trained model.
    """)
    
    st.markdown("### Download Test Data")
    
    col_test1, col_test2, col_test3 = st.columns(3)
    
    with col_test1:
        csv_buffer_test = io.StringIO()
        st.session_state.test_data.to_csv(csv_buffer_test, index=False)
        csv_test_data = csv_buffer_test.getvalue()
        
        st.download_button(
            label="Download Test Data (CSV)",
            data=csv_test_data,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_test2:
        excel_buffer_test = io.BytesIO()
        st.session_state.test_data.to_excel(excel_buffer_test, index=False, engine='openpyxl')
        excel_test_data = excel_buffer_test.getvalue()
        
        st.download_button(
            label="Download Test Data (Excel)",
            data=excel_test_data,
            file_name="test_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_test3:
        if st.button("Preview Test Data", use_container_width=True):
            st.session_state.show_test_preview = not st.session_state.get('show_test_preview', False)
    
    if st.session_state.get('show_test_preview', False):
        st.divider()
        st.markdown("### Test Data Preview")
        st.dataframe(st.session_state.test_data.head(100), use_container_width=True)
        st.caption(f"Showing first 100 rows of {len(st.session_state.test_data)} total rows")

st.divider()

with st.expander("Guide to Data Balancing"):
    st.markdown("""
    ### Why Balance Data?
    
    Machine learning algorithms often perform poorly on imbalanced datasets where one class 
    significantly outnumbers others. Balancing helps create better models by:
    
    - Preventing bias toward majority class
    - Improving model performance on minority classes
    - Creating more robust predictions
    
    ### Which Method to Choose?
    
    **For Small Datasets:**
    - Use **SMOTE** (creates synthetic samples)
    - Avoid heavy undersampling (loses too much data)
    
    **For Large Datasets:**
    - **Random Undersampling** works well
    - **Cluster Centroids** for efficient reduction
    
    **For Best Results:**
    - Try **Hybrid methods** (SMOTE + Tomek Links or SMOTE + ENN)
    - These combine benefits of both approaches
    
    **For Borderline Cases:**
    - **Tomek Links** removes ambiguous samples
    - **ENN** cleans noisy data
    
    ### Important Considerations
    
    1. **Data Loss**: Undersampling removes data, which may lose important information
    2. **Overfitting**: Oversampling may cause overfitting if not used carefully
    3. **Validation**: Always use cross-validation to evaluate balanced models
    4. **Original Data**: Keep your original cleaned data separate from balanced data
    5. **Model Training**: Use balanced data ONLY for training, not for final analysis
    
    ### Edge Cases Handled
    
    - **Missing Values**: System blocks balancing if any missing values exist in selected columns
    - **Non-Numeric Features**: Only numeric columns can be used as features
    - **Too Few Classes**: Target must have at least 2 classes
    - **Too Many Classes**: Warning shown if more than 10 classes (may not balance well)
    - **Insufficient Data**: At least 10 rows required for balancing
    """)

st.divider()
st.markdown("### Next Steps")

nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("Get AI Assistance", use_container_width=True):
        st.switch_page("pages/7_AI_Assistant.py")

with nav_cols[1]:
    if st.button("Visualize Data", use_container_width=True):
        st.switch_page("pages/4_Visualization.py")

with nav_cols[2]:
    if st.button("Generate Report", use_container_width=True, type="primary"):
        st.switch_page("pages/8_Reports.py")
