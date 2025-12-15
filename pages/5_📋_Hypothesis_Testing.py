import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state
from modules.hypothesis_analysis import HypothesisAnalyzer
from modules.ai_hypothesis_helper import AIHypothesisHelper
from modules.test_registry import TEST_REGISTRY
import modules.test_registration
from modules.test_registration import get_analyzer
import plotly.graph_objects as go



st.title("🔬 Hypothesis Testing & Statistical Analysis")

st.markdown("""
Use AI to find the perfect statistical test for your research question, or browse from 24+ comprehensive tests across parametric and non-parametric categories.
""")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset
analyzer = get_analyzer()
ai_helper = AIHypothesisHelper()

if 'hypothesis_results' not in st.session_state:
    st.session_state.hypothesis_results = []
if 'ai_suggestions' not in st.session_state:
    st.session_state.ai_suggestions = None

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
all_cols = df.columns.tolist()

st.divider()
st.subheader("🤖 Step 1: Describe Your Research Question")

tab1, tab2 = st.tabs(["🎯 AI-Powered Suggestion", "📋 Browse Tests Manually"])

with tab1:
    st.markdown("**Tell the AI what you want to test in plain language:**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_prompt = st.text_area(
            "Your research question:",
            placeholder="Example: I want to test if sales differ between product categories, or if there's a relationship between age and income",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Quick Examples:**")
        if st.button("📊 Compare groups", use_container_width=True):
            st.session_state.example_prompt = "Compare means between two or more groups"
        if st.button("🔗 Find relationship", use_container_width=True):
            st.session_state.example_prompt = "Find correlation between two variables"
        if st.button("📈 Check normality", use_container_width=True):
            st.session_state.example_prompt = "Test if my data is normally distributed"
        if st.button("🎲 Test distribution", use_container_width=True):
            st.session_state.example_prompt = "Test if categorical data follows expected distribution"
    
    if 'example_prompt' in st.session_state:
        user_prompt = st.session_state.example_prompt
        del st.session_state.example_prompt
    
    if st.button("🔍 Get AI Recommendation", type="primary", use_container_width=True, disabled=not user_prompt):
        if user_prompt:
            with st.spinner("🤖 AI is analyzing your question..."):
                data_context = {
                    'numeric_columns': numeric_cols,
                    'categorical_columns': categorical_cols,
                    'sample_size': len(df)
                }
                
                test_metadata = TEST_REGISTRY.get_ai_metadata()
                
                result = ai_helper.suggest_test(user_prompt, data_context, test_metadata)
                
                if result['success']:
                    st.session_state.ai_suggestions = result['suggestions']
                    st.success("✅ AI recommendations ready!")
                else:
                    st.error(f"❌ {result.get('error', 'AI suggestion failed')}")
                    if result.get('fallback'):
                        st.info("💡 Browse tests manually below or check your GROQ_API_KEY configuration.")
    
    if st.session_state.ai_suggestions:
        st.divider()
        st.markdown("### 🎯 AI Recommendations")
        
        suggestions = st.session_state.ai_suggestions
        
        if 'primary_recommendation' in suggestions:
            primary = suggestions['primary_recommendation']
            
            st.success(f"""
**Primary Recommendation: {primary.get('test_name', 'N/A')}**

📊 **Category:** {primary.get('category', 'N/A').replace('_', ' ').title()}

💡 **Why:** {primary.get('rationale', 'No rationale provided')}

🎯 **Confidence:** {int(primary.get('confidence', 0) * 100)}%
            """)
            
            st.session_state.selected_test_id = primary.get('test_id')
        
        if suggestions.get('alternative_recommendations'):
            with st.expander("🔄 Alternative Options"):
                for alt in suggestions['alternative_recommendations']:
                    st.info(f"**{alt.get('test_name')}** ({alt.get('category', '').replace('_', ' ').title()})\n\n{alt.get('rationale')}")
        
        if suggestions.get('warnings'):
            with st.expander("⚠️ Important Notes"):
                for warning in suggestions['warnings']:
                    st.warning(warning)
        
        if suggestions.get('suggested_columns'):
            with st.expander("📋 Suggested Columns"):
                sug_cols = suggestions['suggested_columns']
                if sug_cols.get('numeric'):
                    st.write("**Numeric:**", ', '.join(sug_cols['numeric']))
                if sug_cols.get('categorical'):
                    st.write("**Categorical:**", ', '.join(sug_cols['categorical']))

with tab2:
    st.markdown("**Browse all available statistical tests:**")
    
    test_category = st.selectbox(
        "Filter by category:",
        options=['All Tests', 'Parametric', 'Non-Parametric'],
        key='manual_category'
    )
    
    if test_category == 'Parametric':
        available_tests = TEST_REGISTRY.get_by_category('parametric')
    elif test_category == 'Non-Parametric':
        available_tests = TEST_REGISTRY.get_by_category('non_parametric')
    else:
        available_tests = TEST_REGISTRY.list_all_tests()
    
    test_by_subcategory = {}
    for test in available_tests:
        subcat = test.subcategory.replace('_', ' ').title()
        if subcat not in test_by_subcategory:
            test_by_subcategory[subcat] = []
        test_by_subcategory[subcat].append(test)
    
    for subcat, tests in test_by_subcategory.items():
        with st.expander(f"📁 {subcat} ({len(tests)} tests)"):
            for test in tests:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{test.name}**")
                    st.caption(test.description)
                with col2:
                    if st.button("Select", key=f"select_{test.test_id}", use_container_width=True):
                        st.session_state.selected_test_id = test.test_id
                        st.session_state.ai_suggestions = None

st.divider()

if 'selected_test_id' in st.session_state and st.session_state.selected_test_id:
    selected_test = TEST_REGISTRY.get_test(st.session_state.selected_test_id)
    
    if selected_test:
        st.subheader(f"⚙️ Step 2: Configure Test - {selected_test.name}")
        
        st.info(f"""
**Description:** {selected_test.description}

**Category:** {selected_test.category.replace('_', ' ').title()}

**Assumptions:** {', '.join(selected_test.assumptions)}
        """)
        
        col_sig1, col_sig2 = st.columns(2)
        with col_sig1:
            alpha = st.slider(
                "Significance level (α):",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Typically 0.05 for 95% confidence"
            )
            analyzer.set_alpha(alpha)
        
        with col_sig2:
            use_sample = st.checkbox(
                "Use sample of data",
                value=False,
                help="Test on a subset of the data instead of the full dataset"
            )
        
        # Sample size configuration
        if use_sample:
            sample_col1, sample_col2 = st.columns(2)
            with sample_col1:
                auto_sample = st.checkbox(
                    "Auto-calculate sample size",
                    value=True,
                    help="Automatically determine optimal sample size based on dataset"
                )
            
            with sample_col2:
                if auto_sample:
                    # Use 10% or max 1000 rows as automatic sample size
                    auto_size = min(1000, max(100, int(len(df) * 0.1)))
                    st.info(f"Auto sample size: {auto_size} rows")
                    sample_size = auto_size
                else:
                    sample_size = st.number_input(
                        "Sample size:",
                        min_value=50,
                        max_value=len(df),
                        value=min(500, len(df)),
                        step=50,
                        help="Number of rows to use for the test"
                    )
            
            # Create sampled dataframe
            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                st.info(f"✓ Using {sample_size} rows for hypothesis test")
        
        # Refresh column lists after any potential type conversions
        all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Filter columns based on test requirements and data quality
        def filter_applicable_columns(cols, col_type='numeric'):
            """Filter columns that are applicable for the selected test"""
            applicable = []
            for col in cols:
                # Basic validation
                non_null_count = df[col].notna().sum()
                
                # Skip columns with too few valid values
                if non_null_count < 5:
                    continue
                
                if col_type == 'numeric':
                    # For numeric columns, check if there's enough variation
                    if df[col].nunique() >= 2:  # Need at least 2 unique values
                        applicable.append(col)
                elif col_type == 'categorical':
                    # For categorical columns, check uniqueness
                    unique_count = df[col].nunique()
                    # Need at least 2 categories but not too many (max 50% unique or 20 categories)
                    if 2 <= unique_count <= min(20, len(df) * 0.5):
                        applicable.append(col)
            
            return applicable
        
        # Apply filtering based on test requirements
        reqs = selected_test.input_requirements
        
        if reqs.get('numeric_cols', 0) > 0:
            numeric_cols = filter_applicable_columns(all_numeric_cols, 'numeric')
            if not numeric_cols:
                st.warning("⚠️ No applicable numeric columns found for this test. Columns need at least 5 non-null values and 2 unique values.")
                numeric_cols = ['No applicable numeric columns']
        else:
            numeric_cols = all_numeric_cols
        
        if reqs.get('categorical_cols', 0) > 0:
            categorical_cols = filter_applicable_columns(all_categorical_cols, 'categorical')
            if not categorical_cols:
                st.warning("⚠️ No applicable categorical columns found for this test. Columns need at least 5 non-null values and 2-20 unique categories.")
                categorical_cols = ['No applicable categorical columns']
        else:
            categorical_cols = all_categorical_cols
        
        test_params = {}
        
        if reqs.get('manual_input'):
            st.markdown("**Manual Input Required:**")
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                successes1 = st.number_input("Successes in group 1:", min_value=0, value=0, step=1)
                total1 = st.number_input("Total in group 1:", min_value=1, value=100, step=1)
            with param_col2:
                successes2 = st.number_input("Successes in group 2:", min_value=0, value=0, step=1)
                total2 = st.number_input("Total in group 2:", min_value=1, value=100, step=1)
            test_params = {'successes': [successes1, successes2], 'totals': [total1, total2]}
        
        else:
            if reqs.get('numeric_cols') == 1:
                if reqs.get('categorical_cols') == 1:
                    param_col1, param_col2 = st.columns(2)
                    with param_col1:
                        numeric_col = st.selectbox("Numeric variable:", options=numeric_cols, help="Only showing columns with sufficient data for this test")
                    with param_col2:
                        group_col = st.selectbox("Grouping variable:", options=categorical_cols, help="Only showing columns with 2-20 unique categories")
                    test_params = {'numeric_col': numeric_col, 'group_col': group_col}
                
                elif reqs.get('test_value'):
                    param_col1, param_col2 = st.columns(2)
                    with param_col1:
                        column = st.selectbox("Numeric variable:", options=numeric_cols, help="Only showing columns with sufficient data for this test")
                    with param_col2:
                        test_value = st.number_input("Test value (H0: μ =):", value=0.0, format="%.4f")
                    test_params = {'column': column, 'test_value': test_value}
                
                else:
                    column = st.selectbox("Select column:", options=numeric_cols, help="Only showing columns with sufficient data for this test")
                    test_params = {'column': column}
            
            elif reqs.get('numeric_cols') == 2:
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    col1 = st.selectbox("First variable:", options=numeric_cols, help="Only showing columns with sufficient data for this test")
                with param_col2:
                    col2_opts = [c for c in numeric_cols if c != col1] if (numeric_cols and 'No applicable' not in numeric_cols[0]) else ['No applicable numeric columns']
                    col2 = st.selectbox("Second variable:", options=col2_opts, help="Only showing columns with sufficient data for this test")
                test_params = {'col1': col1, 'col2': col2}
            
            elif reqs.get('categorical_cols') == 2:
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    col1 = st.selectbox("First categorical variable:", options=categorical_cols, help="Only showing columns with 2-20 unique categories")
                with param_col2:
                    col2_opts = [c for c in categorical_cols if c != col1] if (categorical_cols and 'No applicable' not in categorical_cols[0]) else ['No applicable categorical columns']
                    col2 = st.selectbox("Second categorical variable:", options=col2_opts, help="Only showing columns with 2-20 unique categories")
                test_params = {'col1': col1, 'col2': col2}
            
            elif reqs.get('categorical_cols') == 1:
                column = st.selectbox("Categorical variable:", options=categorical_cols, help="Only showing columns with 2-20 unique categories")
                if reqs.get('test_proportion'):
                    param_col1, param_col2 = st.columns(2)
                    with param_col1:
                        success_value = st.text_input("Success value:", value="")
                    with param_col2:
                        test_proportion = st.number_input("Test proportion:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                    test_params = {'column': column, 'success_value': success_value, 'test_proportion': test_proportion}
                else:
                    test_params = {'column': column}
        
        if st.button(f"🔬 Run {selected_test.name}", type="primary", use_container_width=True):
            with st.spinner(f"Running {selected_test.name}..."):
                validation = TEST_REGISTRY.validate_inputs(st.session_state.selected_test_id, df, **test_params)
                
                if not validation.get('valid', False):
                    st.error(f"❌ Validation Error: {validation.get('error', 'Invalid inputs')}")
                else:
                    result = TEST_REGISTRY.execute_test(st.session_state.selected_test_id, df, **test_params)
                    
                    if 'error' in result:
                        st.error(f"❌ Test Error: {result['error']}")
                    else:
                        result['test_id'] = st.session_state.selected_test_id
                        result['test_category'] = selected_test.category
                        result['columns_used'] = test_params
                        st.session_state.hypothesis_results.append(result)
                        st.success(f"✅ {selected_test.name} completed successfully!")
                        st.rerun()

st.divider()

if st.session_state.hypothesis_results:
    st.subheader("📊 Step 3: Test Results")
    
    for idx, result in enumerate(reversed(st.session_state.hypothesis_results)):
        result_idx = len(st.session_state.hypothesis_results) - idx - 1
        
        category_icon = "📐" if result.get('test_category') == 'parametric' else "📊"
        
        with st.expander(f"{category_icon} {result['test_name']}", expanded=(idx == 0)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test Statistic", f"{result['statistic']:.4f}" if isinstance(result['statistic'], (int, float)) else str(result['statistic']))
            
            with col2:
                p_val = result['p_value']
                st.metric("p-value", f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val))
            
            with col3:
                decision_color = "🟢" if "Reject" in result.get('decision', '') or "Significant" in result.get('decision', '') or "differ" in result.get('decision', '') else "🔴"
                st.markdown(f"""
                <div style='text-align: left;'>
                    <p style='color: gray; font-size: 0.8rem; margin: 0;'>Decision</p>
                    <p style='font-size: 0.95rem; margin: 5px 0; line-height: 1.3;'>{decision_color} {result['decision']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if result.get('effect_size'):
                st.markdown(f"**Effect Size:** {result['effect_size']['type']} = {result['effect_size']['value']:.4f}" if isinstance(result['effect_size']['value'], (int, float)) else f"**Effect Size:** {result['effect_size']['type']} = {result['effect_size']['value']}")
            
            if result.get('confidence_interval') and result['confidence_interval'].get('interval') != 'N/A':
                ci = result['confidence_interval']['interval']
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    st.markdown(f"**{result['confidence_interval']['level']} Confidence Interval:** ({ci[0]:.4f}, {ci[1]:.4f})")
            
            if result.get('df'):
                if isinstance(result['df'], dict):
                    st.markdown(f"**Degrees of Freedom:** {', '.join([f'{k}: {v}' for k, v in result['df'].items()])}")
                else:
                    st.markdown(f"**Degrees of Freedom:** {result['df']}")
            
            if result.get('interpretation'):
                st.info(f"**Interpretation:** {result['interpretation']}")
            
            if result.get('sample_sizes'):
                st.markdown(f"**Sample Sizes:** {', '.join([f'{k}: {v}' for k, v in result['sample_sizes'].items()])}")
            
            if result.get('group_stats'):
                with st.expander("📈 Group Statistics"):
                    stats_df = pd.DataFrame(result['group_stats']).T
                    st.dataframe(stats_df, use_container_width=True)
            
            if result.get('assumption_checks'):
                with st.expander("🔍 Assumption Checks"):
                    for assumption, check in result['assumption_checks'].items():
                        if isinstance(check, dict):
                            st.write(f"**{assumption.replace('_', ' ').title()}:** {check}")
                        elif isinstance(check, bool):
                            status = "✅ Passed" if check else "❌ Failed"
                            st.write(f"**{assumption.replace('_', ' ').title()}:** {status}")
                        else:
                            st.write(f"**{assumption.replace('_', ' ').title()}:** {check}")
            
            if result.get('warnings'):
                st.warning("⚠️ **Warnings:**\n" + "\n".join([f"- {w}" for w in result['warnings']]))
            
            if result.get('notes'):
                st.caption(f"ℹ️ {result['notes']}")
            
            action_cols = st.columns([3, 1])
            with action_cols[1]:
                if st.button("🗑️ Remove", key=f"remove_{result_idx}", use_container_width=True):
                    st.session_state.hypothesis_results.pop(result_idx)
                    st.rerun()
    
    st.divider()
    action_cols = st.columns([2, 1, 1])
    
    with action_cols[0]:
        st.markdown(f"**Total tests performed:** {len(st.session_state.hypothesis_results)}")
    
    with action_cols[1]:
        if st.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state.hypothesis_results = []
            st.success("All results cleared")
            st.rerun()
    
    with action_cols[2]:
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
            st.session_state.hypothesis_test_results = st.session_state.hypothesis_results
            st.switch_page("pages/7_Reports.py")

else:
    st.info("💡 No hypothesis tests have been run yet. Use the AI suggestion or browse tests above to get started.")

st.divider()

with st.expander("📚 Quick Guide to Choosing Statistical Tests"):
    st.markdown("""
    ### How to Choose the Right Test
    
    **Use the AI suggestion above for automatic recommendations, or follow these guidelines:**
    
    #### Comparing Groups
    - **2 groups, numeric outcome:** Independent t-test (parametric) or Mann-Whitney U (non-parametric)
    - **3+ groups, numeric outcome:** One-way ANOVA (parametric) or Kruskal-Wallis (non-parametric)
    - **Paired/matched data:** Paired t-test (parametric) or Wilcoxon/Sign test (non-parametric)
    
    #### Relationships
    - **Two numeric variables:** Pearson correlation (parametric) or Spearman/Kendall (non-parametric)
    - **Two categorical variables:** Chi-square test or Fisher's exact test
    
    #### Testing Assumptions
    - **Normality:** Shapiro-Wilk, Anderson-Darling, or Kolmogorov-Smirnov tests
    - **Equal variances:** Levene's or Bartlett's test
    - **Distribution fit:** Chi-square goodness-of-fit or KS test
    
    ### Interpreting Results
    - **p-value < α (typically 0.05):** Result is statistically significant
    - **Effect size:** Indicates the magnitude/importance of the finding
    - **Always check assumptions** before trusting the results
    """)

st.divider()
st.markdown("### 📋 Next Steps")

nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("📊 Visualize Data", use_container_width=True):
        st.switch_page("pages/4_Visualization.py")
with nav_cols[1]:
    if st.button("🤖 Ask AI Assistant", use_container_width=True):
        st.switch_page("pages/6_AI_Assistant.py")
with nav_cols[2]:
    if st.button("📄 Generate Report", use_container_width=True, type="primary"):
        st.switch_page("pages/7_Reports.py")
