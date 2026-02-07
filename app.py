import streamlit as st
import pandas as pd
import numpy as np
import traceback


def main():
    from modules.utils import initialize_session_state, detect_column_types
    from modules.data_analyzer import ColumnAnalyzer
    from modules.design_system import apply_global_styles

    # Apply global styling
    apply_global_styles()
    initialize_session_state()

    st.title("Renvo AI - Intelligent Data Cleaning Assistant")

    st.markdown("""
    Welcome to the Survey Data Cleaning Assistant - an AI-powered tool designed specifically for statistical agencies.

    ### Key Features:
    - **Individual Column Analysis**
    - **AI-Powered Assistance**
    - **Multiple Cleaning Strategies**
    - **Comprehensive Audit Trail**
    - **Statistical Rigor**
    """)

    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    **Data Cleaning**
    - Anomaly Detection  
    - Data Transformation  
    - Column Analysis  
    - Cleaning Wizard  

    **Data Analysis**
    - Hypothesis Testing  
    - Data Balancer  

    **Visualization**
    - Charts  
    - Reports  

    **AI**
    - AI Assistant
    """)

    st.divider()

    # ===================== DATA IMPORT =====================
    st.header("📊 Data Import")

    import_tab1, import_tab2, import_tab3 = st.tabs(
        ["📁 File Upload", "🔌 MySQL Database", "🟢 Supabase"]
    )

    # ---------- FILE UPLOAD ----------
    with import_tab1:
        st.markdown("Upload a CSV or Excel file to get started:")

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, Excel",
            key="file_upload_tab"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(
                    f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns"
                )

                if (
                    st.session_state.dataset is None
                    or not df.equals(st.session_state.dataset)
                ):
                    st.session_state.dataset = df.copy()
                    st.session_state.original_dataset = df.copy()
                    st.session_state.column_types = detect_column_types(df)

                    st.session_state.column_analysis = {}
                    st.session_state.cleaning_history = {}
                    st.session_state.undo_stack = []
                    st.session_state.redo_stack = []

                    st.info("🔍 Column types auto-detected. Review below.")

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.stop()

    # ---------- MYSQL ----------
    with import_tab2:
        from modules.db_connector import render_database_connector_ui
        render_database_connector_ui()

    # ---------- SUPABASE ----------
    with import_tab3:
        from modules.db_connector import render_supabase_connector_ui
        render_supabase_connector_ui()

    # ===================== DATASET VIEW =====================
    if st.session_state.dataset is not None:
        df = st.session_state.dataset

        st.divider()
        st.subheader("📋 Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c4.metric(
            "Memory",
            f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )

        st.subheader("🔍 Data Preview")

        preview_rows = st.slider(
            "Rows to preview",
            5,
            min(100, len(df)),
            10,
            key="preview_slider"
        )

        show_info = st.checkbox("Show column info", key="show_info")

        if show_info:
            st.dataframe(pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes,
                "Non-null": df.count(),
                "Missing": df.isnull().sum(),
                "Unique": df.nunique()
            }), use_container_width=True)

        st.dataframe(df.head(preview_rows), use_container_width=True)

        # ===================== COLUMN TYPES =====================
        st.divider()
        st.subheader("⚙️ Column Type Configuration")

        type_options = [
            "continuous", "integer", "ordinal",
            "categorical", "binary", "text",
            "datetime", "empty", "unknown"
        ]

        updated_types = {}

        for col in df.columns:
            cols = st.columns([3, 2, 3])
            cols[0].write(col)
            cols[1].write(st.session_state.column_types.get(col, "unknown"))

            selected = cols[2].selectbox(
                "Type",
                type_options,
                index=type_options.index(
                    st.session_state.column_types.get(col, "unknown")
                ),
                key=f"type_{col}",
                label_visibility="collapsed"
            )

            updated_types[col] = selected

        c1, c2 = st.columns(2)

        with c1:
            if st.button("💾 Update Column Types", use_container_width=True):
                st.session_state.column_types = updated_types
                st.success("Updated successfully")
                st.rerun()

        with c2:
            if st.button("🔍 Start Column Analysis", use_container_width=True):
                analyzer = ColumnAnalyzer()
                bar = st.progress(0)

                for i, col in enumerate(df.columns):
                    st.session_state.column_analysis[col] = analyzer.analyze_column(df, col)
                    bar.progress((i + 1) / len(df.columns))

                st.success("🎉 Column analysis completed!")

        # ===================== CONFIG IMPORT =====================
        st.divider()
        st.subheader("💾 Configuration Management")

        config_file = st.file_uploader(
            "📥 Import Configuration",
            type=["json"],
            key="config_import_uploader"
        )

        if config_file is not None:
            from modules.utils import import_configuration
            content = config_file.read().decode("utf-8")

            if import_configuration(content):
                st.success("✅ Configuration imported")
                st.rerun()
            else:
                st.error("❌ Import failed")

    else:
        st.info("👆 Upload a dataset to begin")


if __name__ == "__main__":
    main()
