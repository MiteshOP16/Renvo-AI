# Intelligent Data Cleaning Assistant

## Overview
A Streamlit-based web application for statistical agencies, designed to clean and analyze survey data. It provides AI-powered guidance, comprehensive analysis tools, and detailed reporting capabilities for data quality assessment and cleaning operations. The application aims to help data analysts and statisticians upload, analyze, detect, handle, and clean survey datasets, apply various cleaning methods with survey weight support, get AI-powered recommendations, and generate comprehensive reports.

## User Preferences
None configured yet.

## System Architecture
The application is built using **Streamlit** (Python web framework) and leverages **Pandas** and **NumPy** for data processing. For analysis, it uses **Scikit-learn** and **SciPy**, while **Plotly**, **Seaborn**, and **Matplotlib** handle visualization. AI assistance is integrated via the **Groq API** (llama-3.1-8b-instant model). Reporting is managed with **Jinja2** templates and **ReportLab** for PDF generation.

The core structure includes:
- **Streamlit Pages**: Home page with integrated data upload, then dedicated pages for Anomaly Detection & Duplicate Removal, Data Transformation, Column Analysis, Cleaning Wizard, Visualization, Hypothesis Analysis, Data Balancer, AI Assistant, and Reports.
- **Modular Design**: Functionality is organized into `modules/` for AI integration, anomaly detection, cleaning engine, data analysis, hypothesis testing, report generation, survey weights, and utility functions.
- **UI/UX Decisions**: Features include interactive visualization builders with multi-column selection and 9 chart types, enhanced distribution analysis with statistical explanations and visual interpretations, and multi-method statistical outlier detection.
- **Reporting**: Professional PDF reports with modern styling, color palettes, improved table styling, enhanced typography, and color-coded messages are generated using ReportLab, alongside Markdown, HTML, and JSON export options. Reports include executive summaries, anomaly detection results, column analysis summaries, embedded high-resolution visualizations, and a cleaning operations audit trail.
- **Key Features**:
    - **Data Upload & Configuration**: Integrated into home page - supports CSV and Excel (.xlsx, .xls) with automatic column type detection and openpyxl for Excel support.
    - **Column Analysis**: Individual column analysis including missing data patterns, outlier detection, and quality assessment.
    - **Cleaning Wizard**: Multiple cleaning methods (imputation, outlier handling, text operations) with impact assessments.
    - **AI Assistant**: Context-aware guidance for cleaning recommendations.
    - **Undo/Redo**: Full operation history.
    - **Data Type Anomaly Detection**: Dedicated page for type mismatch detection, clear display of anomalous values, and flexible correction options.
    - **Duplicate Removal**: Complete row duplicate detection and removal with configurable options (check all columns or specific columns, keep first/last/none).
    - **Hypothesis Testing**: Comprehensive statistical testing with 15 test types, intelligent recommendations based on data characteristics, and detailed output with interpretations.
    - **Data Balancer**: Machine learning dataset balancing with 14 methods across 3 categories (Oversampling: Random Oversampling, SMOTE; Undersampling: Random Undersampling, Tomek Links, NearMiss-1/2/3, ENN, CNN, OSS, Cluster Centroids, NCR; Hybrid: SMOTE + Tomek Links, SMOTE + ENN), before/after class distribution visualization, CSV/Excel download, and validation to ensure data quality (no missing values, numeric features, valid target classes).
    - **Performance Optimizations**: Implemented deterministic caching, vectorized operations, optimized imputation and outlier detection, and memory optimizations for large datasets.

## External Dependencies
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy, openpyxl (for Excel support)
- **Analysis**: Scikit-learn, SciPy, imbalanced-learn
- **Visualization**: Plotly, Seaborn, Matplotlib
- **AI Assistant**: Groq API (llama-3.1-8b-instant model)
- **Reporting**: Jinja2, ReportLab

## Recent Updates (Dec 2025)
- **Navigation Restructure & Feature Cleanup (Dec 15, 2025)**:
    - Reorganized navigation into logical sections: Data Cleaning (Anomaly Detection, Data Transformation, Column Analysis, Cleaning Wizard), Data Analysis (Hypothesis Testing, Data Balancer), Data Visualization (Visualization, Reports), AI (AI Assistant)
    - Fixed schema configuration error on home page - column type selection now properly handles legacy/custom types without overwriting user selections
    - Removed Data Type Conversion from Data Transformation page (now only Merge/Split and JSON expansion)
    - Removed Survey Weights Configuration, Type Standardisation, and Remove Duplicates from Cleaning Wizard
- **Data Transformation Feature (Dec 14, 2025)**: Added new Data Transformation page with:
    - **Merge/Split Columns**: Toggle between merge and split modes with DateTime-aware operations
    - **Expand JSON Data**: Automatic JSON detection, key extraction, array explosion, and reverse operations
    - **Preview Functionality**: All operations include preview before applying
    - **Undo Support**: Integrated with backup/undo system

## Recent Updates (Nov 2025)
- **Data Balancer Fix and Enhancement (Nov 30, 2025)**: Fixed critical '_validate_data' error affecting all balancing methods by implementing proper NumPy array conversion pipeline with `_prepare_features` and `_prepare_target` methods. Added safe fit_resample wrapper that handles label encoding for categorical targets and properly reverses encoding after balancing. New split data flow added: users can now choose between "Use Whole Data" or "Use Split Data" with configurable stratified train/test split percentage. Test data is preserved unchanged and available for download at the bottom of the page. Proper validation blocks usage until data is cleaned.
- **Data Balancer Feature (Nov 23, 2025)**: Added comprehensive data balancing section after Hypothesis Analysis with 14 balancing methods, feature/target column selection, before/after class distribution visualization, CSV/Excel download with warnings, and data quality validation (checks for missing values, numeric features, valid target classes). Works with both cleaned datasets from Cleaning Wizard or pre-cleaned uploaded datasets. Advanced methods (GAN, VAE, Cost-Sensitive Learning) shown with "not yet implemented" status.
- **Merged Data Upload with Home Page**: Data upload and configuration functionality integrated into the main home page (app.py) for streamlined user experience
- **Added Excel Support**: Installed openpyxl for full Excel file (.xlsx, .xls) upload compatibility
- **Enhanced Duplicate Removal**: New tab in Anomaly Detection page with clear UI showing duplicate rows are removed based on complete row matching (all columns or selected subset)
- **Improved Navigation**: Page references updated to reflect new structure (Home instead of Data Upload)
- **Replit Environment Setup (Nov 19, 2025)**: Successfully configured for Replit with Python 3.11.13, all dependencies installed, Streamlit configured for port 5000 with webview output, and autoscale deployment ready
- **UX Improvements (Nov 19, 2025)**:
    - Verified outlier detection works correctly for both integer and continuous numeric types using `pd.api.types.is_numeric_dtype()`
    - Streamlined reports page by removing redundant anomaly detection results section
    - Enhanced cleaning wizard sidebar to sort columns by quality score (ascending) with percentage display for better prioritization
    - Added intelligent column filtering in hypothesis testing to show only applicable columns based on test requirements (numeric: ≥5 valid values, ≥2 unique; categorical: ≥5 valid, 2-20 unique categories) with helpful tooltips and warnings