# 🧹 Intelligent Data Cleaning Assistant
## Comprehensive Project Report

---

## Table of Contents
1. [Project Details](#1-project-details)
2. [Problem Statement](#2-problem-statement)
3. [Need of Project](#3-need-of-project)
4. [Proposed Solution](#4-proposed-solution)
5. [Technology Used](#5-technology-used)
6. [Project Outcomes](#6-project-outcomes)
7. [System Architecture & Modelling](#7-system-architecture--modelling)
8. [Feature Modules](#8-feature-modules)
   - 8.1 [Data Upload & Pre-processing](#81-data-upload--pre-processing)
   - 8.2 [Anomaly Detection](#82-anomaly-detection)
   - 8.3 [Data Transformation](#83-data-transformation)
   - 8.4 [Column Analysis](#84-column-analysis)
   - 8.5 [Cleaning Wizard](#85-cleaning-wizard)
   - 8.6 [Hypothesis Testing](#86-hypothesis-testing)
   - 8.7 [Data Balancing](#87-data-balancing)
   - 8.8 [Data Visualization](#88-data-visualization)
   - 8.9 [Report Generation](#89-report-generation)
   - 8.10 [AI Assistant](#810-ai-assistant)
9. [Future Scope for Project Enhancement](#9-future-scope-for-project-enhancement)
10. [Conclusion](#10-conclusion)

---

## 1. Project Details

An **AI-powered, web-based data processing platform** designed to assist users in cleaning, analyzing, and reporting survey and business data. The system supports guided data preparation by identifying issues such as **missing values, outliers, and imbalance**. It enables users to generate visual insights, analytical reports, and interact with data using **natural language**.

The platform is specifically designed for:
- **Statistical agencies** processing census and survey data
- **Data analysts** and **data scientists** performing data quality control
- **Researchers** requiring transparent and reproducible data cleaning workflows
- **Business analysts** preparing datasets for analytical projects

---

## 2. Problem Statement

To develop an **intelligent platform** that supports efficient data cleaning, transformation, analysis, and reporting, enabling users to derive meaningful insights from raw datasets **without requiring extensive technical expertise**.

### Key Challenges Addressed:
- Complex data cleaning operations requiring programming knowledge
- Lack of transparency in automated data cleaning processes
- Difficulty in maintaining proper audit trails
- Inconsistent treatment of survey weights and sampling methodologies
- Time-consuming manual data quality assessment

---

## 3. Need of Project

Data preparation is a **crucial yet highly time-consuming phase** in data analysis, often consuming **nearly 80% of the total analytical effort**. Raw datasets typically contain:
- Missing values
- Outliers
- Inconsistencies
- Imbalanced classes

These issues require extensive manual intervention before analysis can be performed. Traditional tools such as programming languages and standalone visualization software demand **technical proficiency** and involve **fragmented workflows**, making the process inefficient and error-prone.

### Industry Pain Points:
1. **Manual Data Cleaning Risks**: Human errors and inconsistencies leading to biased or unreliable decisions
2. **Fragmented Tools**: Lack of platforms combining data cleaning, statistical analysis, visualization, reporting, and explainability
3. **Transparency Gap**: Missing audit trails and documentation of cleaning operations
4. **Accessibility Barrier**: High technical barriers for non-technical users

**Therefore, there is a strong need for an AI-driven solution** that automates end-to-end data processing, reduces manual effort, improves accuracy, and makes data analysis accessible to both technical and non-technical users.

---

## 4. Proposed Solution

The proposed platform allows users to:
- **Upload CSV or Excel datasets** and perform guided data cleaning and analysis
- Receive **intelligent suggestions** for handling missing values, outliers, data balancing, and hypothesis testing
- Generate **interactive visualizations** and **professional PDF reports** with clear explanations
- Interact with an **integrated AI chatbot** to query the dataset and reports using **natural language**

### Core Capabilities:
| Feature | Description |
|---------|-------------|
| Smart Type Detection | Automatically identifies numeric, categorical, binary, ordinal, and text columns |
| Context-Aware Recommendations | AI-powered suggestions tailored to specific data issues |
| Comprehensive Audit Trails | Complete documentation of all cleaning operations |
| Survey Weight Integration | Proper handling of sampling weights in all operations |
| Interactive Visualizations | Distribution plots, outlier detection, correlation analysis |
| Professional Reports | PDF/HTML reports with statistical summaries and explanations |

---

## 5. Technology Used

| Category | Technologies |
|----------|-------------|
| **Artificial Intelligence and Machine Learning** | Groq API (Llama 3.1), scikit-learn (Isolation Forest, KNN) |
| **Statistical Analysis Techniques** | SciPy, Statsmodels, NumPy |
| **Web-Based Application Framework** | Streamlit (Multi-page Application) |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization Tools** | Plotly, Matplotlib, Seaborn |
| **Natural Language Processing** | Groq LLM Integration |
| **PDF Report Generation** | ReportLab |
| **Data Balancing** | imbalanced-learn (SMOTE, ADASYN, Tomek Links) |

---

## 6. Project Outcomes

The system assists users in **converting raw datasets into clean and analysis-ready data**. It enables users to:

✅ **Understand data patterns** through comprehensive visualizations  
✅ **Generate professional reports** documenting all data transformations  
✅ **Engage in conversational interaction** with an AI assistant  
✅ **Make faster and more confident decisions** based on clean, validated data  
✅ **Perform statistical hypothesis testing** to validate research questions  
✅ **Balance imbalanced datasets** using advanced resampling techniques  

---

## 7. System Architecture & Modelling

### The project consists of **3 main processing steps**:

### Step 1: Dataset Upload and Pre-processing
The system accepts raw datasets in **CSV or Excel format** through a web-based interface. Once uploaded:
- Initial validation and pre-processing are performed
- Data types, missing values, outliers, inconsistencies, and class imbalance are identified
- Basic transformations such as formatting and normalization are applied based on user selection and system recommendations

### Step 2: Guided Data Cleaning and Analysis
The pre-processed dataset is passed to the **AI-assisted processing layer**, where intelligent techniques are applied for:
- **Missing value handling** (Mean, Median, Mode, KNN, Interpolation)
- **Outlier treatment** (IQR, Z-score, Winsorization, Capping)
- **Data balancing** (SMOTE, Random Oversampling, Undersampling)
- **Statistical weighting** (Survey weight integration)
- **Hypothesis testing** (T-tests, ANOVA, Chi-square, Correlation tests)

These operations are executed through a **guided workflow**, allowing user interaction and control at each stage.

### Step 3: Insight Generation, Visualization, and Reporting
After processing, the refined dataset is used to generate:
- **Interactive visualizations** (distributions, correlations, outlier plots)
- **Professional PDF reports** containing before-and-after comparisons, statistical summaries, and AI-generated explanations
- **Cleaned dataset download** for further analysis
- **Natural language AI interaction** for querying results

### Technical Flow Diagram:
```
┌─────────────────────────┐
│  Data Uploading and     │
│  Schema Configuration   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Basic Cleaning:        │
│  1. Remove Anomalies    │
│     and Duplicates      │
│  2. Data Transformation │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Column-wise Analysis   │
│  & Cleaning             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Data Visualisation,    │────▶│  Cleaned Dataset        │
│  Hypothesis Testing,    │     │  Download & Reports     │
│  Data Balancing         │     │  regarding errors       │
└───────────┬─────────────┘     └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  AI ASSISTANT &         │
│  RECOMMENDATIONS        │
└─────────────────────────┘
```

---

## 8. Feature Modules

### 8.1 Data Upload & Pre-processing

**File:** `app.py` (Main Application Entry)

The main application module handles:
- **File Upload**: Support for CSV (.csv) and Excel (.xlsx, .xls) formats
- **Automatic Type Detection**: Smart identification of column data types
- **Data Preview**: Interactive data exploration with filtering and sorting
- **Schema Configuration**: User can override detected types if needed
- **Initial Quality Assessment**: Summary statistics and data quality overview

**Key Features:**
- Upload datasets up to configurable size limits
- Automatic detection of delimiter and encoding
- Support for handling multiple sheets in Excel files
- Initial data profiling with statistics summary

---

### 8.2 Anomaly Detection

**Page:** `1_🔍_Anomaly_Detection.py`  
**Module:** `modules/anomaly_detector.py`

This module identifies unusual patterns and anomalies in the dataset using multiple detection methods.

**Detection Methods:**
| Method | Description |
|--------|-------------|
| **IQR (Interquartile Range)** | Detects outliers based on Q1-Q3 range with customizable multiplier |
| **Z-Score** | Identifies values more than n standard deviations from mean |
| **Modified Z-Score** | Robust version using median absolute deviation (MAD) |
| **Isolation Forest** | Machine learning-based anomaly detection for complex patterns |

**Features:**
- **Consensus-based detection**: Uses multiple methods to identify reliable outliers
- **Severity grading**: Classifies anomalies as low, medium, or high severity
- **Visual highlighting**: Interactive plots showing detected anomalies
- **Treatment recommendations**: AI-powered suggestions for handling each anomaly

---

### 8.3 Data Transformation

**Page:** `2_🔄_Data_Transformation.py`  
**Module:** `modules/data_transformer.py`

Provides comprehensive data transformation capabilities for preparing data for analysis.

**Transformation Types:**

#### Numeric Transformations:
| Transformation | Use Case |
|----------------|----------|
| Log Transform | Reduces right skewness |
| Square Root | Moderate skewness reduction |
| Box-Cox | Optimal power transformation |
| Yeo-Johnson | Works with negative values |
| Standardization (Z-score) | Centers data to mean=0, std=1 |
| Min-Max Scaling | Scales to [0,1] range |
| Robust Scaling | Uses median and IQR, robust to outliers |

#### Categorical Transformations:
- One-Hot Encoding
- Label Encoding
- Ordinal Encoding
- Target Encoding

**Features:**
- Preview transformations before applying
- Undo/Redo functionality
- Distribution comparisons (before vs after)
- Automatic recommendation based on data characteristics

---

### 8.4 Column Analysis

**Page:** `3_📊_Column_Analysis.py`  
**Module:** `modules/data_analyzer.py`

Performs in-depth analysis of each column individually to understand data characteristics.

**Analysis Metrics:**

For Numeric Columns:
- Count, Mean, Median, Mode
- Standard Deviation, Variance
- Skewness, Kurtosis
- Min, Max, Range
- Percentiles (25th, 50th, 75th)
- Missing value count and percentage
- Outlier detection results

For Categorical Columns:
- Unique value count
- Value frequency distribution
- Mode and frequency
- Missing value analysis
- Cardinality assessment

**Quality Scoring:**
- Automated quality score (0-100) based on multiple factors
- Issue detection and prioritization
- Pattern recognition in missing data

---

### 8.5 Cleaning Wizard

**Page:** `4_🧹_Cleaning_Wizard.py`  
**Module:** `modules/cleaning_engine.py`

Interactive step-by-step cleaning interface with AI-powered recommendations.

**Missing Value Handling Methods:**

| Category | Methods |
|----------|---------|
| **Imputation** | Mean, Median, Mode, Constant Value |
| **Advanced Imputation** | KNN Imputation, Iterative Imputation |
| **Time-based** | Forward Fill, Backward Fill, Interpolation |
| **Deletion** | Listwise Deletion, Pairwise Deletion |

**Outlier Treatment Options:**
- **Removal**: Delete outlier records
- **Winsorization**: Replace with percentile values
- **Capping**: Replace with boundary values
- **Transformation**: Apply log/sqrt to reduce impact
- **Imputation**: Replace with mean/median

**Features:**
- Real-time preview of changes
- Undo/Redo with complete history
- AI recommendations for each column
- Impact assessment before applying changes

---

### 8.6 Hypothesis Testing

**Page:** `5_📋_Hypothesis_Testing.py`  
**Module:** `modules/hypothesis_analysis.py`  
**AI Helper:** `modules/ai_hypothesis_helper.py`

A comprehensive statistical hypothesis testing module that enables users to validate research questions and test assumptions about their data.

#### Purpose:
Hypothesis testing allows users to make data-driven decisions by statistically testing claims or assumptions about population parameters based on sample data.

#### Available Statistical Tests:

**Parametric Tests (Assumes Normal Distribution):**

| Test | Purpose | When to Use |
|------|---------|-------------|
| **One-Sample T-Test** | Compare sample mean to known value | Single numeric variable vs theoretical value |
| **Independent T-Test** | Compare means of two groups | Two independent groups, equal variances assumed |
| **Welch's T-Test** | Compare means of two groups | Two groups, unequal variances |
| **Paired T-Test** | Compare matched samples | Before/after measurements, matched pairs |
| **One-Way ANOVA** | Compare means of 3+ groups | Multiple group comparison |
| **Pearson Correlation** | Linear relationship between variables | Two continuous variables |
| **Simple Linear Regression** | Predict one variable from another | Continuous predictor and outcome |
| **Logistic Regression** | Binary outcome prediction | Binary dependent variable |

**Non-Parametric Tests (No Distribution Assumptions):**

| Test | Purpose | When to Use |
|------|---------|-------------|
| **Mann-Whitney U** | Compare two groups | Non-normal data, two independent groups |
| **Wilcoxon Signed-Rank** | Compare matched samples | Non-normal paired data |
| **Sign Test** | Compare paired samples | Paired data, nominal scale |
| **Kruskal-Wallis H** | Compare 3+ groups | Non-normal data, multiple groups |
| **Spearman Correlation** | Monotonic relationship | Ordinal or non-normal data |
| **Kendall's Tau** | Rank correlation | Small sample sizes, many tied ranks |
| **Chi-Square Test** | Association between categoricals | Two categorical variables |
| **Fisher's Exact Test** | Association in 2x2 tables | Small sample sizes |

**Normality & Goodness-of-Fit Tests:**

| Test | Purpose |
|------|---------|
| **Shapiro-Wilk Test** | Test for normality (small samples) |
| **Kolmogorov-Smirnov Test** | Test against any distribution |
| **Anderson-Darling Test** | Sensitive tail normality test |
| **Chi-Square Goodness of Fit** | Compare observed vs expected frequencies |
| **Levene's Test** | Test equality of variances |

**Post-Hoc Tests:**
- **Tukey's HSD**: Pairwise comparisons after ANOVA

#### AI-Powered Test Recommendation:
Users can describe their research question in **natural language** and receive:
- **Primary Test Recommendation** with rationale
- **Alternative Tests** for consideration
- **Suggested Columns** for the analysis
- **Assumption Warnings** if data violates test requirements

#### How to Use:
1. **Step 1**: Describe your research question or browse available tests
2. **Step 2**: Configure test parameters (columns, significance level)
3. **Step 3**: Run the test and interpret results
4. **Step 4**: View detailed statistical output with visualizations

#### Output Includes:
- Test statistic and p-value
- Effect size measures
- Confidence intervals
- Interpretation in plain language
- Assumption violation warnings
- Visual representation of results

---

### 8.7 Data Balancing

**Page:** `6_⚖️_Data_Balancer.py`  
**Module:** `modules/data_balancer.py`

A comprehensive module for handling **imbalanced datasets**, which is crucial for machine learning model training and statistical analysis.

#### What is Data Imbalance?
Data imbalance occurs when the distribution of classes in a target variable is highly skewed. For example:
- Fraud detection: 99% legitimate, 1% fraudulent transactions
- Medical diagnosis: 95% healthy, 5% diseased patients
- Customer churn: 90% retained, 10% churned customers

Imbalanced data can lead to biased models that favor the majority class.

#### Available Balancing Methods:

**Oversampling Techniques (Increase Minority Class):**

| Method | Description | Best For |
|--------|-------------|----------|
| **Random Oversampling** | Randomly duplicates minority class samples | Simple baseline, small datasets |
| **SMOTE** | Synthetic Minority Over-sampling Technique - Creates synthetic samples by interpolating between existing minority samples | Numeric features, avoiding overfitting |

**Undersampling Techniques (Reduce Majority Class):**

| Method | Description | Best For |
|--------|-------------|----------|
| **Random Undersampling** | Randomly removes majority class samples | Large datasets where data loss is acceptable |
| **Tomek Links** | Removes majority samples that form Tomek links with minority samples | Cleaning borderline samples |
| **NearMiss-1** | Selects majority samples closest to minority samples | Preserving decision boundary |
| **NearMiss-2** | Selects majority samples based on average distance | More aggressive selection |
| **NearMiss-3** | Selects majority samples for each minority sample | Local neighborhood preservation |
| **Edited Nearest Neighbours (ENN)** | Removes samples misclassified by KNN | Cleaning noisy data |
| **Condensed Nearest Neighbour (CNN)** | Creates consistent subset of data | Reducing dataset size |
| **One-Sided Selection (OSS)** | Combines Tomek Links and CNN | Combined cleaning and reduction |
| **Cluster Centroids** | Replaces majority class with cluster centroids | Summarizing majority class |
| **Neighbourhood Cleaning Rule (NCR)** | Removes noisy majority samples | Focused noise removal |

**Hybrid Techniques (Combine Over and Undersampling):**

| Method | Description | Best For |
|--------|-------------|----------|
| **SMOTE + Tomek Links** | SMOTE followed by Tomek Links cleanup | Balanced approach with cleaning |
| **SMOTE + ENN** | SMOTE followed by ENN cleanup | More aggressive noise removal |

#### Workflow:

1. **Select Feature Columns**: Choose numeric columns as features
2. **Select Target Column**: Choose the class column to balance
3. **View Class Distribution**: Analyze current imbalance ratio
4. **Choose Data Usage**:
   - **Use Whole Data**: Apply balancing to entire dataset
   - **Use Split Data**: First split into train/test, then balance only training data
5. **Select Balancing Method**: Choose appropriate technique based on data characteristics
6. **Apply Balancing**: Execute the selected method
7. **Download Results**: Export balanced dataset

#### Visualization:
- **Before/After Distribution Charts**: Compare class distributions
- **Imbalance Ratio Calculation**: Displays majority:minority ratio
- **Sample Count Metrics**: Shows exact numbers for each class

#### Best Practices:
- Use **stratified split** before balancing to maintain class proportions in test set
- **Never balance test data** - it should represent real-world distribution
- Consider **hybrid methods** for optimal results
- SMOTE requires at least k+1 minority samples (k=5 by default)

---

### 8.8 Data Visualization

**Page:** `7_📈_Visualization.py`  
**Module:** `modules/visualization.py`

Interactive visualization module for comprehensive data exploration.

**Visualization Types:**

| Visualization | Purpose |
|---------------|---------|
| **Distribution Plots** | Histograms, KDE plots for numeric columns |
| **Box Plots** | Outlier visualization and quartile analysis |
| **Bar Charts** | Categorical variable frequencies |
| **Correlation Matrix** | Heatmap of variable relationships |
| **Scatter Plots** | Relationship between two variables |
| **Missing Pattern Heatmap** | Visualize missing value patterns |
| **Before/After Comparisons** | Impact of cleaning operations |

**Features:**
- Interactive Plotly charts with zoom, pan, hover
- Export visualizations as images
- Customizable color palettes
- Save visualizations for report inclusion

---

### 8.9 Report Generation

**Page:** `8_📄_Reports.py`  
**Module:** `modules/report_generator.py`

Professional documentation and reporting capabilities.

**Report Types:**

| Report | Contents |
|--------|----------|
| **Executive Summary** | High-level overview, key metrics, recommendations |
| **Detailed Analysis** | Column-by-column analysis with statistics |
| **Methodology Report** | Documentation of all cleaning methods applied |
| **Audit Trail** | Complete history of all operations |
| **Comprehensive Report** | Combined full report with all sections |

**Export Formats:**
- **PDF**: Professional formatted document with visualizations
- **HTML**: Interactive web-based report
- **JSON**: Machine-readable format for integration

**Report Contents:**
- Dataset overview and statistics
- Data quality scores
- Missing value analysis
- Outlier detection results
- Cleaning operations applied
- Before/after comparisons
- Statistical test results
- Data balancing outcomes
- AI-generated insights and recommendations

---

### 8.10 AI Assistant

**Page:** `9_🤖_AI_Assistant.py`  
**Module:** `modules/ai_assistant.py`

Natural language conversational interface powered by **Groq's Llama 3.1** model.

**Capabilities:**

| Feature | Description |
|---------|-------------|
| **Ask Questions** | Query your data in natural language |
| **Get Recommendations** | Receive tailored cleaning suggestions |
| **Compare Methods** | Evaluate different cleaning approaches |
| **Impact Assessment** | Understand effects of operations |
| **Concept Explanation** | Learn statistical concepts in context |
| **Workflow Guidance** | Get step-by-step cleaning workflow |

**Example Queries:**
- "What's the best way to handle missing values in the income column?"
- "Compare median imputation vs KNN imputation for age variable"
- "What impact will removing outliers have on my analysis?"
- "Explain what skewness means for my sales data"
- "Which columns should I clean first?"

**Features:**
- Context-aware responses based on current data state
- Column-specific recommendations
- Conversation history with export capability
- Educational explanations alongside recommendations

---

## 9. Future Scope for Project Enhancement

The current system follows a **guided, click-based workflow** with user interaction. Future enhancements include:

### Short-term Enhancements:
- **Agentic AI Integration**: Enable step-by-step decision-making with minimal user involvement
- **Prompt-Driven Interface**: Allow users to perform data processing using natural language commands
- **Human-in-the-Loop Mechanisms**: Ensure transparency, control, and trust in AI-assisted decisions

### Long-term Vision:
- **Support for Additional File Formats**: JSON, Parquet, SQL databases
- **Advanced Rule-Based Validation**: Custom constraint definitions
- **Automated Workflow Scheduling**: Recurring data cleaning pipelines
- **Multi-User Collaboration**: Team-based data cleaning projects
- **Platform Integration**: Connect with popular data science platforms
- **Real-Time Processing**: Handle streaming data sources
- **Advanced NLP**: Enhanced text column cleaning using NLP techniques
- **Plugin Architecture**: Domain-specific cleaning extensions

---

## 10. Conclusion

The **Intelligent Data Cleaning Assistant** represents a significant advancement in making data preprocessing accessible, transparent, and efficient. By combining:

- **AI-powered recommendations** for intelligent decision support
- **Comprehensive statistical tools** for rigorous analysis
- **User-friendly interface** for accessibility
- **Complete audit trails** for reproducibility
- **Professional reporting** for documentation

The platform addresses the critical challenges faced by statistical agencies, researchers, and data professionals in preparing data for analysis. The integration of **hypothesis testing** and **data balancing** modules ensures that users can not only clean their data but also validate research questions and prepare datasets for machine learning applications.

---

**Project Version:** 1.0  
**Last Updated:** February 2026  
**Built for:** Statistical Agencies, Data Scientists, Researchers, Business Analysts

---

*Made with ❤️ for Data Professionals Worldwide*
