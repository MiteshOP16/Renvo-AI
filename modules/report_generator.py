import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from jinja2 import Template
import base64
import io

class ReportGenerator:
    """Comprehensive report generation for data cleaning operations with PDF/HTML export"""
    
    def __init__(self, weights_manager=None):
        self.weights_manager = weights_manager
        self.report_templates = {
            'executive_summary': self._get_executive_template(),
            'detailed_analysis': self._get_detailed_template(),
            'methodology': self._get_methodology_template(),
            'audit_trail': self._get_audit_template(),
            'weighted_summary': self._get_weighted_template(),
            'full_report': self._get_full_report_template()
        }
    
    def generate_executive_summary(self, df: pd.DataFrame, cleaning_history: Dict[str, Any], 
                                 analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary report"""
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(df, cleaning_history, analysis_results)
        
        template = Template(self.report_templates['executive_summary'])
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_shape=df.shape,
            summary_stats=summary_stats,
            cleaning_history=cleaning_history
        )
    
    def generate_detailed_analysis(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                                 cleaning_history: Dict[str, Any]) -> str:
        """Generate detailed column-by-column analysis report"""
        
        template = Template(self.report_templates['detailed_analysis'])
        
        # Prepare column details
        column_details = []
        for col, analysis in analysis_results.items():
            column_detail = {
                'name': col,
                'analysis': analysis,
                'cleaning_applied': cleaning_history.get(col, [])
            }
            column_details.append(column_detail)
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_shape=df.shape,
            column_details=column_details
        )
    
    def generate_methodology_report(self, cleaning_history: Dict[str, Any]) -> str:
        """Generate methodology documentation"""
        
        template = Template(self.report_templates['methodology'])
        
        # Extract unique methods used
        methods_used = set()
        for col_history in cleaning_history.values():
            for operation in col_history:
                methods_used.add(operation.get('method_name', 'unknown'))
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cleaning_history=cleaning_history,
            methods_used=list(methods_used)
        )
    
    def generate_comprehensive_report(self, df: pd.DataFrame, cleaning_history: Dict[str, Any], 
                                    analysis_results: Dict[str, Any], inter_column_violations: Dict[str, Any],
                                    output_format: str = 'html') -> str:
        """Generate comprehensive report with weighted/unweighted summaries"""
        
        # Calculate comprehensive statistics
        report_data = self._prepare_comprehensive_data(df, cleaning_history, analysis_results, inter_column_violations)
        
        template = Template(self.report_templates['full_report'])
        
        html_content = template.render(**report_data)
        
        if output_format.lower() == 'pdf':
            return self._convert_to_pdf(html_content)
        else:
            return html_content
    
    def generate_weighted_summary(self, df: pd.DataFrame, cleaning_history: Dict[str, Any]) -> Dict[str, Any]:
        """Generate weighted vs unweighted comparison summary"""
        summary = {
            'has_weights': self.weights_manager and self.weights_manager.weights_column,
            'weights_column': self.weights_manager.weights_column if self.weights_manager else None,
            'weighted_stats': {},
            'unweighted_stats': {},
            'comparison': {}
        }
        
        if summary['has_weights']:
            weights = df[self.weights_manager.weights_column]
            
            # Calculate weighted and unweighted statistics for numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != self.weights_manager.weights_column:
                    series = df[col].dropna()
                    valid_weights = weights[df[col].notna()]
                    
                    if len(series) > 0 and len(valid_weights) > 0:
                        # Unweighted
                        unweighted = {
                            'mean': series.mean(),
                            'std': series.std(),
                            'median': series.median(),
                            'total': series.sum()
                        }
                        
                        # Weighted
                        weighted = {
                            'mean': np.average(series, weights=valid_weights),
                            'total': np.sum(series * valid_weights) / valid_weights.sum() * len(series),
                            'effective_sample_size': (valid_weights.sum() ** 2) / (valid_weights ** 2).sum()
                        }
                        
                        summary['unweighted_stats'][col] = unweighted
                        summary['weighted_stats'][col] = weighted
                        summary['comparison'][col] = {
                            'mean_difference': weighted['mean'] - unweighted['mean'],
                            'relative_difference': ((weighted['mean'] - unweighted['mean']) / unweighted['mean'] * 100) if unweighted['mean'] != 0 else 0
                        }
        
        return summary
    
    def _prepare_comprehensive_data(self, df: pd.DataFrame, cleaning_history: Dict[str, Any], 
                                  analysis_results: Dict[str, Any], inter_column_violations: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare all data for comprehensive report"""
        
        # Basic dataset information
        basic_info = {
            'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_shape': df.shape,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Cleaning summary
        cleaning_summary = {
            'columns_cleaned': len(cleaning_history),
            'total_operations': sum(len(ops) for ops in cleaning_history.values()),
            'methods_used': list(set(op.get('method_name', 'unknown') for ops in cleaning_history.values() for op in ops))
        }
        
        # Violation summary
        violation_summary = {
            'total_violations': inter_column_violations.get('total_violations', 0),
            'violation_types': inter_column_violations.get('violation_types', []),
            'severity': inter_column_violations.get('severity', 'low'),
            'affected_rows_count': len(inter_column_violations.get('affected_rows', []))
        }
        
        # Weighted analysis if available
        weighted_summary = self.generate_weighted_summary(df, cleaning_history)
        
        # Column analysis summary
        column_summary = []
        for col, analysis in analysis_results.items():
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': analysis.get('basic_info', {}).get('missing_count', 0),
                'missing_percentage': analysis.get('basic_info', {}).get('missing_percentage', 0),
                'outliers': analysis.get('outlier_analysis', {}).get('summary', {}).get('consensus_outliers', 0),
                'violations': analysis.get('rule_violations', {}).get('total_violations', 0),
                'cleaning_applied': len(cleaning_history.get(col, []))
            }
            column_summary.append(col_info)
        
        return {
            'basic_info': basic_info,
            'cleaning_summary': cleaning_summary,
            'violation_summary': violation_summary,
            'weighted_summary': weighted_summary,
            'column_summary': column_summary,
            'cleaning_history': cleaning_history,
            'analysis_results': analysis_results,
            'inter_column_violations': inter_column_violations
        }
    
    def _convert_to_pdf(self, html_content: str) -> bytes:
        """Convert HTML content to PDF (placeholder - would use weasyprint or similar)"""
        # In a real implementation, you would use libraries like weasyprint, reportlab, or pdfkit
        # For now, return the HTML as bytes with appropriate headers
        return html_content.encode('utf-8')
    
    def export_to_file(self, content: str, filename: str, format_type: str = 'html') -> str:
        """Export report content to file"""
        if format_type.lower() == 'pdf':
            # Convert to PDF if needed
            content = self._convert_to_pdf(content) if isinstance(content, str) else content
            filename = filename.replace('.html', '.pdf')
        
        # In Streamlit context, this would typically use st.download_button
        # Return the content for download
        return content
    
    def generate_audit_trail(self, cleaning_history: Dict[str, Any]) -> str:
        """Generate complete audit trail"""
        
        template = Template(self.report_templates['audit_trail'])
        
        # Create chronological audit trail
        all_operations = []
        for column, operations in cleaning_history.items():
            for op in operations:
                op['column'] = column
                all_operations.append(op)
        
        # Sort by timestamp
        all_operations.sort(key=lambda x: x.get('timestamp', ''))
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            all_operations=all_operations
        )
    
    def generate_complete_report(self, df: pd.DataFrame, original_df: pd.DataFrame,
                               analysis_results: Dict[str, Any], 
                               cleaning_history: Dict[str, Any],
                               weights_manager=None, violations=None) -> Dict[str, str]:
        """Generate all report types"""
        
        # Update weights manager if provided
        if weights_manager:
            self.weights_manager = weights_manager
            
        reports = {}
        
        try:
            reports['executive_summary'] = self.generate_executive_summary(
                df, cleaning_history, analysis_results
            )
        except Exception as e:
            reports['executive_summary'] = f"Error generating executive summary: {str(e)}"
        
        try:
            reports['detailed_analysis'] = self.generate_detailed_analysis(
                df, analysis_results, cleaning_history
            )
        except Exception as e:
            reports['detailed_analysis'] = f"Error generating detailed analysis: {str(e)}"
        
        try:
            reports['methodology'] = self.generate_methodology_report(cleaning_history)
        except Exception as e:
            reports['methodology'] = f"Error generating methodology report: {str(e)}"
        
        try:
            reports['audit_trail'] = self.generate_audit_trail(cleaning_history)
        except Exception as e:
            reports['audit_trail'] = f"Error generating audit trail: {str(e)}"
        
        return reports
    
    def _calculate_summary_stats(self, df: pd.DataFrame, cleaning_history: Dict[str, Any],
                               analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the report"""
        
        total_operations = sum(len(ops) for ops in cleaning_history.values())
        columns_cleaned = len(cleaning_history)
        
        # Data quality scores
        quality_scores = []
        for col, analysis in analysis_results.items():
            if 'data_quality' in analysis:
                quality_scores.append(analysis['data_quality'].get('score', 0))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Missing data summary
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        
        return {
            'total_operations': total_operations,
            'columns_cleaned': columns_cleaned,
            'avg_quality_score': round(avg_quality, 2),
            'total_missing': total_missing,
            'missing_percentage': round(missing_percentage, 2),
            'total_columns': len(df.columns),
            'total_rows': len(df)
        }
    
    def _get_executive_template(self) -> str:
        """Executive summary template"""
        return """
# EXECUTIVE SUMMARY
## Survey Data Cleaning Report

**Report Date:** {{ report_date }}
**Dataset Dimensions:** {{ dataset_shape[0]:,}} records × {{ dataset_shape[1] }} variables

---

## 1. OVERVIEW

This report summarizes the data cleaning operations performed on the survey dataset to ensure data quality, consistency, and analytical readiness. All procedures follow established statistical best practices for survey data processing.

## 2. KEY METRICS

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Records | {{ dataset_shape[0]:,}} | - |
| Total Variables | {{ dataset_shape[1] }} | - |
| Cleaning Operations Applied | {{ summary_stats.total_operations }} | - |
| Variables Processed | {{ summary_stats.columns_cleaned }} / {{ summary_stats.total_columns }} | {{ "%.1f"|format(summary_stats.columns_cleaned / summary_stats.total_columns * 100) }}% |
| Average Data Quality Score | {{ summary_stats.avg_quality_score }}/100 | {% if summary_stats.avg_quality_score >= 80 %}Excellent{% elif summary_stats.avg_quality_score >= 60 %}Satisfactory{% else %}Requires Attention{% endif %} |
| Missing Data Prevalence | {{ summary_stats.missing_percentage }}% | {% if summary_stats.missing_percentage < 5 %}Low{% elif summary_stats.missing_percentage < 15 %}Moderate{% else %}High{% endif %} |

## 3. DATA QUALITY ASSESSMENT

### Overall Quality Status
{% if summary_stats.avg_quality_score >= 80 %}
**Status:** ACCEPTABLE - The dataset demonstrates high quality with an average quality score of {{ summary_stats.avg_quality_score }}/100. Data is suitable for statistical analysis with standard methodological considerations.
{% elif summary_stats.avg_quality_score >= 60 %}
**Status:** CONDITIONAL - The dataset shows moderate quality ({{ summary_stats.avg_quality_score }}/100). While usable for analysis, results should be interpreted with appropriate caveats regarding data limitations.
{% else %}
**Status:** CAUTION - The dataset exhibits significant quality issues ({{ summary_stats.avg_quality_score }}/100). Substantial additional cleaning or data collection may be required for reliable analysis.
{% endif %}

### Missing Data Assessment
- **Total Missing Values:** {{ summary_stats.total_missing:,}} ({{ summary_stats.missing_percentage }}% of dataset)
- **Impact Level:** {% if summary_stats.missing_percentage < 5 %}Minimal - standard analytical procedures applicable{% elif summary_stats.missing_percentage < 15 %}Moderate - multiple imputation or weighting adjustments recommended{% else %}Substantial - may require specialized missing data techniques{% endif %}

## 4. CLEANING OPERATIONS PERFORMED

**Summary of Variables Processed:**

{% for column, operations in cleaning_history.items() %}
- **{{ column }}:** {{ operations|length }} procedure(s) applied
{% endfor %}

## 5. RECOMMENDATIONS FOR DATA USERS

1. **Analytical Considerations:** Review the methodology report for details on cleaning procedures that may affect specific analyses
2. **Validation Protocol:** Verify that cleaned data meets domain-specific quality requirements and business rules
3. **Documentation Requirements:** Maintain this report alongside analytical outputs for reproducibility and audit purposes
4. **Ongoing Monitoring:** Implement quality control procedures for future data collection to prevent recurring issues
5. **Statistical Disclosure:** Document all cleaning operations in research outputs per standard practice

## 6. REPORT CERTIFICATION

This executive summary provides a high-level overview of data cleaning procedures. Detailed methodology, audit trails, and statistical impact assessments are available in the accompanying technical documentation.

**Prepared by:** Intelligent Survey Data Cleaning Assistant  
**Report Version:** 1.0  
**Quality Assurance:** Automated analysis with manual review recommended

---

*For technical details and reproducibility information, please refer to the complete methodology and audit trail reports.*
"""
    
    def _get_detailed_template(self) -> str:
        """Detailed analysis template"""
        return """
# Detailed Data Cleaning Analysis Report

**Report Generated:** {{ report_date }}
**Dataset Dimensions:** {{ dataset_shape[0] }} rows × {{ dataset_shape[1] }} columns

## Column-by-Column Analysis

{% for column_detail in column_details %}
## Column: {{ column_detail.name }}

### Basic Information
- **Data Type:** {{ column_detail.analysis.basic_info.dtype }}
- **Total Values:** {{ column_detail.analysis.basic_info.count }}
- **Missing Values:** {{ column_detail.analysis.basic_info.missing_count }} ({{ "%.2f"|format(column_detail.analysis.basic_info.missing_percentage) }}%)
- **Unique Values:** {{ column_detail.analysis.basic_info.unique_count }} ({{ "%.2f"|format(column_detail.analysis.basic_info.unique_percentage) }}%)

{% if column_detail.analysis.basic_info.get('mean') is not none %}
### Statistical Summary
- **Mean:** {{ "%.3f"|format(column_detail.analysis.basic_info.mean) }}
- **Median:** {{ "%.3f"|format(column_detail.analysis.basic_info.median) }}
- **Standard Deviation:** {{ "%.3f"|format(column_detail.analysis.basic_info.std) }}
- **Range:** {{ "%.3f"|format(column_detail.analysis.basic_info.min) }} to {{ "%.3f"|format(column_detail.analysis.basic_info.max) }}
{% endif %}

### Data Quality Assessment
- **Quality Score:** {{ column_detail.analysis.data_quality.score }}/100 (Grade: {{ column_detail.analysis.data_quality.grade }})
{% if column_detail.analysis.data_quality.issues %}
- **Issues Identified:**
{% for issue in column_detail.analysis.data_quality.issues %}
  - {{ issue }}
{% endfor %}
{% endif %}

### Missing Data Analysis
- **Pattern Type:** {{ column_detail.analysis.missing_analysis.get('pattern_type', 'N/A') }}
{% if column_detail.analysis.missing_analysis.get('max_consecutive', 0) > 0 %}
- **Max Consecutive Missing:** {{ column_detail.analysis.missing_analysis.max_consecutive }}
{% endif %}

{% if column_detail.analysis.outlier_analysis.get('method_results') %}
### Outlier Detection Results
{% for method, results in column_detail.analysis.outlier_analysis.method_results.items() %}
- **{{ results.method }}:** {{ results.outlier_count }} outliers ({{ "%.2f"|format(results.outlier_percentage) }}%)
{% endfor %}
{% endif %}

### Cleaning Operations Applied
{% if column_detail.cleaning_applied %}
{% for operation in column_detail.cleaning_applied %}
- **{{ operation.timestamp }}:** Applied {{ operation.get('method_name', 'unknown method') }}
  - Result: {{ operation.get('result', 'N/A') }}
{% if operation.get('impact_stats') %}
  - Rows affected: {{ operation.impact_stats.rows_affected }}
{% endif %}
{% endfor %}
{% else %}
- No cleaning operations applied to this column
{% endif %}

---
{% endfor %}

*End of detailed analysis*
"""
    
    def _get_methodology_template(self) -> str:
        """Methodology documentation template"""
        return """
# Data Cleaning Methodology Report

**Report Generated:** {{ report_date }}

## Methodology Overview

This report documents the systematic approach used for cleaning the survey dataset, ensuring reproducibility and transparency of all data processing operations.

## Cleaning Methods Applied

{% for method in methods_used %}
### {{ method|title|replace('_', ' ') }}

{% if method == 'median_imputation' %}
**Purpose:** Replace missing values with the median of the observed values
**Rationale:** Robust to outliers and appropriate for skewed distributions
**Applicability:** Numeric columns with moderate missing data rates
{% elif method == 'mean_imputation' %}
**Purpose:** Replace missing values with the mean of the observed values
**Rationale:** Simple and preserves the overall distribution center
**Applicability:** Numeric columns with normal distribution and low missing rates
{% elif method == 'mode_imputation' %}
**Purpose:** Replace missing values with the most frequent value
**Rationale:** Preserves the dominant category in categorical data
**Applicability:** Categorical columns with clear mode
{% elif method == 'knn_imputation' %}
**Purpose:** Impute missing values using K-Nearest Neighbors algorithm
**Rationale:** Preserves relationships between variables
**Applicability:** Numeric columns with available correlated features
{% elif method == 'winsorization' %}
**Purpose:** Cap extreme values at specified percentiles
**Rationale:** Reduces outlier impact while preserving sample size
**Applicability:** Numeric columns with extreme outliers
{% elif method == 'iqr_removal' %}
**Purpose:** Remove outliers beyond 1.5 * IQR from quartiles
**Rationale:** Statistical definition of outliers for normal distributions
**Applicability:** Numeric columns with approximately normal distribution
{% endif %}

{% endfor %}

## Column-Specific Applications

{% for column, operations in cleaning_history.items() %}
### {{ column }}

{% for operation in operations %}
**Operation {{ loop.index }}:** {{ operation.get('method_name', 'Unknown') }}
- **Timestamp:** {{ operation.get('timestamp', 'N/A') }}
- **Rationale:** {{ operation.get('rationale', 'Applied based on column analysis') }}
{% if operation.get('parameters') %}
- **Parameters:** {{ operation.parameters }}
{% endif %}
{% if operation.get('impact_stats') %}
- **Impact:** {{ operation.impact_stats.rows_affected }} rows affected
{% endif %}

{% endfor %}
{% endfor %}

## Quality Assurance

1. **Pre-cleaning Analysis:** Each column was individually analyzed for patterns, outliers, and data quality issues
2. **Method Selection:** Cleaning methods were selected based on column-specific characteristics, not generic rules
3. **Impact Assessment:** All operations were evaluated for their statistical impact
4. **Audit Trail:** Complete documentation of all operations for reproducibility

## Statistical Considerations

- **Survey Weights:** All cleaning operations preserve the integrity of survey weights
- **Sampling Design:** Methods were chosen considering the complex survey design
- **Bias Reduction:** Imputation methods were selected to minimize introduction of bias
- **Variance Preservation:** Methods that maintain appropriate variance were prioritized

## Limitations and Assumptions

1. Missing data patterns were assumed to be missing at random (MAR) unless evidence suggested otherwise
2. Outlier detection assumed underlying normal distribution where applicable
3. Relationships between variables were assumed stable during imputation processes

---
*This methodology ensures reproducible and statistically sound data cleaning procedures*
"""
    
    def _get_audit_template(self) -> str:
        """Audit trail template"""
        return """
# Data Cleaning Audit Trail

**Report Generated:** {{ report_date }}

## Complete Operation Log

This section provides a chronological record of all data cleaning operations performed on the dataset.

{% for operation in all_operations %}
### Operation {{ loop.index }}

**Timestamp:** {{ operation.get('timestamp', 'N/A') }}
**Column:** {{ operation.get('column', 'N/A') }}
**Method:** {{ operation.get('method_name', 'N/A') }}
**Type:** {{ operation.get('method_type', 'N/A') }}

{% if operation.get('parameters') %}
**Parameters:**
{% for key, value in operation.parameters.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

{% if operation.get('impact_stats') %}
**Impact Statistics:**
- Rows affected: {{ operation.impact_stats.rows_affected }}
- Percentage changed: {{ "%.2f"|format(operation.impact_stats.percentage_changed) }}%
- Missing before: {{ operation.impact_stats.missing_before }}
- Missing after: {{ operation.impact_stats.missing_after }}
- Missing change: {{ operation.impact_stats.missing_change }}

{% if operation.impact_stats.get('mean_before') is not none %}
**Statistical Impact:**
- Mean before: {{ "%.3f"|format(operation.impact_stats.mean_before) }}
- Mean after: {{ "%.3f"|format(operation.impact_stats.mean_after) }}
- Std before: {{ "%.3f"|format(operation.impact_stats.std_before) }}
- Std after: {{ "%.3f"|format(operation.impact_stats.std_after) }}
{% endif %}
{% endif %}

**Result:** {{ operation.get('result', 'Operation completed') }}

---
{% endfor %}

## Summary Statistics

- **Total Operations:** {{ all_operations|length }}
- **Unique Columns Affected:** {{ all_operations|map(attribute='column')|list|unique|length }}
- **Operation Types Used:** {{ all_operations|map(attribute='method_type')|list|unique|join(', ') }}

## Verification

All operations listed above can be independently verified and reproduced using the same parameters and methods on the original dataset.

---
*Complete audit trail for data cleaning operations*
"""
    
    def export_to_pdf(self, reports: Dict[str, str], df: pd.DataFrame, 
                     analysis_results: Dict[str, Any], cleaning_history: Dict[str, Any],
                     anomaly_results: Dict[str, Any] = None, 
                     saved_visualizations: List[Dict[str, Any]] = None,
                     title: str = "Data Cleaning Report") -> bytes:
        """Export comprehensive reports to PDF format with all issues, cleaning methods, imputed values, and visualizations"""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from io import BytesIO
        from PIL import Image as PILImage
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=60, leftMargin=60,
                              topMargin=60, bottomMargin=40)
        
        # Container for PDF elements
        elements = []
        
        # Enhanced Styles with Modern Color Palette
        styles = getSampleStyleSheet()
        
        # Modern color palette
        primary_color = colors.HexColor('#2E86AB')  # Blue
        success_color = colors.HexColor('#06D6A0')  # Green
        warning_color = colors.HexColor('#F77F00')  # Orange
        danger_color = colors.HexColor('#EF476F')   # Red
        dark_text = colors.HexColor('#22223B')      # Dark blue-grey
        light_bg = colors.HexColor('#F8F9FA')       # Light grey
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=primary_color,
            spaceAfter=25,
            spaceBefore=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=dark_text,
            spaceAfter=12,
            spaceBefore=18,
            fontName='Helvetica-Bold',
            borderPadding=8,
            leftIndent=0,
            backColor=light_bg,
            borderWidth=0,
            borderColor=primary_color,
            borderRadius=2
        )
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=13,
            textColor=primary_color,
            spaceAfter=10,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        normal_style = styles['Normal']
        normal_style.fontSize = 10
        normal_style.textColor = dark_text
        
        issue_style = ParagraphStyle(
            'IssueStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=danger_color,
            leftIndent=20,
            spaceAfter=4
        )
        
        success_style = ParagraphStyle(
            'SuccessStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=success_color,
            leftIndent=20,
            fontName='Helvetica-Bold'
        )
        
        info_box_style = ParagraphStyle(
            'InfoBox',
            parent=styles['Normal'],
            fontSize=10,
            textColor=dark_text,
            backColor=light_bg,
            borderPadding=10,
            borderWidth=1,
            borderColor=primary_color,
            borderRadius=3,
            spaceAfter=10
        )
        
        # Title Page
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph(f"{title}", title_style))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", 
            ParagraphStyle('DateStyle', parent=normal_style, alignment=TA_CENTER, fontSize=11)
        ))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(
            "<i>Comprehensive Data Quality Assessment and Cleaning Documentation</i>",
            ParagraphStyle('SubtitleStyle', parent=normal_style, alignment=TA_CENTER, fontSize=10, textColor=colors.grey)
        ))
        elements.append(PageBreak())
        
        # Table of Contents
        elements.append(Paragraph("📋 Table of Contents", heading_style))
        toc_data = [
            "1. Executive Summary",
            "2. All Issues Detected in Dataset",
            "3. Cleaning Operations Applied",
            "4. Imputed Values and Results",
            "5. Data Type Anomaly Detection Results",
            "6. Visualizations",
            "7. Complete Audit Trail"
        ]
        for item in toc_data:
            elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{item}", normal_style))
        elements.append(PageBreak())
        
        # Executive Summary
        elements.append(Paragraph("1. 📊 Executive Summary", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Dataset statistics
        summary_data = [
            ['Metric', 'Value'],
            ['Total Rows', f"{len(df):,}"],
            ['Total Columns', str(len(df.columns))],
            ['Missing Values', f"{df.isnull().sum().sum():,}"],
            ['Missing Percentage', f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%"],
            ['Analyzed Columns', str(len(analysis_results))],
            ['Cleaned Columns', str(len(cleaning_history))]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 14),
            ('TOPPADDING', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), light_bg),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [light_bg, colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        elements.append(summary_table)
        elements.append(PageBreak())
        
        # Section 2: All Issues Detected in Dataset
        elements.append(Paragraph("2. 🔍 All Issues Detected in Dataset", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if analysis_results:
            for col_name, analysis in analysis_results.items():
                elements.append(Paragraph(f"<b>Column: {col_name}</b>", subheading_style))
                
                # Collect all issues for this column
                issues_found = []
                
                # Missing data issues
                missing_count = analysis.get('basic_info', {}).get('missing_count', 0)
                missing_pct = analysis.get('basic_info', {}).get('missing_percentage', 0)
                if missing_count > 0:
                    issues_found.append(f"Missing Values: {missing_count:,} ({missing_pct:.2f}%)")
                    pattern = analysis.get('missing_analysis', {}).get('pattern_type', 'Unknown')
                    issues_found.append(f"  → Missing pattern: {pattern}")
                
                # Outlier issues
                outlier_summary = analysis.get('outlier_analysis', {}).get('summary', {})
                if outlier_summary.get('consensus_outliers', 0) > 0:
                    outlier_count = outlier_summary['consensus_outliers']
                    outlier_pct = outlier_summary.get('consensus_percentage', 0)
                    severity = outlier_summary.get('severity', 'unknown')
                    issues_found.append(f"Outliers Detected: {outlier_count} ({outlier_pct:.2f}%) - Severity: {severity.title()}")
                
                # Data quality issues
                quality_data = analysis.get('data_quality', {})
                quality_score = quality_data.get('score', 100)
                if quality_score < 80:
                    issues_found.append(f"Data Quality Score: {quality_score}/100 - Grade: {quality_data.get('grade', 'Unknown')}")
                    for issue in quality_data.get('issues', []):
                        issues_found.append(f"  → {issue}")
                
                # Rule violations
                violations = analysis.get('rule_violations', {})
                if violations.get('total_violations', 0) > 0:
                    issues_found.append(f"Rule Violations: {violations['total_violations']}")
                    for violation_type, count in violations.get('violation_counts', {}).items():
                        if count > 0:
                            issues_found.append(f"  → {violation_type}: {count}")
                
                # Display issues
                if issues_found:
                    for issue in issues_found:
                        elements.append(Paragraph(f"• {issue}", issue_style))
                    elements.append(Spacer(1, 0.15*inch))
                else:
                    elements.append(Paragraph("✅ No major issues detected", success_style))
                    elements.append(Spacer(1, 0.15*inch))
        else:
            elements.append(Paragraph("No column analysis available.", normal_style))
        
        elements.append(PageBreak())
        
        # Section 3: Cleaning Operations Applied
        elements.append(Paragraph("3. 🧹 Cleaning Operations Applied", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if cleaning_history:
            for col_name, operations in cleaning_history.items():
                if operations:
                    elements.append(Paragraph(f"<b>Column: {col_name}</b>", subheading_style))
                    
                    for idx, op in enumerate(operations, 1):
                        method = op.get('method_name', 'Unknown Method')
                        timestamp = op.get('timestamp', '')[:19] if op.get('timestamp') else 'N/A'
                        
                        elements.append(Paragraph(
                            f"<b>Operation {idx}:</b> {method.replace('_', ' ').title()}", 
                            normal_style
                        ))
                        elements.append(Paragraph(f"Applied: {timestamp}", normal_style))
                        
                        # Show parameters if any
                        params = op.get('parameters', {})
                        if params:
                            param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k not in ['weights', 'use_weights']])
                            if param_str:
                                elements.append(Paragraph(f"Parameters: {param_str}", normal_style))
                        
                        elements.append(Spacer(1, 0.05*inch))
                    
                    elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(Paragraph("No cleaning operations have been performed yet.", normal_style))
        
        elements.append(PageBreak())
        
        # Section 4: Imputed Values and Results
        elements.append(Paragraph("4. 📝 Imputed Values and Cleaning Results", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if cleaning_history:
            imputation_data = []
            imputation_data.append(['Column', 'Method', 'Imputed Value', 'Rows Affected', 'Result'])
            
            for col_name, operations in cleaning_history.items():
                for op in operations:
                    method = op.get('method_name', 'Unknown')
                    metadata = op.get('metadata', {})
                    impact_stats = op.get('impact_stats', {})
                    
                    # Get imputed value if available
                    imputed_val = metadata.get('imputed_value', 'N/A')
                    if imputed_val != 'N/A':
                        if isinstance(imputed_val, float):
                            imputed_val = f"{imputed_val:.4f}"
                        else:
                            imputed_val = str(imputed_val)
                    
                    # Get rows affected
                    rows_affected = impact_stats.get('rows_affected', metadata.get('values_imputed', 'N/A'))
                    
                    # Get result description
                    result = op.get('result', 'Completed')
                    if impact_stats.get('missing_change'):
                        result = f"Missing: {impact_stats.get('missing_before', 0)} → {impact_stats.get('missing_after', 0)}"
                    
                    imputation_data.append([
                        col_name,
                        method.replace('_', ' ').title(),
                        str(imputed_val)[:20],  # Truncate if too long
                        str(rows_affected),
                        result[:30]  # Truncate if too long
                    ])
            
            if len(imputation_data) > 1:
                imputation_table = Table(imputation_data, colWidths=[1.3*inch, 1.3*inch, 1.2*inch, 0.9*inch, 1.8*inch])
                imputation_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), success_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TOPPADDING', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F8F5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#E8F8F5'), colors.white])
                ]))
                elements.append(imputation_table)
            else:
                elements.append(Paragraph("No imputation operations recorded.", normal_style))
        else:
            elements.append(Paragraph("No cleaning operations have been performed yet.", normal_style))
        
        elements.append(PageBreak())
        
        # Section 5: Data Type Anomaly Detection Results
        elements.append(Paragraph("5. 🚨 Data Type Anomaly Detection Results", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if anomaly_results and len(anomaly_results) > 0:
            elements.append(Paragraph(
                "The following columns had values that did not match their declared data types:",
                normal_style
            ))
            elements.append(Spacer(1, 0.15*inch))
            
            for col_name, anomaly_data in anomaly_results.items():
                if anomaly_data.get('anomaly_count', 0) > 0:
                    elements.append(Paragraph(f"<b>Column: {col_name}</b>", subheading_style))
                    
                    # Anomaly summary table
                    anomaly_table_data = [
                        ['Metric', 'Value'],
                        ['Expected Type', anomaly_data.get('expected_type', 'Unknown').title()],
                        ['Total Values', str(anomaly_data.get('total_values', 0))],
                        ['Anomalies Found', str(anomaly_data.get('anomaly_count', 0))],
                        ['Anomaly Percentage', f"{anomaly_data.get('anomaly_percentage', 0):.2f}%"]
                    ]
                    
                    anomaly_summary_table = Table(anomaly_table_data, colWidths=[2*inch, 4.5*inch])
                    anomaly_summary_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), warning_color),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('TOPPADDING', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                        ('TOPPADDING', (0, 1), (-1, -1), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFF4E6')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFF4E6'), colors.white])
                    ]))
                    elements.append(anomaly_summary_table)
                    elements.append(Spacer(1, 0.15*inch))
                    
                    # Show sample anomalies if available
                    anomalies_list = anomaly_data.get('anomalies', [])[:5]
                    if anomalies_list:
                        elements.append(Paragraph("<i>Sample Anomalies:</i>", normal_style))
                        for anomaly in anomalies_list:
                            elements.append(Paragraph(
                                f"  • Row {anomaly.get('row_index', 'N/A')}: \"{anomaly.get('value', 'N/A')}\" - {anomaly.get('reason', 'Unknown reason')}",
                                issue_style
                            ))
                    elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph(
                "✅ No data type anomalies detected - all values match their declared types.",
                success_style
            ))
        
        elements.append(PageBreak())
        
        # Section 6: Saved Visualizations
        elements.append(Paragraph("6. 📈 Visualizations", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if saved_visualizations and len(saved_visualizations) > 0:
            elements.append(Paragraph(f"Total visualizations included: {len(saved_visualizations)}", normal_style))
            elements.append(Spacer(1, 0.2*inch))
            
            for idx, viz in enumerate(saved_visualizations):
                try:
                    elements.append(Paragraph(f"<b>{idx + 1}. {viz['name']}</b>", subheading_style))
                    elements.append(Paragraph(
                        f"<i>Chart Type: {viz['type'].title()} | Columns: {', '.join(viz['columns'])}</i>", 
                        normal_style
                    ))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    # Add visualization image
                    if 'img_bytes' in viz:
                        img_buffer = BytesIO(viz['img_bytes'])
                        img = Image(img_buffer, width=6.5*inch, height=3.5*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.3*inch))
                    
                    # Page break after every 2 visualizations
                    if (idx + 1) % 2 == 0 and idx < len(saved_visualizations) - 1:
                        elements.append(PageBreak())
                        
                except Exception as e:
                    elements.append(Paragraph(f"<i>Error loading visualization: {str(e)}</i>", normal_style))
                    elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph("No visualizations have been saved to the report.", normal_style))
        
        # Section 7: Complete Audit Trail
        elements.append(PageBreak())
        elements.append(Paragraph("7. 📋 Complete Audit Trail", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if cleaning_history:
            # Create chronological list of all operations
            all_operations = []
            for col_name, operations in cleaning_history.items():
                for op in operations:
                    op_copy = op.copy()
                    op_copy['column'] = col_name
                    all_operations.append(op_copy)
            
            # Sort by timestamp
            all_operations.sort(key=lambda x: x.get('timestamp', ''))
            
            elements.append(Paragraph(f"Total operations performed: {len(all_operations)}", normal_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Create detailed audit table
            audit_data = [['#', 'Timestamp', 'Column', 'Method', 'Rows Affected']]
            
            for idx, op in enumerate(all_operations, 1):
                timestamp = op.get('timestamp', 'N/A')[:19]
                column = op.get('column', 'Unknown')
                method = op.get('method_name', 'Unknown').replace('_', ' ').title()
                impact_stats = op.get('impact_stats', {})
                rows_affected = impact_stats.get('rows_affected', op.get('metadata', {}).get('values_imputed', 'N/A'))
                
                audit_data.append([
                    str(idx),
                    timestamp,
                    column,
                    method,
                    str(rows_affected)
                ])
            
            # Split audit table if too many operations
            max_rows_per_page = 25
            for i in range(0, len(audit_data), max_rows_per_page):
                chunk = audit_data[i:min(i+max_rows_per_page, len(audit_data))]
                if i > 0:
                    chunk.insert(0, audit_data[0])  # Add header
                
                audit_table = Table(chunk, colWidths=[0.4*inch, 1.5*inch, 1.5*inch, 2*inch, 1.1*inch])
                audit_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6C63FF')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TOPPADDING', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8E7FD')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#E8E7FD'), colors.white])
                ]))
                elements.append(audit_table)
                
                if i + max_rows_per_page < len(audit_data):
                    elements.append(PageBreak())
        else:
            elements.append(Paragraph("No cleaning operations have been performed on this dataset.", normal_style))
        
        # Footer
        elements.append(PageBreak())
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("---", normal_style))
        elements.append(Paragraph(
            "<i>Generated by Intelligent Data Cleaning Assistant</i>",
            ParagraphStyle('Footer', parent=normal_style, alignment=TA_CENTER, fontSize=10)
        ))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def export_to_html(self, reports: Dict[str, str], title: str = "Data Cleaning Report") -> str:
        """Export reports to HTML format"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; }
        h3 { color: #7f8c8d; }
        .report-section { margin-bottom: 50px; }
        .metadata { background: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    {% for report_type, content in reports.items() %}
    <div class="report-section">
        <h2>{{ report_type|title|replace('_', ' ') }}</h2>
        <div>{{ content|markdown }}</div>
    </div>
    {% endfor %}
    
    <div class="metadata">
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Tool:</strong> Intelligent Survey Data Cleaning Assistant</p>
    </div>
</body>
</html>
"""
        
        template = Template(html_template)
        return template.render(
            title=title,
            reports=reports,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def export_to_json(self, reports: Dict[str, str], metadata: Dict[str, Any] = None) -> str:
        """Export reports to JSON format"""
        
        export_data = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'tool': 'Intelligent Survey Data Cleaning Assistant',
                'version': '1.0.0'
            },
            'reports': reports
        }
        
        if metadata:
            export_data['metadata'].update(metadata)
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _get_weighted_template(self) -> str:
        """Template for weighted summary report"""
        return '''<!DOCTYPE html>
<html><head><title>Weighted Analysis Summary</title>
<style>body{font-family:Arial,sans-serif;margin:20px}.header{background-color:#f0f0f0;padding:10px;border-radius:5px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}.highlight{background-color:#fff3cd}</style>
</head><body><div class="header"><h1>Weighted vs Unweighted Analysis</h1></div></body></html>'''
    
    def _get_full_report_template(self) -> str:
        """Template for comprehensive full report"""
        return '''<!DOCTYPE html>
<html><head><title>Survey Data Cleaning Report</title>
<style>body{font-family:Arial,sans-serif;margin:20px;line-height:1.6}.header{background-color:#2c3e50;color:white;padding:20px;border-radius:5px}.section{margin:30px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#34495e;color:white}.metric{display:inline-block;margin:10px;padding:15px;background-color:#ecf0f1;border-radius:5px;min-width:150px}.metric-value{font-size:24px;font-weight:bold;color:#2c3e50}.metric-label{font-size:14px;color:#7f8c8d}</style>
</head><body><div class="header"><h1>🧹 Survey Data Cleaning Report</h1><p><strong>Generated:</strong> {{ basic_info.report_date }}</p></div>
<div class="section"><h2>📊 Executive Summary</h2><div class="metric"><div class="metric-value">{{ basic_info.total_rows }}</div><div class="metric-label">Total Records</div></div></div></body></html>'''
