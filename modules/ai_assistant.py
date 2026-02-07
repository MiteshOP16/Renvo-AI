import streamlit as st
import os
from groq import Groq
from typing import Dict, List, Any, Optional
import json
import pandas as pd
from datetime import datetime

class AIAssistant:
    """AI-powered conversational assistant for data cleaning guidance"""
    
    def __init__(self):
        # Load key from Streamlit Secrets first
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                self.groq_api_key = st.secrets["GROQ_API_KEY"]
            else:
                # fallback for local dev
                self.groq_api_key = os.getenv("GROQ_API_KEY")
        except Exception:
            self.groq_api_key = os.getenv("GROQ_API_KEY")

        self.client = None
        self.model = "llama-3.1-8b-instant"
        self.conversation_history = []
        self.context = {}

        # Try to initialize Groq client
        if self.groq_api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print("❌ Failed to initialize Groq client:", e)
                self.client = None
        else:
            print("❌ GROQ_API_KEY missing — AI disabled.")

    
    def set_context(self, dataset_info: Dict[str, Any], column_analysis: Optional[Dict[str, Any]] = None):
        """Set the current context for AI assistance"""
        self.context = {
            'dataset_shape': dataset_info.get('shape', 'Unknown'),
            'column_count': dataset_info.get('columns', 0),
            'missing_data_summary': dataset_info.get('missing_summary', {}),
            'column_types': dataset_info.get('column_types', {}),
            'current_column_analysis': column_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def ask_question(self, question: str, column_specific: Optional[str] = None, 
                   current_data_state: Optional[Dict[str, Any]] = None) -> str:
        """Ask the AI assistant a question with current context"""
        if self.client is None:
           try:
              self.client = Groq(api_key=self.groq_api_key)
           except Exception as e:
              return f"AI initialization failed: {str(e)}"

        
        try:
            # Update context with current data state if provided
            if current_data_state:
                self._update_context_with_current_state(current_data_state)
            
            # Build context-aware prompt
            system_prompt = self._build_system_prompt(column_specific)
            user_message = self._build_user_message(question, column_specific)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content or "No response received"
            
            # Store conversation
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'column': column_specific,
                'response': ai_response
            })
            
            return ai_response
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}"
    
    def _update_context_with_current_state(self, current_state: Dict[str, Any]):
        """Update AI context with current application state"""
        if 'current_dataset_stats' in current_state:
            self.context['current_dataset_stats'] = current_state['current_dataset_stats']
        
        if 'cleaning_history' in current_state:
            self.context['cleaning_history'] = current_state['cleaning_history']
        
        if 'inter_column_violations' in current_state:
            self.context['inter_column_violations'] = current_state['inter_column_violations']
        
        if 'weights_info' in current_state:
            self.context['weights_info'] = current_state['weights_info']
    
    def _build_system_prompt(self, column_specific: Optional[str] = None) -> str:
        """Build context-aware system prompt"""
        base_prompt = """You are an expert survey data analyst and statistician working for a statistical agency. 
        Your role is to provide column-specific, contextual guidance for data cleaning operations.

        CRITICAL REQUIREMENTS:
        - NEVER provide generic advice that applies to all columns
        - ALWAYS analyze each column individually based on its specific characteristics
        - Consider survey methodology and sampling design implications
        - Explain the statistical reasoning behind your recommendations
        - Compare different cleaning methods with specific pros and cons
        - Be educational - explain statistical concepts when relevant
        - Focus on maintaining data integrity for statistical agencies

        Your responses should be:
        - Specific to the column and data context provided
        - Methodologically sound for survey data
        - Clear and actionable
        - Educational when explaining statistical concepts
        """
        
        if self.context:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            safe_column_types = convert_numpy_types(self.context.get('column_types', {}))
            safe_dataset_info = convert_numpy_types({
                'dataset_shape': self.context.get('dataset_shape', 'Unknown'),
                'column_count': self.context.get('column_count', 'Unknown'),
                'missing_data_summary': self.context.get('missing_data_summary', {}),
                'current_dataset_stats': self.context.get('current_dataset_stats', {}),
                'cleaning_history': self.context.get('cleaning_history', {}),
                'weights_info': self.context.get('weights_info', {})
            })
            
            context_info = f"""
            CURRENT DATASET CONTEXT:
            - Dataset shape: {safe_dataset_info['dataset_shape']}
            - Total columns: {safe_dataset_info['column_count']}
            - Missing values total: {safe_dataset_info.get('current_dataset_stats', {}).get('missing_total', 'Unknown')}
            - Columns cleaned so far: {safe_dataset_info.get('current_dataset_stats', {}).get('columns_cleaned', 0)}
            - Column types: {json.dumps(safe_column_types, indent=2)}
            
            CLEANING PROGRESS:
            {json.dumps(safe_dataset_info.get('cleaning_history', {}), indent=2) if safe_dataset_info.get('cleaning_history') else 'No cleaning operations performed yet'}
            
            SURVEY WEIGHTS:
            {json.dumps(safe_dataset_info.get('weights_info', {}), indent=2) if safe_dataset_info.get('weights_info') else 'No design weights configured'}
            """
            base_prompt += context_info
        
        if column_specific and self.context.get('current_column_analysis'):
            column_analysis = self.context['current_column_analysis']
            
            # Convert numpy types for safe JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            safe_analysis = convert_numpy_types(column_analysis)
            
            column_prompt = f"""
            CURRENT COLUMN ANALYSIS FOR '{column_specific}':
            - Basic info: {json.dumps(safe_analysis.get('basic_info', {}), indent=2)}
            - Missing data: {json.dumps(safe_analysis.get('missing_analysis', {}), indent=2)}
            - Outliers: {json.dumps(safe_analysis.get('outlier_analysis', {}).get('summary', {}), indent=2)}
            - Data quality: {json.dumps(safe_analysis.get('data_quality', {}), indent=2)}
            - Rule violations: {json.dumps(safe_analysis.get('rule_violations', {}), indent=2)}
            
            Focus your response specifically on this column's characteristics and needs.
            Provide concrete, actionable advice based on these exact statistics and violations.
            """
            base_prompt += column_prompt
        
        return base_prompt
    
    def _build_user_message(self, question: str, column_specific: Optional[str] = None) -> str:
        """Build user message with context"""
        message = question
        
        if column_specific:
            message = f"For column '{column_specific}': {question}"
        
        return message
    
    def get_intelligent_cleaning_recommendation(self, column: str, analysis: Dict[str, Any], df: pd.DataFrame) -> str:
        """Intelligently detect issues and recommend specific cleaning methods with exact parameters and explanations"""
        if not self.client:
            return "AI assistant is not available."
        
        # Update context with current column analysis
        self.context['current_column_analysis'] = analysis
        
        # Detect specific issues in the column
        issues_detected = self._detect_column_issues(analysis, df[column])
        
        # Build comprehensive prompt with detected issues
        question = f"""You are analyzing column '{column}' in a survey dataset. I have detected the following specific issues:

{issues_detected}

Based on these SPECIFIC issues, provide CONCRETE cleaning recommendations:

1. **For Each Issue Detected**: 
   - Specify the EXACT cleaning method to use (e.g., "median_imputation", "iqr_removal", "winsorization")
   - Provide SPECIFIC parameters for the method (e.g., "n_neighbors=5", "percentile=0.95", "multiplier=1.5")
   - Explain WHY this method is best for THIS specific issue in THIS column
   
2. **Reasoning**: 
   - Explain how each method addresses the specific problem
   - Discuss what makes this method optimal given the data characteristics
   - Mention any trade-offs or considerations
   
3. **Order of Operations**: 
   - Specify which issue to address first and why
   - Explain dependencies between cleaning steps
   
4. **Expected Impact**:
   - Describe how each cleaning operation will affect the data
   - Mention any potential risks or side effects

Be SPECIFIC and ACTIONABLE. Don't give generic advice - every recommendation should be tailored to the exact issues detected in this column."""
        
        return self.ask_question(question, column)
    
    def _detect_column_issues(self, analysis: Dict[str, Any], series: pd.Series) -> str:
        """Detect specific issues in a column and format them for the AI"""
        issues = []
        
        # 1. Missing Data Issues
        basic_info = analysis.get('basic_info', {})
        missing_count = basic_info.get('missing_count', 0)
        missing_pct = basic_info.get('missing_percentage', 0)
        
        if missing_count > 0:
            pattern = analysis.get('missing_analysis', {}).get('pattern_type', 'Unknown')
            consecutive = analysis.get('missing_analysis', {}).get('max_consecutive', 0)
            
            issue_desc = f"• MISSING VALUES: {missing_count:,} missing ({missing_pct:.2f}%)"
            issue_desc += f"\n  - Pattern: {pattern}"
            if consecutive > 1:
                issue_desc += f"\n  - Max consecutive missing: {consecutive}"
            
            # Determine data type for appropriate imputation suggestion
            if pd.api.types.is_numeric_dtype(series):
                stats = series.describe()
                issue_desc += f"\n  - Data type: Numeric (mean={stats.get('mean', 'N/A'):.2f}, median={stats.get('50%', 'N/A'):.2f})"
                
                # Check for skewness
                skewness = abs(stats.get('mean', 0) - stats.get('50%', 0)) / (stats.get('std', 1) + 0.001)
                if skewness > 0.5:
                    issue_desc += f"\n  - Distribution: Skewed (consider median over mean)"
            else:
                mode_val = series.mode()
                mode_count = (series == mode_val[0]).sum() if len(mode_val) > 0 else 0
                issue_desc += f"\n  - Data type: Categorical (mode frequency={mode_count})"
            
            issues.append(issue_desc)
        
        # 2. Outlier Issues
        outlier_summary = analysis.get('outlier_analysis', {}).get('summary', {})
        outlier_count = outlier_summary.get('consensus_outliers', 0)
        
        if outlier_count > 0:
            severity = outlier_summary.get('severity', 'Unknown')
            outlier_pct = outlier_summary.get('consensus_percentage', 0)
            
            issue_desc = f"• OUTLIERS: {outlier_count} outliers detected ({outlier_pct:.2f}%)"
            issue_desc += f"\n  - Severity: {severity.title()}"
            
            # Get method-specific outlier counts
            method_results = analysis.get('outlier_analysis', {}).get('method_results', {})
            for method_name, result in method_results.items():
                method_count = result.get('outlier_count', 0)
                if method_count > 0:
                    issue_desc += f"\n  - {result.get('method', method_name)}: {method_count} outliers"
            
            # Add distribution info if numeric
            if pd.api.types.is_numeric_dtype(series):
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                issue_desc += f"\n  - IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
            
            issues.append(issue_desc)
        
        # 3. Data Quality Issues
        quality_data = analysis.get('data_quality', {})
        quality_score = quality_data.get('score', 100)
        
        if quality_score < 80:
            issue_desc = f"• DATA QUALITY: Score {quality_score}/100 (Grade: {quality_data.get('grade', 'Unknown')})"
            
            specific_issues = quality_data.get('issues', [])
            if specific_issues:
                issue_desc += "\n  - Specific problems:"
                for issue in specific_issues:
                    issue_desc += f"\n    * {issue}"
            
            issues.append(issue_desc)
        
        # 4. Distribution Issues
        dist_analysis = analysis.get('distribution_analysis', {})
        if dist_analysis:
            dist_type = dist_analysis.get('type', 'Unknown')
            
            if dist_type == 'numeric':
                skewness = dist_analysis.get('skewness', 0)
                kurtosis = dist_analysis.get('kurtosis', 0)
                
                if abs(skewness) > 1:
                    issue_desc = f"• DISTRIBUTION: Highly skewed (skewness={skewness:.2f})"
                    if skewness > 1:
                        issue_desc += "\n  - Right-skewed: Consider log transformation or winsorization"
                    else:
                        issue_desc += "\n  - Left-skewed: May need special handling"
                    issues.append(issue_desc)
                
                if abs(kurtosis) > 3:
                    issue_desc = f"• DISTRIBUTION: Heavy tails (kurtosis={kurtosis:.2f})"
                    issue_desc += "\n  - Many extreme values: Consider robust methods"
                    issues.append(issue_desc)
        
        # 5. Rule Violations
        violations = analysis.get('rule_violations', {})
        total_violations = violations.get('total_violations', 0)
        
        if total_violations > 0:
            issue_desc = f"• RULE VIOLATIONS: {total_violations} violations detected"
            
            violation_counts = violations.get('violation_counts', {})
            if violation_counts:
                for violation_type, count in violation_counts.items():
                    if count > 0:
                        issue_desc += f"\n  - {violation_type}: {count}"
            
            issues.append(issue_desc)
        
        # Format all issues
        if not issues:
            return "No significant issues detected in this column."
        
        return "\n\n".join(issues)
    
    def get_cleaning_recommendation(self, column: str, analysis: Dict[str, Any]) -> str:
        """Get AI recommendation for cleaning a specific column"""
        if not self.client:
            return "AI assistant is not available."
        
        # Update context with current column analysis
        self.context['current_column_analysis'] = analysis
        
        question = f"""Based on the analysis of column '{column}', what are the best cleaning strategies? 
        Please provide:
        1. Specific recommendations for this column's missing values, outliers, and data quality issues
        2. Pros and cons of each recommended method
        3. The reasoning behind your recommendations considering this column's characteristics
        4. Any survey methodology considerations
        5. Suggested order of operations for cleaning this column"""
        
        return self.ask_question(question, column)
    
    def compare_methods(self, column: str, method1: str, method2: str, analysis: Dict[str, Any]) -> str:
        """Compare two cleaning methods for a specific column"""
        self.context['current_column_analysis'] = analysis
        
        question = f"""For column '{column}', compare {method1} vs {method2}. 
        Consider:
        1. Which method is more appropriate for this column's specific characteristics?
        2. How would each method affect the data distribution and statistical properties?
        3. Survey methodology implications of each approach
        4. Computational complexity and practical considerations
        5. Your specific recommendation for this column and why"""
        
        return self.ask_question(question, column)
    
    def explain_concept(self, concept: str, context_column: Optional[str] = None) -> str:
        """Explain a statistical concept in the context of current data"""
        question = f"Please explain {concept} in the context of survey data cleaning"
        
        if context_column:
            question += f", specifically as it relates to column '{context_column}' in my current dataset"
        
        question += ". Please provide practical examples and when to use it."
        
        return self.ask_question(question, context_column)
    
    def assess_impact(self, column: str, proposed_method: str, analysis: Dict[str, Any]) -> str:
        """Assess the impact of a proposed cleaning method"""
        self.context['current_column_analysis'] = analysis
        
        question = f"""I'm planning to apply {proposed_method} to column '{column}'. 
        Please assess:
        1. How will this method affect the column's distribution and statistical properties?
        2. What are the potential impacts on survey estimates and analysis results?
        3. Are there any risks or unintended consequences I should consider?
        4. How might this affect relationships with other variables?
        5. Any alternative approaches that might be better for this specific column?"""
        
        return self.ask_question(question, column)
    
    def get_workflow_guidance(self, columns_analysis: Dict[str, Dict[str, Any]]) -> str:
        """Get guidance on cleaning workflow and column prioritization"""
        # Summarize all columns for context
        columns_summary = {}
        for col, analysis in columns_analysis.items():
            columns_summary[col] = {
                'missing_pct': analysis.get('missing_analysis', {}).get('percentage', 0),
                'quality_score': analysis.get('data_quality', {}).get('score', 100),
                'outlier_severity': analysis.get('outlier_analysis', {}).get('summary', {}).get('severity', 'low')
            }
        
        question = f"""Based on my dataset with columns: {json.dumps(columns_summary, indent=2)}
        
        Please provide workflow guidance:
        1. Which columns should I prioritize cleaning first and why?
        2. Are there any dependencies between columns that affect cleaning order?
        3. What's the recommended sequence of operations (missing values, outliers, etc.)?
        4. Any columns that require special attention or careful handling?
        5. How should I validate my cleaning results across all columns?"""
        
        return self.ask_question(question)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
    
    def export_conversation(self) -> str:
        """Export conversation history as JSON"""
        return json.dumps(self.conversation_history, indent=2)
