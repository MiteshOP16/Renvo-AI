"""
AI-powered hypothesis test suggestion using Groq
"""
import os
import json
from typing import Dict, List, Any

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

class AIHypothesisHelper:
    """AI assistant for suggesting appropriate hypothesis tests"""
    
    def __init__(self):
        self.client = None
        self.api_key = os.getenv('GROQ_API_KEY')
        if GROQ_AVAILABLE and self.api_key and Groq:
            self.client = Groq(api_key=self.api_key)
    
    def suggest_test(self, user_prompt: str, data_context: Dict[str, Any], test_metadata: str) -> Dict[str, Any]:
        """
        Suggest appropriate statistical test based on user's natural language description
        
        Args:
            user_prompt: User's description of what they want to test
            data_context: Dict with 'numeric_columns', 'categorical_columns', 'sample_size', etc.
            test_metadata: Formatted string of all available tests
            
        Returns:
            Dict with suggestions including test_id, category, rationale, confidence
        """
        if not self.client:
            return {
                'success': False,
                'error': 'Groq API key not configured. Please add GROQ_API_KEY to environment variables.',
                'fallback': True
            }
        
        # Build system prompt
        system_prompt = f"""You are a statistical analysis expert helping users choose the right hypothesis test.

Available statistical tests:
{test_metadata}

Your task is to analyze the user's request and recommend the most appropriate statistical test(s).

Data context available:
- Numeric columns: {data_context.get('numeric_columns', [])}
- Categorical columns: {data_context.get('categorical_columns', [])}
- Sample size: {data_context.get('sample_size', 'unknown')}

Respond in JSON format with:
{{
  "primary_recommendation": {{
    "test_id": "test_identifier",
    "test_name": "Human-readable test name",
    "category": "parametric or non_parametric",
    "rationale": "1-2 sentence explanation why this test is appropriate",
    "confidence": 0.0-1.0
  }},
  "alternative_recommendations": [
    {{
      "test_id": "test_identifier",
      "test_name": "Human-readable test name", 
      "category": "parametric or non_parametric",
      "rationale": "Brief explanation"
    }}
  ],
  "warnings": ["Any caveats or assumptions to check"],
  "suggested_columns": {{
    "numeric": ["column names if applicable"],
    "categorical": ["column names if applicable"]
  }}
}}

Be concise, beginner-friendly, and always explain technical terms simply."""

        user_message = f"""User wants to perform: {user_prompt}

Available data columns:
- Numeric: {', '.join(data_context.get('numeric_columns', [])) or 'None'}
- Categorical: {', '.join(data_context.get('categorical_columns', [])) or 'None'}
- Total observations: {data_context.get('sample_size', 'Unknown')}

Please recommend the most appropriate statistical test(s)."""

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            suggestions = json.loads(content)
            
            return {
                'success': True,
                'suggestions': suggestions,
                'fallback': False
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Failed to parse AI response: {str(e)}',
                'fallback': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'AI suggestion failed: {str(e)}',
                'fallback': True
            }
    
    def explain_test(self, test_name: str, user_level: str = 'beginner') -> str:
        """
        Get a simple explanation of a statistical test
        
        Args:
            test_name: Name of the test
            user_level: 'beginner', 'intermediate', or 'advanced'
        """
        if not self.client:
            return "Explanation unavailable - Groq API not configured."
        
        prompt = f"""Explain the {test_name} statistical test in {user_level}-friendly language.

Include:
1. What it tests (1 sentence)
2. When to use it (1-2 sentences)
3. Key assumptions (bullet points)
4. How to interpret results (1-2 sentences)

Keep it concise and avoid jargon."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
