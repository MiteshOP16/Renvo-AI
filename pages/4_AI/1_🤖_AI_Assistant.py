import streamlit as st
import pandas as pd
import json
from modules.utils import initialize_session_state
from modules.ai_assistant import AIAssistant
from datetime import datetime

# Initialize session state
initialize_session_state()

def main():
    st.title("🤖 AI Data Cleaning Assistant")
    
    # Check if dataset is loaded
    if st.session_state.dataset is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
        st.stop()
    
    df = st.session_state.dataset
    assistant = AIAssistant()

    st.markdown("""
    Get expert guidance from your AI assistant. Ask questions about data cleaning methods, statistical concepts, 
    and get personalized recommendations based on your specific dataset and columns.
    """)

    # Initialize AI context
    dataset_info = {
        'shape': df.shape,
        'columns': len(df.columns),
        'missing_summary': df.isnull().sum().to_dict(),
        'column_types': st.session_state.column_types
    }
    
    assistant.set_context(dataset_info)
    # Initialize conversation history in session state
    if 'ai_conversation' not in st.session_state:
        st.session_state.ai_conversation = []

# AI Assistant Interface
    st.subheader("💬 Conversation")
    
    # Context selection
    context_cols = st.columns([2, 2])
    
    with context_cols[0]:
        context_mode = st.selectbox(
            "Conversation context:",
            options=['general', 'column_specific', 'workflow'],
            format_func=lambda x: {
                'general': '🌐 General Data Cleaning',
                'column_specific': '📊 Specific Column',
                'workflow': '🔄 Workflow Guidance'
            }[x]
        )
    
    with context_cols[1]:
        if context_mode == 'column_specific':
            context_column = st.selectbox(
                "Select column:",
                options=[''] + list(df.columns),
                help="Choose a column for context-specific guidance"
            )
        else:
            context_column = None
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions")
    
    quick_actions = st.columns(4)
    
    with quick_actions[0]:
        if st.button("📋 Dataset Overview", width='stretch'):
            question = "Please provide an overview of my dataset and highlight the main data quality issues I should address first."
            st.session_state.ai_conversation.append({
                'type': 'user',
                'message': question,
                'timestamp': datetime.now().isoformat(),
                'context': 'general'
            })
            
            with st.spinner("🤖 Analyzing your dataset..."):
                response = assistant.get_workflow_guidance(st.session_state.column_analysis)
                st.session_state.ai_conversation.append({
                    'type': 'assistant',
                    'message': response,
                    'timestamp': datetime.now().isoformat(),
                    'context': 'general'
                })
            st.rerun()
    
    with quick_actions[1]:
        if st.button("🧠 Smart Recommendations", width='stretch'):
            if context_column and context_column in st.session_state.column_analysis:
                analysis = st.session_state.column_analysis[context_column]
                question = f"🧠 Requesting intelligent analysis and specific cleaning recommendations for column '{context_column}'"
                
                st.session_state.ai_conversation.append({
                    'type': 'user',
                    'message': question,
                    'timestamp': datetime.now().isoformat(),
                    'context': context_column
                })
                
                with st.spinner("🤖 Analyzing issues and preparing specific recommendations..."):
                    response = assistant.get_intelligent_cleaning_recommendation(context_column, analysis, df)
                    st.session_state.ai_conversation.append({
                        'type': 'assistant',
                        'message': response,
                        'timestamp': datetime.now().isoformat(),
                        'context': context_column
                    })
                st.rerun()
            else:
                st.warning("Please select a column first")
    
    with quick_actions[2]:
        if st.button("📚 Explain Concept", width='stretch'):
            # Show concept selection
            concept = st.selectbox(
                "Select concept to explain:",
                options=[
                    'missing_data_patterns',
                    'outlier_detection',
                    'imputation_methods',
                    'data_normalization',
                    'survey_weights',
                    'sampling_bias'
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            question = f"Please explain {concept.replace('_', ' ')} in the context of survey data cleaning with practical examples."
            
            st.session_state.ai_conversation.append({
                'type': 'user',
                'message': question,
                'timestamp': datetime.now().isoformat(),
                'context': 'educational'
            })
            
            with st.spinner("🤖 Preparing explanation..."):
                response = assistant.explain_concept(concept.replace('_', ' '), context_column)
                st.session_state.ai_conversation.append({
                    'type': 'assistant',
                    'message': response,
                    'timestamp': datetime.now().isoformat(),
                    'context': 'educational'
                })
            st.rerun()
    
    with quick_actions[3]:
        if st.button("🔍 Impact Assessment", width='stretch'):
            if context_column:
                # Get recent cleaning history for this column
                column_history = st.session_state.cleaning_history.get(context_column, [])
                if column_history:
                    last_operation = column_history[-1]
                    method = last_operation.get('method_name', 'unknown method')
                    
                    question = f"Assess the impact of applying {method} to column '{context_column}' on my survey analysis and statistical estimates."
                    
                    st.session_state.ai_conversation.append({
                        'type': 'user',
                        'message': question,
                        'timestamp': datetime.now().isoformat(),
                        'context': context_column
                    })
                    
                    with st.spinner("🤖 Assessing impact..."):
                        analysis = st.session_state.column_analysis.get(context_column, {})
                        response = assistant.assess_impact(context_column, method, analysis)
                        st.session_state.ai_conversation.append({
                            'type': 'assistant',
                            'message': response,
                            'timestamp': datetime.now().isoformat(),
                            'context': context_column
                        })
                    st.rerun()
                else:
                    st.warning("No cleaning operations found for this column")
            else:
                st.warning("Please select a column first")

# Chat interface
    st.markdown("### 💬 Ask Your Question")
    
    # Question input
    question_input = st.text_area(
        "Type your question:",
        placeholder="e.g., 'Should I use median or mean imputation for the age column?' or 'How do I handle missing data that appears to be systematic?'",
        key="ai_question_input",
        value=st.session_state.get("ai_question_input", ""),
        height=100
    )
    # Apply pending suggested question safely
    if "pending_question" in st.session_state:
         st.session_state["ai_question_input"] = st.session_state.pop("pending_question")

    
    
    # Send button and options
    send_cols = st.columns([3, 1])
    
    with send_cols[0]:
        send_button = st.button("🚀 Send Question", type="primary", width='stretch')
    
    with send_cols[1]:
        if st.button("🗑️ Clear Chat", width='stretch'):
            st.session_state.ai_conversation = []
            assistant.clear_conversation_history()
            st.rerun()
    
    # Process question
    if send_button and question_input.strip():
        # Add user message to conversation
        st.session_state.ai_conversation.append({
            'type': 'user',
            'message': question_input.strip(),
            'timestamp': datetime.now().isoformat(),
            'context': context_column if context_mode == 'column_specific' else context_mode
        })
        
        # Get AI response
        with st.spinner("🤖 Thinking..."):
            try:
                # Prepare current data state
                current_data_state = {
                    'current_dataset_stats': {
                        'shape': df.shape,
                        'missing_total': df.isnull().sum().sum(),
                        'columns_cleaned': len(st.session_state.get('cleaning_history', {}))
                    },
                    'cleaning_history': st.session_state.get('cleaning_history', {}),
                    'weights_info': getattr(st.session_state.get('weights_manager', None), 'weights_metadata', {})
                }
                
                # Add inter-column violations if available
                if hasattr(st.session_state, 'inter_column_violations'):
                    current_data_state['inter_column_violations'] = st.session_state.inter_column_violations
                
                if context_mode == 'column_specific' and context_column:
                    # Set specific column context
                    if context_column in st.session_state.column_analysis:
                        column_analysis = st.session_state.column_analysis[context_column]
                        assistant.set_context(dataset_info, column_analysis)
                    
                    response = assistant.ask_question(question_input.strip(), context_column, current_data_state)
                elif context_mode == 'workflow':
                    response = assistant.get_workflow_guidance(st.session_state.column_analysis)
                else:
                    response = assistant.ask_question(question_input.strip(), None, current_data_state)
                
                # Add AI response to conversation
                st.session_state.ai_conversation.append({
                    'type': 'assistant',
                    'message': response,
                    'timestamp': datetime.now().isoformat(),
                    'context': context_column if context_mode == 'column_specific' else context_mode
                })
                
            except Exception as e:
                error_message = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question or check your API configuration."
                st.session_state.ai_conversation.append({
                    'type': 'assistant',
                    'message': error_message,
                    'timestamp': datetime.now().isoformat(),
                    'context': 'error'
                })
        
        # Refresh to show new conversation
        st.rerun()
    
    # Display conversation history
    st.markdown("### 📜 Conversation History")
    
    if st.session_state.ai_conversation:
        # Reverse to show most recent first
        for i, msg in enumerate(reversed(st.session_state.ai_conversation)):
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M:%S")
            context_info = f" ({msg.get('context', 'general')})" if msg.get('context') != 'general' else ""
            
            if msg['type'] == 'user':
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong>👤 You</strong> <small style="color: gray;">{timestamp}{context_info}</small><br>
                    {msg['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong>🤖 AI Assistant</strong> <small style="color: gray;">{timestamp}</small><br>
                    {msg['message']}
                </div>
                """, unsafe_allow_html=True)
            
            # Add action buttons for assistant messages
            if msg['type'] == 'assistant' and i < 3:  # Only for recent messages
                action_cols = st.columns(4)
                
                with action_cols[0]:
                    if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.ai_conversation)-i-1}"):
                        st.success("Thank you for the feedback!")
                
                with action_cols[1]:
                    if st.button("❓ Follow-up", key=f"followup_{len(st.session_state.ai_conversation)-i-1}"):
                        st.info("💬 Type your follow-up question in the text box above!")
                
                with action_cols[2]:
                    if st.button("🔄 Rephrase", key=f"rephrase_{len(st.session_state.ai_conversation)-i-1}"):
                        if len(st.session_state.ai_conversation) >= 2:
                            # Get the user question that led to this response
                            user_msg = st.session_state.ai_conversation[-(i+2)]['message'] if i < len(st.session_state.ai_conversation)-1 else ""
                            st.info(f"💬 Ask the AI: 'Please rephrase your answer to: {user_msg[:50]}...'")
                
                with action_cols[3]:
                    if st.button("📋 Copy", key=f"copy_{len(st.session_state.ai_conversation)-i-1}"):
                        st.code(msg['message'], language=None)
    else:
        st.info("💡 Start a conversation by asking a question or using one of the quick actions above!")
    
    # Pre-written question suggestions
    st.markdown("### 💡 Suggested Questions")
    
    suggestions = {
        "General Questions": [
            "What's the best approach to clean my dataset?",
            "How should I prioritize cleaning different columns?",
            "What are the risks of removing outliers in survey data?",
            "How do I validate my cleaning results?"
        ],
        "Column-Specific Questions": [
            "Which imputation method is best for this column?",
            "How should I handle outliers in this variable?",
            "What does this missing data pattern mean?",
            "Will this cleaning method bias my results?"
        ],
        "Statistical Questions": [
            "How do I preserve survey weights during cleaning?",
            "What's the difference between MCAR, MAR, and MNAR?",
            "When should I use multiple imputation?",
            "How do I assess the impact of data cleaning on my analysis?"
        ]
    }
    
    suggestion_tabs = st.tabs(list(suggestions.keys()))
    
    for tab, (category, questions) in zip(suggestion_tabs, suggestions.items()):
        with tab:
            for question in questions:
                if st.button(f"❓ {question}", key=f"suggest_{category}_{question[:20]}"):
                    st.session_state["pending_question"]= question
                    st.rerun()
    
    # Export conversation
    st.markdown("### 📤 Export Conversation")
    
    export_cols = st.columns([2, 1, 1])
    
    with export_cols[0]:
        if st.session_state.ai_conversation:
            conversation_text = ""
            for msg in st.session_state.ai_conversation:
                timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                sender = "You" if msg['type'] == 'user' else "AI Assistant"
                context = f" ({msg.get('context', 'general')})" if msg.get('context') != 'general' else ""
                conversation_text += f"{timestamp} - {sender}{context}:\n{msg['message']}\n\n"
            
            st.download_button(
                "💾 Download Conversation",
                data=conversation_text,
                file_name=f"ai_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with export_cols[1]:
        if st.button("📊 Generate Summary"):
            if st.session_state.ai_conversation:
                summary_question = "Please provide a summary of our conversation and the key recommendations discussed."
                
                st.session_state.ai_conversation.append({
                    'type': 'user',
                    'message': summary_question,
                    'timestamp': datetime.now().isoformat(),
                    'context': 'summary'
                })
                
                with st.spinner("🤖 Generating summary..."):
                    response = assistant.ask_question(summary_question)
                    st.session_state.ai_conversation.append({
                        'type': 'assistant',
                        'message': response,
                        'timestamp': datetime.now().isoformat(),
                        'context': 'summary'
                    })
                st.rerun()
    
    with export_cols[2]:
        if st.button("📋 Copy All", help="Copy conversation as JSON"):
            if st.session_state.ai_conversation:
                conversation_json = json.dumps(st.session_state.ai_conversation, indent=2)
                st.code(conversation_json, language='json')
    
    # Sidebar with context information
    with st.sidebar:
        st.markdown("### 🤖 AI Assistant Status")
        
        # Show API status
        if assistant.client:
            st.success("✅ AI Assistant Connected")
            st.write(f"**Model:** {assistant.model}")
        else:
            st.error("❌ AI Assistant Unavailable")
            st.write("Check your API configuration")
        
        # Conversation stats
        st.markdown("### 📊 Session Statistics")
        
        total_messages = len(st.session_state.ai_conversation)
        user_messages = len([m for m in st.session_state.ai_conversation if m['type'] == 'user'])
        ai_messages = total_messages - user_messages
        
        st.metric("Total Messages", total_messages)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Questions", user_messages)
        with col2:
            st.metric("AI Responses", ai_messages)
        
        # Context information
        st.markdown("### 🔍 Current Context")
        st.write(f"**Mode:** {context_mode.replace('_', ' ').title()}")
        if context_column:
            st.write(f"**Column:** {context_column}")
            
            # Show column info if analysis exists
            if context_column in st.session_state.column_analysis:
                analysis = st.session_state.column_analysis[context_column]
                quality_score = analysis['data_quality']['score']
                st.write(f"**Quality Score:** {quality_score}/100")
                
                missing_pct = analysis['basic_info']['missing_percentage']
                if missing_pct > 0:
                    st.write(f"**Missing:** {missing_pct:.1f}%")
        
        # Dataset context
        st.markdown("### 📋 Dataset Context")
        st.write(f"**Rows:** {df.shape[0]:,}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing:** {df.isnull().sum().sum():,}")
        
        analyzed_columns = len(st.session_state.column_analysis)
        st.write(f"**Analyzed:** {analyzed_columns}/{len(df.columns)}")
        
        cleaned_columns = len(st.session_state.cleaning_history)
        st.write(f"**Cleaned:** {cleaned_columns}/{len(df.columns)}")
        
        # Quick navigation
        st.markdown("### ⚡ Quick Navigation")
        
        if st.button("🔍 Column Analysis", width='stretch'):
            st.switch_page("pages/2_Column_Analysis.py")
        
        if st.button("🧹 Cleaning Wizard", width='stretch'):
            st.switch_page("pages/3_Cleaning_Wizard.py")
        
        if st.button("📊 Generate Reports", width='stretch'):
            st.switch_page("pages/7_Reports.py")
    
    # Footer with tips
    st.markdown("---")
    st.markdown("""
    **💡 Tips for Better AI Interactions:**
    - Be specific about your data and goals
    - Mention the column name and its characteristics
    - Ask follow-up questions to dive deeper
    - Use context modes for focused guidance
    - Request explanations of statistical concepts
    """)

if __name__ == "__main__":
    main()

