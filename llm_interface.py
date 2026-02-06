import os
from typing import Dict, List
import google.generativeai as genai
# Try importing the Gemini library
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class LLMReasoningEngine:
    """
    Task 2: Interactive Reasoning Engine (Powered by Gemini)
    Takes the mathematical analysis from Task 1 and generates 
    context-aware natural language responses.
    """
    def __init__(self, api_key=None):
        self.mock_mode = False
        
        if api_key and HAS_GEMINI:
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            print("⚠️ No Gemini API Key provided (or google-generativeai missing). Running in MOCK MODE.")
            self.mock_mode = True

    def summarize_context(self, analysis_result: Dict) -> str:
        """
        Converts the complex Python/Numpy analysis object into a 
        text-based summary that the LLM can read.
        """
        conversation = analysis_result['conversation']
        states = analysis_result['predicted_states']
        root_cause = analysis_result['root_cause_analysis']
        
        # 1. Create the Call Transcript View
        transcript_text = ""
        for i, turn in enumerate(conversation):
            state_label = f"S{states[i]}" # Neutral, Anger, etc.
            # Highlight the specific root cause turn
            is_cause = " [ROOT CAUSE TRIGGER]" if i == root_cause['root_cause_turn'] else ""
            transcript_text += f"Turn {i} ({turn['speaker']}): \"{turn['text']}\" [State: {state_label}]{is_cause}\n"

        # 2. Summarize the Math (The "Grounded" Facts)
        math_summary = f"""
        - Did Escalation Occur?: {root_cause['escalated']}
        - Root Cause Identified At: Turn {root_cause['root_cause_turn']}
        - Trigger Phrase: "{conversation[root_cause['root_cause_turn']]['text']}"
        - Risk Spike Confidence: {max(root_cause['delta_sequence']):.2f}
        """
        
        return f"TRANSCRIPT:\n{transcript_text}\n\nMATHEMATICAL ANALYSIS:{math_summary}"

    def ask(self, user_query: str, analysis_context: Dict, chat_history: List) -> str:
        """
        Generates the response using Gemini.
        """
        # 1. Prepare the Prompt Context
        context_str = self.summarize_context(analysis_context)
        
        # 2. Construct the Conversation History string for the Prompt
        # (This gives Gemini the context of the previous questions)
        history_str = ""
        for q, a in chat_history[-3:]: # Keep last 3 turns
            history_str += f"User: {q}\nAI: {a}\n"

        # 3. Final Prompt Construction
        full_prompt = f"""
        SYSTEM: You are an expert Root Cause Analysis AI for Customer Service.
        You have access to a deep-learning analysis of a specific call.
        
        DATA SOURCE:
        {context_str}
        
        INSTRUCTIONS:
        1. Answer the user's question based strictly on the TRANSCRIPT and MATHEMATICAL ANALYSIS provided.
        2. If asked "Why did they get angry?", cite the Trigger Phrase and the Risk Spike.
        3. Be concise and professional.
        
        CHAT HISTORY:
        {history_str}
        
        User: {user_query}
        AI:
        """

        # 4. Call Gemini (or Mock)
        if self.mock_mode:
            return self._mock_response(user_query, analysis_context)
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"

    def _mock_response(self, query: str, context: Dict) -> str:
        """Simple rule-based responses for testing without API Key"""
        query = query.lower()
        rc = context['root_cause_analysis']
        
        if "why" in query or "cause" in query:
            return f"Based on the analysis, the root cause was Turn {rc['root_cause_turn']}. The agent said something that caused a risk spike of {max(rc['delta_sequence']):.2f}."
        elif "transcript" in query:
            return f"I have analyzed the full transcript. It contains {len(context['conversation'])} turns."
        else:
            return "This is a Mock Response (No API Key). I received your query: " + query
    
    # ... (Keep existing __init__, summarize_context, and ask methods) ...

    def generate_executive_summary(self, analysis_context: Dict, chat_history: List) -> str:
        """
        Generates a formal executive summary for the downloaded report.
        """
        context_str = self.summarize_context(analysis_context)
        
        history_str = ""
        for q, a in chat_history:
            history_str += f"Analyst: {q}\nAI: {a}\n"

        prompt = f"""
        SYSTEM: You are writing a Formal Root Cause Analysis Report for a Stakeholder.
        
        DATA:
        {context_str}
        
        INVESTIGATION LOG (Q&A with Analyst):
        {history_str}
        
        INSTRUCTIONS:
        Write a 1-paragraph Executive Summary of this incident.
        1. State clearly why the call escalated (The Root Cause).
        2. Mention the specific trigger phrase used by the agent.
        3. Summarize any key insights the analyst discovered during the Q&A session.
        4. Use professional, business-standard language.
        """
        
        if self.mock_mode:
            return "This is a Mock Executive Summary. The call escalated due to a specific trigger phrase detected at Turn X."
            
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"