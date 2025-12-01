import streamlit as st
import google.generativeai as genai
import os

# --- Page Config ---
st.set_page_config(
    page_title="AI Prompt Optimizer",
    page_icon="✨",
    layout="centered"
)

# --- Header ---
st.title("✨ AI Prompt Optimizer")
st.markdown(
    """
    Transform basic ideas into **expert-level prompts** using Google's Gemini AI. 
    Enter your rough idea below, and we'll engineer it for optimal results.
    """
)

# --- Sidebar: API Key Configuration ---
with st.sidebar:
    st.header("Settings")
    
    # Try to get key from Streamlit secrets (for deployment)
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
        st.success("API Key loaded from Secrets")
    else:
        # Fallback to user input (for local testing without secrets.toml)
        api_key = st.text_input("Enter Gemini API Key", type="password")
        st.caption("Get your key at [Google AI Studio](https://aistudio.google.com/)")

# --- The "Meta-Prompt" Logic ---
# This is the instruction sent to Gemini to tell it how to fix the user's prompt
SYSTEM_INSTRUCTION = """
You are a World-Class Prompt Engineer and AI Optimization Specialist. 
Your goal is to take a raw, basic user input and transform it into a highly optimized, structured prompt for an LLM (like GPT-4 or Gemini).

Follow these steps to optimize the prompt:
1. **Analyze:** Understand the user's core intent.
2. **Structure:** Apply a framework (e.g., CO-STAR: Context, Objective, Style, Tone, Audience, Response).
3. **Refine:** Add specific constraints, output formats, and examples if helpful.
4. **Clarity:** Ensure the prompt is unambiguous.

Output ONLY the optimized prompt. Do not include introductory text like "Here is your optimized prompt." Just give the prompt itself.
"""

def get_optimized_prompt(user_text, key):
    try:
        genai.configure(api_key=key)
        # Using gemini-1.5-flash for speed and efficiency, or switch to 'gemini-1.5-pro' for complex reasoning
        model = genai.GenerativeModel('gemini-flash-latest', system_instruction=SYSTEM_INSTRUCTION)
        
        response = model.generate_content(user_text)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- Main Interface ---

basic_prompt = st.text_area(
    "Enter your basic prompt idea:", 
    height=150, 
    placeholder="e.g., Write a blog post about coffee."
)

if st.button("Optimize Prompt ✨", type="primary"):
    if not basic_prompt:
        st.warning("Please enter a prompt to optimize.")
    elif not api_key:
        st.error("Please provide a Gemini API Key in the sidebar.")
    else:
        with st.spinner("Engineering the perfect prompt..."):
            optimized_result = get_optimized_prompt(basic_prompt, api_key)
            
            if "Error:" in optimized_result:
                st.error(optimized_result)
            else:
                st.subheader("🚀 Your Optimized Prompt")
                st.code(optimized_result, language="markdown")
                
                # Copy button logic is handled natively by the Streamlit code block hover menu
                st.caption("Copy the code block above to use in your AI chats.")
