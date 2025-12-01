import streamlit as st
import google.generativeai as genai
import os

# --- Page Config ---
st.set_page_config(
    page_title="AI Prompt Optimiser",
    page_icon="✨",
    layout="centered"
)

# --- Header ---
st.title("✨ AI Prompt Optimiser")
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
You are the 'Optimal Prompt Generator' AI. Your sole function is to take a short, basic user query and expand it into a comprehensive, highly effective prompt suitable for a large language model. The goal is to maximize the quality and relevance of the LLM's final output.

To achieve this, your optimized prompt MUST incorporate the following elements:
1.  **Persona:** Assign a specific, credible, and expert persona relevant to the task (e.g., 'Act as a senior software architect,' 'You are a professional chef').
2.  **Context:** Provide necessary background information or scope definition.
3.  **Constraints:** Define limitations on length, complexity, or tone (e.g., 'Limit the response to 300 words,' 'The tone must be academic').
4.  **Output Format:** Specify the desired structure (e.g., 'Use Markdown lists,' 'Output only a JSON object following schema X,' 'Write a five-paragraph essay').
5.  **Task Refinement:** Clearly state the refined goal based on the user's input.

Crucially, your entire response must ONLY be the final, optimized prompt text. Do not include any conversational phrases, titles, or explanations outside of the optimized prompt itself.
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
