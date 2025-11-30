import streamlit as st
import requests
import json
import os

# --- Configuration ---
# The model endpoint for the Gemini API
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'
API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent'

# System instruction to define the AI's role as a prompt optimizer
SYSTEM_PROMPT = """You are the 'Optimal Prompt Generator' AI. Your sole function is to take a short, basic user query and expand it into a comprehensive, highly effective prompt suitable for a large language model. The goal is to maximize the quality and relevance of the LLM's final output.

To achieve this, your optimized prompt MUST incorporate the following elements:
1.  **Persona:** Assign a specific, credible, and expert persona relevant to the task (e.g., 'Act as a senior software architect,' 'You are a professional chef').
2.  **Context:** Provide necessary background information or scope definition.
3.  **Constraints:** Define limitations on length, complexity, or tone (e.g., 'Limit the response to 300 words,' 'The tone must be academic').
4.  **Output Format:** Specify the desired structure (e.g., 'Use Markdown lists,' 'Output only a JSON object following schema X,' 'Write a five-paragraph essay').
5.  **Task Refinement:** Clearly state the refined goal based on the user's input.

Crucially, your entire response must ONLY be the final, optimized prompt text. Do not include any conversational phrases, titles, or explanations outside of the optimized prompt itself.
"""

# --- API Key Handling ---
def get_api_key():
    """Fetches the API key from Streamlit secrets or environment variables."""
    # Check Streamlit secrets first (best practice for deployment)
    api_key = st.secrets.get("GEMINI_API_KEY")

    # Fallback to environment variable (useful for local development)
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    return api_key

# --- Core Logic ---

def optimize_prompt(basic_prompt, api_key):
    """
    Calls the Gemini API to generate an optimal prompt.
    """
    if not basic_prompt:
        st.error("Please enter a basic prompt.")
        return "Please enter a basic prompt."
        
    if not api_key:
        st.error("API Key not configured. Please set the GEMINI_API_KEY secret.")
        return "API Key not configured."

    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "contents": [{"parts": [{"text": basic_prompt}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.8},
    }
    
    # Add API key to the URL parameters
    full_url = f"{API_URL}?key={api_key}"

    try:
        response = requests.post(full_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raises an HTTPError if the status is 4xx or 5xx
        
        result = response.json()
        
        # Safely extract the generated text
        candidate = result.get('candidates', [{}])[0]
        optimized_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not extract text from API response.')
        
        return optimized_text
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the Gemini API: {e}")
        return f"API Error: {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return f"Unexpected Error: {e}"


# --- Streamlit UI ---

st.set_page_config(
    page_title="AI Optimal Prompt Generator",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("💡 AI Optimal Prompt Generator")
st.markdown("Transform your simple idea into a powerful, effective prompt for any Large Language Model.")

# Get API Key
api_key = get_api_key()
if not api_key:
    st.warning("Please set your `GEMINI_API_KEY` in Streamlit secrets or environment variables to run the optimizer.")
    
# Input Area
st.subheader("1. Enter Your Basic Prompt Idea")
basic_prompt = st.text_area(
    "Basic Prompt", 
    placeholder="Example: Write a story about a dragon who loves to read.", 
    height=150,
    label_visibility="collapsed"
)

# Initialize output state
if 'optimized_prompt' not in st.session_state:
    st.session_state.optimized_prompt = ""

# Button to trigger optimization
if st.button("🚀 Generate Optimal Prompt", type="primary", use_container_width=True, disabled=not api_key):
    with st.spinner('Generating and optimizing the prompt...'):
        result = optimize_prompt(basic_prompt, api_key)
        st.session_state.optimized_prompt = result
        
# Output Area
st.subheader("2. Optimized Prompt Result")
optimized_display = st.text_area(
    "Optimized Prompt", 
    value=st.session_state.optimized_prompt, 
    height=300, 
    key="optimized_display",
    label_visibility="collapsed"
)

# Copy button using st.empty for better placement
if optimized_display.strip():
    if st.button("📋 Copy Optimized Prompt", use_container_width=False):
        st.code(optimized_display, language='text')
        st.success("Copied to clipboard! (Note: Streamlit's clipboard functionality varies; manually copying from the text box is often more reliable.)")

st.markdown("""
---
### How to Deploy
1.  **Save** this code as `app.py`.
2.  **Create** the `requirements.txt` file below.
3.  **Set the API Key:** Deploy the app and set a Streamlit secret named `GEMINI_API_KEY` with your actual Google AI Studio API key.
4.  **Deploy** to Streamlit Cloud!
""")
