import streamlit as st
import requests
import json
import os
import time # Added for sleep function in retry logic

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
    Calls the Gemini API to generate an optimal prompt with detailed error reporting,
    including timeout and retry logic, and a final check for empty content.
    """
    if not basic_prompt:
        st.error("Please enter a basic prompt.")
        return "Please enter a basic prompt."
        
    if not api_key:
        st.error("API Key not configured. Please set the GEMINI_API_KEY secret.")
        return "ERROR: API Key not configured. Check the status box above."

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

    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30  # Set a generous but strict timeout

    for attempt in range(MAX_RETRIES):
        try:
            st.info(f"Attempt {attempt + 1} of {MAX_RETRIES}: Sending request...")
            response = requests.post(
                full_url, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=TIMEOUT_SECONDS # Apply the strict timeout
            )
            
            # --- Check for API HTTP Errors (4xx or 5xx) ---
            if not response.ok:
                st.error(f"API Request Failed: HTTP Status {response.status_code}")
                try:
                    error_details = response.json()
                except json.JSONDecodeError:
                    error_details = {"message": response.text}
                    
                st.code(json.dumps(error_details, indent=2), language="json")
                
                # If it's a 429 (Rate Limit), wait and retry
                if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** (attempt + 1)
                    st.warning(f"Rate limited (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue # Go to the next attempt
                else:
                    # For other non-retryable errors (400, 403, 500) or last attempt failure, break out
                    return f"API ERROR ({response.status_code}): {error_details.get('error', {}).get('message', 'Check console for details.')}"


            # --- Successful Response Processing (200 OK) ---
            result = response.json()
            
            # Safely extract the generated text
            candidate = result.get('candidates', [{}])[0]
            optimized_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', None)

            # --- FINAL CHECK FOR EMPTY/NULL CONTENT ---
            if optimized_text is None:
                st.error("DEBUG ERROR: Content extraction failed (optimized_text is None). Check the raw JSON below.")
                st.code(json.dumps(result, indent=2), language="json")
                return "CRITICAL DEBUG: Content extraction failed. See raw JSON above."
                
            if optimized_text.strip() == "":
                st.error("WARNING: API returned success (200), but the generated content was an empty string. This may indicate content violation, safety block, or an ambiguous prompt.")
                st.code(json.dumps(result, indent=2), language="json")
                return "WARNING: Empty response received. Try a different prompt or check the raw JSON output above for safety blocks."
                
            # If we reach here, the attempt was successful and we have text
            return optimized_text
            
        except requests.exceptions.Timeout:
            st.error(f"Attempt {attempt + 1}/{MAX_RETRIES} timed out after {TIMEOUT_SECONDS} seconds.")
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** (attempt + 1)
                st.warning(f"Connection timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue # Go to the next attempt
            else:
                return f"CRITICAL ERROR: Request timed out after {MAX_RETRIES} attempts. Check your network connection and API usage limits."
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network/Connection Error: {e}")
            return f"CRITICAL ERROR: Network connection failed. Details: {e}"
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return f"CRITICAL ERROR: Unexpected exception occurred. Details: {e}"

    return "ERROR: All optimization attempts failed due to recurring issues."


# --- Streamlit UI ---

st.set_page_config(
    page_title="AI Optimal Prompt Generator",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("💡 AI Optimal Prompt Generator")
st.markdown("Transform your simple idea into a powerful, effective prompt for any Large Language Model.")

# Get API Key and display status
api_key = get_api_key()
if api_key:
    st.success("✅ Gemini API Key Loaded. Ready to run.")
else:
    st.warning("⚠️ Gemini API Key NOT FOUND. Please set your `GEMINI_API_KEY` environment variable or secret file to enable generation.")
    
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
    # Clear previous messages
    st.session_state.optimized_prompt = ""
    with st.spinner('Generating and optimizing the prompt...'):
        result = optimize_prompt(basic_prompt, api_key)
        st.session_state.optimized_prompt = result
        # Force a rerun to update the text area immediately
        st.rerun()
        
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
if optimized_display.strip() and not optimized_display.startswith("ERROR") and not optimized_display.startswith("WARNING") and not optimized_display.startswith("CRITICAL DEBUG") and not optimized_display.startswith("Please enter"):
    if st.button("📋 Copy Optimized Prompt", use_container_width=False):
        st.code(optimized_display, language='text')
        st.success("Copied to clipboard! (Note: Streamlit's clipboard functionality varies; manually copying from the text box is often more reliable.)")

st.markdown("""
---
### Next Steps for Diagnosis
**CRITICAL:** Please run the app one more time with this code. If the output box is still blank, try a different browser or clear your cache, as the issue would be a severe rendering bug outside the Python logic.

If you now see a message, please report it:
* **"WARNING: Empty response received..."**: The API call succeeded, but the model blocked the content or returned nothing. Try a different prompt (e.g., "Describe a perfect vacation").
* **"CRITICAL DEBUG: Content extraction failed..."**: This means the API response JSON structure was not what the code expected.
* **"CRITICAL ERROR: Request timed out..."**: This confirms a severe network or quota issue.
""")
