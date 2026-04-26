import streamlit as st
import requests
import base64
from io import BytesIO

# --- Configuration ---
FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG AI Assistant", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium UI Aesthetics ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-primary: #020617;
    --bg-secondary: #0f172a;
    --accent-primary: #8b5cf6;
    --accent-secondary: #ec4899;
    --accent-gradient: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --glass-bg: rgba(30, 41, 59, 0.45);
    --glass-border: rgba(255, 255, 255, 0.08);
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top right, #1e1b4b 0%, #020617 60%) !important;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--glass-border) !important;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    background: transparent !important;
    padding-bottom: 20px;
}
div[data-testid="stChatInput"] textarea {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
    padding: 14px 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease;
    font-size: 1.05rem !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-secondary) !important;
    box-shadow: 0 0 20px rgba(236, 72, 153, 0.4) !important;
    outline: none !important;
}

/* Chat Bubbles */
.stChatMessage {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(12px) !important;
    animation: fadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Typography & Links */
h1 {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem !important;
    margin-bottom: 2rem !important;
    text-align: center;
    letter-spacing: -0.02em;
}

/* Image Elements & Cards */
[data-testid="stImage"] {
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 8px;
}
[data-testid="stImage"]:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 30px rgba(139, 92, 246, 0.3);
}
[data-testid="stImage"] img {
    max-width: 100%;
    max-height: 400px;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    background-color: rgba(0,0,0,0.2);
}

.image-evidence-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-secondary);
    margin: 16px 0 12px 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.image-meta {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 4px;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    padding: 6px 10px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    border: 1px solid var(--glass-border);
}

/* Sidebar Elements */
div[data-testid="stForm"] {
    background-color: rgba(15,23,42,0.4) !important;
    border: 1px dashed rgba(139, 92, 246, 0.5) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

button[kind="primary"] {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 8px 16px !important;
    transition: opacity 0.2s ease, transform 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(236, 72, 153, 0.4) !important;
}

/* Loading Spinners */
.stSpinner > div > div {
    border-color: var(--accent-primary) !important;
    border-bottom-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# --- State Management ---
# Each message is: {"role": str, "content": str, "images": list}
# images is a list of image dicts from the API — empty list for user messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Helpers ---

def render_images(images: list):
    if not images:
        return

    st.markdown('<p class="image-evidence-label">📎 Visual Evidence</p>', unsafe_allow_html=True)

    cols_per_row = min(len(images), 3)
    cols = st.columns(cols_per_row)

    for idx, img in enumerate(images):
        with cols[idx % cols_per_row]:
            try:
                image_bytes = base64.b64decode(img["data"])
                # clamp_to_column=True ensures it never overflows its column
                st.image(BytesIO(image_bytes), use_container_width=True)
                st.markdown(
                    f'<p class="image-meta">'
                    f'📄 {img["source_file"]} &nbsp;·&nbsp; '
                    f'Page {img["page_num"]} &nbsp;·&nbsp; '
                    f'Score {img["relevance_score"]:.2f}'
                    f'</p>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.caption(f"Could not render image from {img.get('source_file', 'unknown')}")

def render_message(message: dict):
    """Renders a single message bubble with its text and any attached images."""
    with st.chat_message(message["role"]):
        # Streamlit's markdown parser requires $$ for display math and $ for inline math.
        # Most LLMs return LaTeX delimiters like \[ \] and \( \).
        text = message["content"]
        text = text.replace("\\[", "$$").replace("\\]", "$$")
        text = text.replace("\\(", "$").replace("\\)", "$")
        
        st.markdown(text)
        if message.get("images"):
            render_images(message["images"])


# --- Sidebar: File Ingestion ---
with st.sidebar:
    st.header("⚙️ Inference Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_new_tokens = st.slider("Max New Tokens", min_value=100, max_value=4096, value=500, step=100)

    st.divider()

    st.header("📂 Knowledge Base")
    st.markdown("Upload files here. The system accepts PDFs, Images, and CSVs.")

    with st.form("ingest_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Drop files to index",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Ingest Files", use_container_width=True, type="primary")

    if submitted:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Indexing contents..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), f.type))
                        for f in uploaded_files
                    ]
                    response = requests.post(f"{FASTAPI_URL}/ingest", files=files_payload)

                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Successfully processed {len(res_data.get('files_indexed', []))} files!")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the FastAPI backend. Is it running on port 8000?")


# --- Main Chat Area ---
st.title("Multimodal AI Assistant")

# Replay full conversation history on every rerun
for message in st.session_state.messages:
    render_message(message)

# Handle new user input
if prompt := st.chat_input("Ask something about your documents..."):

    user_message = {"role": "user", "content": prompt, "images": []}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*(Thinking...)*")

        try:
            res = requests.post(f"{FASTAPI_URL}/chat", params={
                "query": prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens
            })

            if res.status_code == 200:
                data = res.json()
                answer = data.get("response", "No response content.")
                images = data.get("images", [])

                placeholder.markdown(answer)

                # Render images right inside the same assistant bubble
                if images:
                    render_images(images)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "images": images,
                })
            else:
                placeholder.empty()
                st.error(f"Error {res.status_code}: {res.text}")

        except requests.exceptions.ConnectionError:
            placeholder.empty()
            st.error("Failed to connect to backend server.")