import streamlit as st
from google.cloud import aiplatform
from vertexai.preview.language_models import (
    TextGenerationModel,
    ChatModel,
    CodeGenerationModel,
    InputOutputTextPair,
)

# 🔹 Initialize Vertex AI
PROJECT_ID = "your-project-id"   # 🔁 Replace this
LOCATION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# 🔹 Load Models
text_model = TextGenerationModel.from_pretrained("text-bison@001")
chat_model = ChatModel.from_pretrained("chat-bison@001")
code_model = CodeGenerationModel.from_pretrained("code-bison@001")

# 🔹 Streamlit UI
st.set_page_config(page_title="Vertex AI GenAI App", layout="centered")
st.title("💡 Vertex AI GenAI Streamlit App")

tab1, tab2, tab3 = st.tabs(["📄 Text Generator", "💬 Chat Assistant", "💻 Code Generator"])

# --- Tab 1: Text Generation ---
with tab1:
    st.header("Text Generator (text-bison)")
    prompt = st.text_area("Enter a prompt for text generation:")
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    if st.button("Generate Text"):
        if prompt.strip():
            with st.spinner("Generating..."):
                response = text_model.predict(prompt, temperature=temp)
                st.success("Generated Text:")
                st.write(response.text)
        else:
            st.warning("Please enter a prompt.")

# --- Tab 2: Chat Assistant ---
with tab2:
    st.header("Chat Assistant (chat-bison)")
    context = st.text_input("Context (optional):", "You are a helpful assistant.")
    user_msg = st.text_area("Ask a question:")
    if st.button("Ask Chatbot"):
        if user_msg.strip():
            chat = chat_model.start_chat(context=context)
            response = chat.send_message(user_msg)
            st.success("Chatbot Response:")
            st.write(response.text)
        else:
            st.warning("Please enter a message.")

# --- Tab 3: Code Generator ---
with tab3:
    st.header("Code Generator (code-bison)")
    code_prompt = st.text_area("Describe the code you want:")
    temp_code = st.slider("Code Temperature", 0.0, 1.0, 0.3)
    if st.button("Generate Code"):
        if code_prompt.strip():
            with st.spinner("Generating code..."):
                response = code_model.predict(prefix=code_prompt, temperature=temp_code)
                st.success("Generated Code:")
                st.code(response.text, language="python")
        else:
            st.warning("Please enter a code prompt.")
