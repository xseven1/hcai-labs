import streamlit as st
from openai import OpenAI
import fitz

#Helper function to read PDFs
def read_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Show title and description.
st.title("📄 LAB 3")
st.write(
    "My Lab 3 question answering chatbot"
)

# Load OpenAI API key from secrets
openai_api_key = st.secrets.get("openai_api_key")

openAI_model = st.sidebar.selectbox("Select Model",
                                    ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

#Create OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets.get("openai_api_key")
    st.session_state.client = OpenAI(api_key=api_key)

if 'messages' not in st.session_state:
    st.session_state["messages"] = \
    [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    client = st.session_state.client
    stream = client.chat.completions.create(
        model = model_to_use,
        messages = st.session_state.messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role":"assistant", "content": response})