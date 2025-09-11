import streamlit as st
from openai import OpenAI
import fitz

# Conversation buffer helper
def keep_last_n_messages(messages, n=2):
    """Keep only the last n user+assistant exchanges."""
    system_msgs = [m for m in messages if m["role"] == "assistant" and "How can I help you?" in m["content"]]
    other_msgs = [m for m in messages if m not in system_msgs]
    return system_msgs + other_msgs[-2*n:]

# Show title and description
st.title("📄 LAB 3")
st.write("My Lab 3 question answering chatbot")

# Load OpenAI API key from secrets
openai_api_key = st.secrets.get("openai_api_key")

openAI_model = st.sidebar.selectbox("Select Model", ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# Create OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets.get("openai_api_key")
    st.session_state.client = OpenAI(api_key=api_key)

if 'messages' not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display conversation
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Apply conversation buffer (keep last 2 exchanges)
    messages_for_llm = keep_last_n_messages(st.session_state.messages, n=2)

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_for_llm,
        stream=True
    )

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response += delta
            response_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
