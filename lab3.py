import streamlit as st
from openai import OpenAI


# Conversation buffer helper (last 2 Q&A pairs)
def keep_last_n_messages(messages, n=2):
    system_msgs = [m for m in messages if m["role"] == "assistant" and "How can I help you?" in m["content"]]
    other_msgs = [m for m in messages if m not in system_msgs]
    return system_msgs + other_msgs[-2*n:]


# Show title
st.title("📄 LAB 3 – Chatbot")


# Load OpenAI API key
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

# Init session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "What question can I help with?"}
    ]
if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False


# Show past messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])


# Chat loop
if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    # Case 1: User is in "more info" loop
    if st.session_state.awaiting_more_info:
        if prompt.strip().lower() in ["yes", "y", "sure", "ok"]:
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=keep_last_n_messages(
                    st.session_state.messages + [
                        {"role": "assistant", "content": "Give a longer, more detailed explanation that a 10-year-old can understand."}
                    ],
                    n=2
                ),
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

            # Ask again
            followup = "Would you like more information?"
            with st.chat_message("assistant"):
                st.write(followup)
            st.session_state.messages.append({"role": "assistant", "content": followup})

        elif prompt.strip().lower() in ["no", "n", "nope"]:
            reset_msg = "What question can I help with?"
            with st.chat_message("assistant"):
                st.write(reset_msg)
            st.session_state.messages.append({"role": "assistant", "content": reset_msg})
            st.session_state.awaiting_more_info = False

    # Case 2: Normal Q&A
    else:
        messages_for_llm = keep_last_n_messages(
            st.session_state.messages + [
                {"role": "assistant", "content": "Answer the user's question so that a 10-year-old can understand."}
            ],
            n=2
        )

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

        # Always enter info loop
        followup = "Would you like more information?"
        with st.chat_message("assistant"):
            st.write(followup)
        st.session_state.messages.append({"role": "assistant", "content": followup})
        st.session_state.awaiting_more_info = True
