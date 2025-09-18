import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- SQLite fix for ChromaDB on Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

chromaDB_path = "./chromaDB_labs"
chroma_client = chromadb.PersistentClient(chromaDB_path)

st.title("Lab 4: RAG Chatbot with Vector DB")

openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"), key="model_selector")
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

if "openai_client" not in st.session_state:
    api_key = st.secrets["openai_api_key"]
    st.session_state.openai_client = OpenAI(api_key=api_key)


def build_lab4_vectorDB(pdf_folder="./pdfs"):
    if "Lab4_vectorDB" in st.session_state:
        return

    collection = chroma_client.get_or_create_collection("Lab4Collection")

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    openai_client = st.session_state.openai_client

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        reader = PdfReader(pdf_path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        collection.add(
            documents=[text],
            ids=[pdf_file],
            embeddings=[embedding],
            metadatas=[{"filename": pdf_file}]
        )

    st.session_state.Lab4_vectorDB = collection
    st.write("✅ VectorDB created and stored in session_state.")


if "Lab4_vectorDB" not in st.session_state:
    build_lab4_vectorDB("./pdfs")


if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Lab 4 Chatbot")

query = st.chat_input("Ask me something about the PDFs...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    openai_client = st.session_state.openai_client

    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    collection = st.session_state.Lab4_vectorDB
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    retrieved_docs = [doc for doc in results["documents"][0]]

    context_text = "\n\n".join(retrieved_docs)
    system_prompt = f"You are a helpful course assistant. Use the following course documents if relevant:\n\n{context_text}\n\nUser question: {query}"

    completion = openai_client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    answer = completion.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
