import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

chromaDB_path = "./chromaDB_labs"
chroma_client = chromadb.PersistentClient(chromaDB_path)

st.title("Lab 4: Embeddings")

openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

if "client" not in st.session_state:
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


col1, col2 = st.columns([1, 2])

with col1:
    topic = st.selectbox("Choose a topic", ("", "Text Mining", "GenAI"))

with col2:
    custom_query = st.text_input("...or type your own query")

submit = st.button("Submit")

if submit:
    if not topic and not custom_query:
        st.warning("Please select a topic or enter a query.")
    else:
        query_text = custom_query if custom_query else topic
        openai_client = st.session_state.openai_client

        # Get embedding for the chosen input
        response = openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Search the vector DB
        collection = st.session_state.Lab4_vectorDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3  # top 3 closest docs
        )

        # Show results
        st.subheader(f"🔎 Results for: {query_text}")
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            st.write(f"{i+1}. The following file/syllabus might be helpful: *{doc_id}*")