# ---------- Setup ----------
import sys, re, streamlit as st, pandas as pd
import pysqlite3
sys.modules["sqlite3"] = pysqlite3  # ensure modern sqlite3 for Chroma

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory


# ---------- Streamlit UI ----------
st.title("🎯 LangChain Research Paper Agent Lab")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent" not in st.session_state:
    st.session_state.agent = None

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Choose Model", ["gpt-4o-mini", "gpt-4o"])
    uploaded = st.file_uploader("📄 Upload the provided arXiv CSV", type=["csv"])


# ---------- Step 1: Build Vector DB ----------
if uploaded and st.session_state.vectorstore is None:
    with st.spinner("Building vector database from uploaded CSV..."):
        df = pd.read_csv(uploaded)
        st.session_state.df = df

        # Create LangChain Documents for embedding
        docs = []
        for _, row in df.iterrows():
            text = (
                f"Title: {row.get('title','')}\n"
                f"Authors: {row.get('authors','')}\n"
                f"Abstract: {row.get('abstract','')}\n"
                f"Year: {row.get('year','')}\n"
                f"Category: {row.get('category','')}\n"
                f"Venue: {row.get('venue','')}"
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "title": row.get("title", ""),
                    "authors": row.get("authors", ""),
                    "link": row.get("link", ""),
                    "year": row.get("year", "")
                }
            ))

        # Create embeddings + vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
        st.session_state.vectorstore = Chroma.from_documents(docs, embeddings)

    st.success(f"✅ Indexed {len(docs)} papers from your CSV.")


# ---------- Step 2: Define Tools ----------
def search_papers(query: str) -> str:
    """Search the vector DB for relevant (unique) papers."""
    db = st.session_state.vectorstore
    results = db.similarity_search(query, k=8)  # slightly higher k to offset duplicates
    if not results:
        return "No results found. Try another topic."

    seen_titles = set()
    output = []
    i = 0

    for doc in results:
        title = doc.metadata.get("title", "Untitled").strip()
        if not title or title.lower() in seen_titles:
            continue  # skip duplicate or empty titles
        seen_titles.add(title.lower())

        authors = doc.metadata.get("authors", "Unknown")
        link = doc.metadata.get("link", "")
        i += 1

        output.append(
            f"### {i}. {title}\n"
            f"👥 {authors}\n"
            f"🔗 [Read Paper]({link})\n\n"
            f"🧾 {doc.page_content[:400]}...\n"
        )

        if i >= 3:  # show top 3 unique papers only
            break

    return "\n".join(output) if output else "No unique papers found."


def summarize_topic(query: str) -> str:
    """Summarize a research topic briefly."""
    llm = ChatOpenAI(model=model, openai_api_key=st.secrets["openai_api_key"], temperature=0.3)
    prompt = f"Summarize this academic research topic for students:\n\n{query}"
    return llm.predict(prompt)


def compare_papers(query: str) -> str:
    """Compare two papers using their titles."""
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+", query, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "Please specify two paper titles, e.g., 'Compare Dynamic Backtracking and GSAT'."

    df = st.session_state.df
    def find_paper(title):
        match = df[df["title"].str.contains(title.strip(), case=False, na=False)]
        if match.empty:
            return None
        row = match.iloc[0]
        return f"Title: {row['title']}\nAuthors: {row['authors']}\nAbstract: {row['abstract'][:600]}..."

    p1, p2 = find_paper(parts[0]), find_paper(parts[1])
    if not p1 or not p2:
        return "Could not find one or both papers in the dataset."

    llm = ChatOpenAI(model=model, openai_api_key=st.secrets["openai_api_key"], temperature=0.4)
    prompt = (
        f"Compare the following two AI papers in terms of problem, methods, and conclusions:\n\n"
        f"{p1}\n\n{p2}\n\nProvide a concise, student-friendly comparison."
    )
    return llm.predict(prompt)


# ---------- Step 3: Register Tools ----------
tools = [
    Tool(name="SearchPapers", func=search_papers, description="Find research papers based on a topic."),
    Tool(name="SummarizeTopic", func=summarize_topic, description="Summarize a research topic."),
    Tool(name="ComparePapers", func=compare_papers, description="Compare two research papers by title.")
]


# ---------- Step 4: Initialize Agent ----------
if st.session_state.vectorstore and st.session_state.agent is None:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model=model, openai_api_key=st.secrets["openai_api_key"], streaming=True, temperature=0.3)

    st.session_state.agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=False
    )
    st.success("🤖 Agent ready! Ask questions below.")


# ---------- Step 5: Chat Interface (Fixed History) ----------
if st.session_state.agent:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # store chat UI messages

    # Display all previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box for user query
    if user_input := st.chat_input("Ask about a topic or paper..."):
        # --- Display and save user message ---
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- Process via Agent ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.invoke({"input": user_input})
                    output = response.get("output") if isinstance(response, dict) else str(response)
                except Exception as e:
                    output = f"⚠️ Error: {e}"

            # --- Display and save assistant message ---
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.markdown(output)
else:
    st.info("⬆️ Upload the provided arXiv CSV to begin.")
