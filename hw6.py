import sys, re, streamlit as st, pandas as pd, os
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# LangChain core components
from langchain_core.documents import Document
from langchain_core.tools import Tool

# LLM integrations - OpenAI only
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import Chroma

# Chains and agents
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


# VECTORSTORE SETUP

@st.cache_resource
def initialize_vectorstore():
    """Initialize vector database from CSV"""
    CSV_PATH = "arxiv_papers_extended_20251019_150748.csv"
    PERSIST_DIR = "HW6_vector_db"

    os.makedirs(PERSIST_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found at {CSV_PATH}")
        st.stop()

    with st.spinner("Building or loading vector database..."):
        df = pd.read_csv(CSV_PATH)

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

        embeddings = OpenAIEmbeddings(api_key=st.secrets["openai_api_key"])

        if os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            st.success(f"Loaded {len(docs)} papers from existing DB")
        else:
            vectorstore = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=PERSIST_DIR
            )
            st.success(f"Indexed {len(docs)} papers and saved DB")

        return vectorstore, df


# MAIN APP START

st.set_page_config(page_title="LangChain Research Paper Agent", page_icon="🎯")
st.title("🎯 LangChain Research Paper Agent")
st.markdown("This Streamlit app uses LangChain + OpenAI to explore arXiv papers interactively.")

# Initialize session state
for key in ["hw6_vectorstore", "hw6_agent", "hw6_df", "hw6_messages"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "hw6_messages" else []

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Choose Model", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"], key="model")

    if st.session_state.hw6_df is not None:
        st.success(f"📊 {len(st.session_state.hw6_df)} papers loaded")

    if st.button("🔄 Rebuild Database"):
        st.session_state.hw6_vectorstore = None
        st.session_state.hw6_agent = None
        st.cache_resource.clear()
        st.rerun()


# Initialize vectorstore and data
if st.session_state.hw6_vectorstore is None:
    try:
        st.session_state.hw6_vectorstore, st.session_state.hw6_df = initialize_vectorstore()
    except Exception as e:
        st.error(f"Vectorstore Error: {e}")
        st.stop()

# Initialize LLM
llm = ChatOpenAI(
    model=model,
    api_key=st.secrets["openai_api_key"],
    temperature=0.3
)


# TOOL FUNCTIONS

def search_papers(query: str) -> str:
    results = st.session_state.hw6_vectorstore.similarity_search(query, k=10)
    if not results:
        return f"No papers found about '{query}'"
    seen = set()
    output = []
    for doc in results:
        title = doc.metadata.get("title", "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        output.append(
            f"{len(output)+1}. {title}\n"
            f"Authors: {doc.metadata.get('authors', 'Unknown')}\n"
            f"Link: {doc.metadata.get('link', '')}\n"
        )
        if len(output) >= 3:
            break
    return "\n".join(output) if output else "No papers found."


def compare_papers(query: str) -> str:
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+", query, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "Specify two papers: 'paper1 and paper2'"
    df = st.session_state.hw6_df

    def find(title):
        match = df[df["title"].str.contains(title.strip(), case=False, na=False)]
        if match.empty:
            return None
        row = match.iloc[0]
        return f"{row['title']}\n{row['authors']}\n{row['abstract'][:400]}..."

    p1, p2 = find(parts[0]), find(parts[1])
    if not p1 or not p2:
        return "Could not find one or both papers"
    return f"## Paper 1\n{p1}\n\n## Paper 2\n{p2}"


# AGENT SETUP

tools = [
    Tool(name="SearchPapers", func=search_papers, description="Find research papers on a topic"),
    Tool(name="ComparePapers", func=compare_papers, description="Compare two papers by title")
]

if st.session_state.hw6_agent is None:
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        st.session_state.hw6_agent = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )
        st.success("Agent ready!")
    except Exception as e:
        st.error(f"Agent initialization error: {e}")
        st.stop()


# CHAT INTERFACE

for msg in st.session_state.hw6_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("💬 Ask about research papers..."):
    st.session_state.hw6_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.hw6_agent.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.hw6_messages
                })
                output = response.get("output", str(response))
            except Exception as e:
                output = f"Error: {e}"
        st.session_state.hw6_messages.append({"role": "assistant", "content": output})
        st.markdown(output)
