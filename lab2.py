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
st.title("📄 LAB 2")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Load OpenAI API key from secrets
openai_api_key = st.secrets.get("openai_api_key")
if not openai_api_key:
    st.info("No valid OpenAI key detected")
else:
    # Create OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Sidebar: Summary Options
    st.sidebar.header("Summary Settings")
    summary_type = st.sidebar.radio(
        "Choose summary format:",
        ["100 words", "2 paragraphs", "5 bullet points"]
    )
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    model = "gpt-4o" if use_advanced else "gpt-4o-mini"

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    document = None
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            document = uploaded_file.read().decode("utf-8")
        elif file_extension == "pdf":
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type")

    # If document is uploaded, generate summary
    if document:
        st.subheader("Summary")
        
         # Sidebar: Language selection
        st.sidebar.header("Language Options")
        language = st.sidebar.selectbox(
            "Select language for summary:",
            ["English", "Spanish", "French", "German", "Chinese"]
        )

        # Update the instruction to include language
        if summary_type == "100 words":
            instruction = f"Summarize the document in exactly 100 words in {language}."
        elif summary_type == "2 paragraphs":
            instruction = f"Summarize the document in 2 connecting paragraphs in {language}."
        else:
            instruction = f"Summarize the document in 5 bullet points in {language}."

        with st.spinner(f"Generating summary using {model}..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                        {"role": "user", "content": f"{instruction}\n\nDocument:\n{document}"}
                    ]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")