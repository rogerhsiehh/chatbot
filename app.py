import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAIError
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="💼 Chat with Hsien-Pang's CV", layout="centered")
st.title("💼 Chat with Hsien-Pang's CV")

# --- Display CV Link at Top ---
with st.expander("📄 View My Full CV (Google Docs)", expanded=False):
    st.markdown(
        '[Open CV in Google Docs](https://docs.google.com/document/d/1opS6_bmLjOhgWy3GCrG2ydImx4SVP1edq9pgTWa7DmQ/edit?usp=sharing)',
        unsafe_allow_html=True
    )

st.markdown("Ask a question about my experience, education, or skills:")

# --- Load and Process CV into Vectorstore ---
@st.cache_resource
def load_vectorstore():
    reader = PdfReader("HsienPangHsiehCV.pdf")
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Load once
vectorstore = load_vectorstore()

# --- Input ---
query = st.text_input("Your question")

# --- Answer Logic ---
if query:
    with st.spinner("Searching..."):
        try:
            docs = vectorstore.similarity_search(query, k=3)
            context = " ".join([doc.page_content.strip().replace("\n", " ") for doc in docs])
            prompt = (
                f"You are an assistant helping answer questions based on the CV content below. "
                f"Answer clearly, completely, and based only on the CV. "
                f"If multiple relevant items exist (e.g., multiple degrees, experiences, skills), include all of them. "
                f"If the answer is not available in the CV, say: "
                f"\"That's a great question, but you might need to ask Hsien-Pang in person to get an accurate answer.\"\n\n"
                f"CV:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            from openai import OpenAI
            client = OpenAI()
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            answer = completion.choices[0].message.content.strip()
            st.subheader("📘 Answer:")
            st.success(answer)

        except OpenAIError as e:
            st.error(f"❌ OpenAI Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")