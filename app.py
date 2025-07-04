import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tempfile import NamedTemporaryFile
import shutil
import time

from rag_system import load_from_azure_file_share  # Reuse your logic

# Load environment variables
load_dotenv()

# Azure config
embed_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBED_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBED_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_EMBED_API_KEY"),
)

chat_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_GEN_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_GEN_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_GEN_API_KEY"),
    temperature=0.7,
    streaming=False,
)

# Load or create vector store
PERSIST_DIR = "./chroma_db"

@st.cache_resource
def load_vectorstore():
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embed_model)
    else:
        return None

def save_to_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vs = Chroma.from_documents(chunks, embed_model, persist_directory=PERSIST_DIR)
    vs.persist()
    return vs

def create_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def load_uploaded_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[1].lower()
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs.extend(PyPDFLoader(tmp_path).load())
        elif suffix == ".txt":
            from langchain_community.document_loaders import TextLoader
            docs.extend(TextLoader(tmp_path).load())
        elif suffix == ".docx":
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            docs.extend(UnstructuredWordDocumentLoader(tmp_path).load())
        else:
            st.warning(f"Unsupported file type: {file.name}")
        os.remove(tmp_path)
    return docs

# --- Streamlit UI ---

st.set_page_config(page_title="Azure RAG Assistant", layout="wide")
st.title("📘 Azure RAG System")
st.markdown("Ask questions based on uploaded documents or your Azure File Share.")

# Upload documents
uploaded = st.file_uploader("Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# Option to load Azure File Share
load_azure = st.checkbox("Also load documents from Azure File Share", value=False)

# REBUILD button
rebuild = st.button("Rebuild Vector Store")

if rebuild:
    docs = []

    if uploaded:
        st.info("Loading uploaded documents...")
        docs.extend(load_uploaded_files(uploaded))

    if load_azure:
        st.info("Loading documents from Azure File Share...")
        azure_docs = load_from_azure_file_share(
            os.getenv("AZURE_FILE_SHARE_CONNECTION_STRING"),
            os.getenv("AZURE_FILE_SHARE_NAME"),
            os.getenv("AZURE_FILE_SHARE_PATH", "")
        )
        docs.extend(azure_docs)

    if not docs:
        st.error("No documents found to build the vector store.")
    else:
        st.success(f"Loaded {len(docs)} document(s). Creating vector store...")
        vectorstore = save_to_vectorstore(docs)
        st.success("Vector store updated!")

# Chat Interface
st.divider()
st.subheader("💬 Ask a question")

question = st.text_input("Your question")

if question:
    vectorstore = load_vectorstore()
    if not vectorstore:
        st.error("Vector store not found. Please rebuild it using uploaded files or Azure File Share.")
    else:
        qa_chain = create_qa_chain(vectorstore)

        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"query": question})
            st.markdown("### ✅ Answer")
            st.write(response["result"])

            st.markdown("### 📄 Sources")
            for i, doc in enumerate(response["source_documents"]):
                meta = doc.metadata
                st.markdown(f"**Source {i+1}:** {meta.get('source', 'unknown')}")
                st.code(doc.page_content[:500] + "...")

