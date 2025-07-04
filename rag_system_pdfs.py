import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import various loaders based on file type
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredImageLoader, # For images (requires OCR setup for best results)
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader, # For .docx files
    UnstructuredExcelLoader, # For .xlsx files
    DirectoryLoader # To load all supported files from a directory
)

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load Environment Variables (remain unchanged from previous working version)
load_dotenv()

# Get Azure-specific environment variables for GENERATIVE model
azure_gen_api_key = os.getenv("AZURE_OPENAI_GEN_API_KEY")
azure_gen_endpoint = os.getenv("AZURE_OPENAI_GEN_ENDPOINT")
azure_gen_api_version = os.getenv("AZURE_OPENAI_GEN_API_VERSION")
azure_chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Get Azure-specific environment variables for EMBEDDING model
azure_embed_api_key = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
azure_embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT")
azure_embed_api_version = os.getenv("AZURE_OPENAI_EMBED_API_VERSION")
azure_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Basic validation
if not all([azure_gen_api_key, azure_gen_endpoint, azure_gen_api_version,
            azure_chat_deployment_name, azure_embed_api_key,
            azure_embed_endpoint, azure_embed_api_version,
            azure_embedding_deployment_name]):
    raise ValueError(
        "Azure OpenAI environment variables for both generative and embedding models "
        "not found. Please set all required AZURE_OPENAI_GEN_*, AZURE_OPENAI_EMBED_*, "
        "and corresponding DEPLOYMENT_NAMEs in your .env file."
    )

# 2. Data Loading and Chunking (UPDATED)
def load_and_chunk_documents(data_path: str):
    """
    Loads documents from a given path (file or directory) and splits them into chunks.
    Automatically determines the loader based on file extension or uses DirectoryLoader.
    """
    documents = []

    if os.path.isfile(data_path):
        # Determine loader based on file extension
        file_extension = os.path.splitext(data_path)[1].lower()
        print(f"Loading single file: {data_path} with extension: {file_extension}")
        if file_extension == ".txt":
            loader = TextLoader(data_path)
        elif file_extension == ".pdf":
            loader = PyPDFLoader(data_path)
        elif file_extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
            # Note: UnstructuredImageLoader requires Tesseract OCR or other backend.
            # For robust OCR, consider Azure Computer Vision or Google Cloud Vision.
            loader = UnstructuredImageLoader(data_path)
        elif file_extension == ".html" or file_extension == ".htm":
            loader = UnstructuredHTMLLoader(data_path)
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(data_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(data_path)
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(data_path)
        else:
            print(f"Warning: No specific loader for {file_extension}. Attempting TextLoader.")
            loader = TextLoader(data_path) # Fallback to TextLoader, might not work well
        documents.extend(loader.load())
    elif os.path.isdir(data_path):
        print(f"Loading documents from directory: {data_path}")
        # DirectoryLoader can discover and load various document types
        # You can specify `glob` to filter by pattern, and `loader_cls` for specific types
        # For simplicity, we'll try to load all discoverable documents
        loader = DirectoryLoader(
            data_path,
            loader_cls=TextLoader, # Default loader if nothing else matches
            glob="**/*", # Load all files recursively
            use_multithreading=True, # Can speed up loading large directories
            # Specify specific loaders for different file types
            # Note: Unstructured requires specific dependencies for each type
            # You can also use loader_map to map extensions to specific Unstructured loaders
            # Example for loader_map:
            # loader_map={
            #     ".pdf": PyPDFLoader,
            #     ".html": UnstructuredHTMLLoader,
            #     ".docx": UnstructuredWordDocumentLoader,
            #     ".xlsx": UnstructuredExcelLoader,
            #     ".md": UnstructuredMarkdownLoader,
            #     ".txt": TextLoader,
            #     ".jpg": UnstructuredImageLoader,
            #     ".png": UnstructuredImageLoader,
            # }
        )
        # Using a simple DirectoryLoader with TextLoader as fallback or use loader_map
        # For a truly robust solution, a loader_map with Unstructured loaders is best.
        # For now, let's load all text-based files and extend if needed.
        # A more advanced approach might involve iterating through files and applying specific loaders.
        # For Unstructured, if [all-docs] is installed, DirectoryLoader often works well.
        try:
             documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading directory with default loader, trying individual files if possible: {e}")
            # Fallback for complex directory structures or missing unstructured dependencies
            for root, _, files in os.walk(data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file_path)[1].lower()
                    try:
                        if file_extension == ".txt":
                            documents.extend(TextLoader(file_path).load())
                        elif file_extension == ".pdf":
                            documents.extend(PyPDFLoader(file_path).load())
                        elif file_extension in [".png", ".jpg", ".jpeg"]:
                            documents.extend(UnstructuredImageLoader(file_path).load())
                        # Add more conditions for other file types as needed
                        else:
                            print(f"Skipping unsupported file: {file_path}")
                    except Exception as inner_e:
                        print(f"Failed to load {file_path}: {inner_e}")
    else:
        raise ValueError(f"Invalid data_path: {data_path}. Must be a file or a directory.")

    if not documents:
        print("No documents loaded. Please check your path and file types.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks

# 3. Create Embeddings and Vector Store (Indexing Phase - remains unchanged)
def create_vector_store(chunks):
    print("Creating embeddings and building vector store (ChromaDB) using Azure OpenAI...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment_name,
        openai_api_version=azure_embed_api_version,
        azure_endpoint=azure_embed_endpoint,
        openai_api_key=azure_embed_api_key
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created and persisted.")
    return vectorstore

# 4. Initialize LLM and RAG Chain (remains unchanged)
def initialize_rag_chain(vectorstore):
    print("Initializing LLM and RAG chain using Azure OpenAI...")
    llm = AzureChatOpenAI(
        azure_deployment=azure_chat_deployment_name,
        openai_api_version=azure_gen_api_version,
        azure_endpoint=azure_gen_endpoint,
        openai_api_key=azure_gen_api_key,
        temperature=0.7
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    print("RAG chain initialized.")
    return qa_chain

# Main execution (UPDATED to allow flexible data_path)
if __name__ == "__main__":
    # --- Configuration ---
    # Point this to your knowledge base.
    # It can be a single file (e.g., "my_document.pdf")
    # or a directory (e.g., "data/") containing various files.
    knowledge_base_path = r"C:\Users\aebrah51\Desktop\Ahmad Abdulrahem - IT Cloud Engineer.pdf" # Change this to your target file/directory

    # Decide if you want to rebuild the vector store every time
    REBUILD_VECTOR_STORE = True # Set to False to load existing DB

    # --- Execution Logic ---
    if REBUILD_VECTOR_STORE or not os.path.exists("./chroma_db"):
        document_chunks = load_and_chunk_documents(knowledge_base_path)
        if not document_chunks: # Exit if no documents were loaded/chunked
            print("No documents were loaded or chunked. Exiting.")
            exit()
        vector_db = create_vector_store(document_chunks)
    else:
        print("Loading existing ChromaDB vector store...")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=azure_embedding_deployment_name,
            openai_api_version=azure_embed_api_version,
            azure_endpoint=azure_embed_endpoint,
            openai_api_key=azure_embed_api_key
        )
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

    rag_chain = initialize_rag_chain(vector_db)

    print("\n--- Azure RAG System Ready! Ask your questions. (Type 'exit' to quit) ---")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            print("Exiting RAG system. Goodbye!")
            break

        print("Searching and generating response...")
        response = rag_chain.invoke({"query": query})

        print("\n--- Answer ---")
        print(response["result"])

       # print("\n--- Sources Used ---")
       # for doc in response["source_documents"]:
       #     print(f"- {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')}, Start Index: {doc.metadata.get('start_index', 'Unknown')})")
       # print("--------------------\n")