import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import various loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredImageLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    DirectoryLoader,
)

# Import for Azure File Share
from azure.storage.fileshare import ShareServiceClient # Corrected import for File Share
from io import BytesIO # To handle file content in memory

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load Environment Variables
load_dotenv()

# Azure OpenAI General Settings
azure_gen_api_key = os.getenv("AZURE_OPENAI_GEN_API_KEY")
azure_gen_endpoint = os.getenv("AZURE_OPENAI_GEN_ENDPOINT")
azure_gen_api_version = os.getenv("AZURE_OPENAI_GEN_API_VERSION")
azure_chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

azure_embed_api_key = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
azure_embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT")
azure_embed_api_version = os.getenv("AZURE_OPENAI_EMBED_API_VERSION")
azure_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Azure File Share Specific Settings
azure_file_share_connection_string = os.getenv("AZURE_FILE_SHARE_CONNECTION_STRING")
azure_file_share_name = os.getenv("AZURE_FILE_SHARE_NAME")
azure_file_share_path = os.getenv("AZURE_FILE_SHARE_PATH", "") # Default to empty string for root

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

# --- HELPER FUNCTIONS FOR LOADING FROM CLOUD SERVICES ---

def load_from_azure_file_share(connection_string: str, share_name: str, folder_path: str = ""):
    print(f"Loading documents from Azure File Share: '{share_name}/{folder_path}'...")
    documents = []
    try:
        share_service_client = ShareServiceClient.from_connection_string(conn_str=connection_string)
        share_client = share_service_client.get_share_client(share_name)

        # Ensure the temp directory exists
        temp_dir = "temp_rag_files"
        os.makedirs(temp_dir, exist_ok=True)

        for file_item in share_client.list_directories_and_files(name_starts_with=folder_path):
            if file_item.is_directory:
                continue # Skip directories for now, focus on files
            
            full_file_path = file_item.name
            print(f"  Attempting to load: {full_file_path}")

            file_client = share_client.get_file_client(full_file_path)
            
            try:
                download_stream = file_client.download_file()
                file_content = download_stream.readall()
                
                file_extension = os.path.splitext(full_file_path)[1].lower()
                
                temp_file_name = os.path.basename(full_file_path)
                temp_file_full_path = os.path.join(temp_dir, temp_file_name)

                with open(temp_file_full_path, "wb") as f:
                    f.write(file_content)
                
                if file_extension == ".txt":
                    documents.extend(TextLoader(temp_file_full_path).load())
                elif file_extension == ".pdf":
                    documents.extend(PyPDFLoader(temp_file_full_path).load())
                elif file_extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
                    documents.extend(UnstructuredImageLoader(temp_file_full_path).load())
                elif file_extension == ".html" or file_extension == ".htm":
                    documents.extend(UnstructuredHTMLLoader(temp_file_full_path).load())
                elif file_extension == ".md":
                    documents.extend(UnstructuredMarkdownLoader(temp_file_full_path).load())
                elif file_extension == ".docx":
                    documents.extend(UnstructuredWordDocumentLoader(temp_file_full_path).load())
                elif file_extension == ".xlsx":
                    documents.extend(UnstructuredExcelLoader(temp_file_full_path).load())
                else:
                    print(f"  Warning: Unsupported file type for Azure File Share: {full_file_path}. Skipping.")

                os.remove(temp_file_full_path) # Clean up the temporary file

            except Exception as e:
                print(f"  Error loading file '{full_file_path}' from Azure File Share: {e}")

        # Clean up the temporary directory if empty, or leave it if you prefer
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)

    except Exception as e:
        print(f"Error accessing Azure File Share: {e}")
    return documents


# 2. Data Loading and Chunking
def load_and_chunk_documents(data_path: str):
    """
    Loads documents from a given path (local file or directory) or from Azure File Share.
    """
    all_documents = []

    # --- Local File/Directory Loading ---
    if data_path and (os.path.isfile(data_path) or os.path.isdir(data_path)):
        if os.path.isfile(data_path):
            file_extension = os.path.splitext(data_path)[1].lower()
            print(f"Loading single local file: {data_path} with extension: {file_extension}")
            if file_extension == ".txt":
                loader = TextLoader(data_path)
            elif file_extension == ".pdf":
                loader = PyPDFLoader(data_path)
            elif file_extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
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
                loader = TextLoader(data_path)
            all_documents.extend(loader.load())
        elif os.path.isdir(data_path):
            print(f"Loading documents from local directory: {data_path}")
            loader = DirectoryLoader(
                data_path,
                glob="**/*",
                use_multithreading=True,
            )
            try:
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading directory with default loader, trying individual files if possible: {e}")
                for root, _, files in os.walk(data_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_extension = os.path.splitext(file_path)[1].lower()
                        try:
                            if file_extension == ".txt":
                                all_documents.extend(TextLoader(file_path).load())
                            elif file_extension == ".pdf":
                                all_documents.extend(PyPDFLoader(file_path).load())
                            elif file_extension in [".png", ".jpg", ".jpeg"]:
                                all_documents.extend(UnstructuredImageLoader(file_path).load())
                            else:
                                print(f"Skipping unsupported local file: {file_path}")
                        except Exception as inner_e:
                            print(f"Failed to load {file_path}: {inner_e}")
    else:
        if data_path:
            print(f"Local path '{data_path}' not found or unsupported. Skipping local file loading.")

    # --- Azure File Share Loading ---
    if azure_file_share_connection_string and azure_file_share_name:
        all_documents.extend(load_from_azure_file_share(
            azure_file_share_connection_string,
            azure_file_share_name,
            azure_file_share_path
        ))
    else:
        print("Azure File Share configuration missing or incomplete. Skipping Azure File Share loading.")

    if not all_documents:
        print("No documents loaded from any source. Please check your paths and configurations.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Loaded {len(all_documents)} total documents from all sources.")
    print(f"Split {len(all_documents)} document(s) into {len(chunks)} chunks.")
    return chunks

# 3. Create Embeddings and Vector Store (Indexing Phase)
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

# 4. Initialize LLM and RAG Chain
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

# Main execution
if __name__ == "__main__":
    # --- Configuration ---
    # Set this to a local path (file or directory) or leave blank ("") if you only
    # want to load from Azure File Share configured in .env
    local_knowledge_base_path = r"knowledge_base.txt" # Example local file (use raw string 'r""' for Windows paths)
    # local_knowledge_base_path = r"C:\Users\aebrah51\Desktop\YourLocalDocsFolder" # Example local directory
    # local_knowledge_base_path = "" # Set to empty string if no local files

    REBUILD_VECTOR_STORE = True # Set to False to load existing DB

    # --- Execution Logic ---
    if REBUILD_VECTOR_STORE or not os.path.exists("./chroma_db"):
        document_chunks = load_and_chunk_documents(local_knowledge_base_path)
        if not document_chunks:
            print("No documents were loaded or chunked from any source. Exiting.")
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