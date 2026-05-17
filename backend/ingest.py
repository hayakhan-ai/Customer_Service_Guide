from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "backend/data/docs"
DB_PATH = "backend/vectorstore"

def ingest():
    loader = DirectoryLoader(DATA_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Vector DB created!")

if __name__ == "__main__":
    ingest()