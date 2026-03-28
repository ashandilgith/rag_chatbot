import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

def build_database():
    print("Loading dummy data...")
    loader = TextLoader("dummy_data.txt")
    documents = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)

    print("Connecting to Qdrant and generating Google Embeddings...")
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Push vectors to Qdrant Cloud
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="automart_chat_history"
    )
    print("Database built successfully!")

if __name__ == "__main__":
    build_database()