from langchain_community.vectorstores import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

DB_PATH = "backend/vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=SecretStr(groq_api_key) if groq_api_key else None
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer:
""")

def get_response(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    return llm.invoke(final_prompt).content