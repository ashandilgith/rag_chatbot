import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

load_dotenv()

app = FastAPI(title="Automart AI Backend")

# Allow the frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Strict Output Schema for Gemini
class RouteDecision(BaseModel):
    action: str = Field(description="Must be 'reply' or 'route_to_human'")
    response: str = Field(description="The answer to the user, or the escalation message.")

# 2. Connect to the Live Qdrant Database
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vectorstore = QdrantVectorStore(client=client, collection_name="automart_chat_history", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Initialize Gemini 1.5 Flash
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0).with_structured_output(RouteDecision)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(RouteDecision)


class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    # Retrieve context from Qdrant
    docs = retriever.invoke(request.message)
    context = "\n".join([doc.page_content for doc in docs])

    # System prompt forcing the routing decision
    system_prompt = """You are an AI router for Automart PH. 
    Analyze the user's message and the provided context.
    
    Rules:
    1. If the context contains the answer, action is 'reply' and generate a helpful response.
    2. If the user is angry, asks for a human, or the context does NOT contain the answer, action is 'route_to_human' and generate a polite transfer message.
    
    Context:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    decision = chain.invoke({"context": context, "question": request.message})
    
    return {"action": decision.action, "response": decision.response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)