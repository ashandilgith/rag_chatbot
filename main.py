import os
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
# from dispatcher import trigger_human_handoff # Uncomment if using the Twilio/Telegram alerts

load_dotenv()

app = FastAPI(title="Automart AI - Omni Router")

# Allow your local index.html to talk to the cloud server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. SHARED AI SETUP ---
class ChatRequest(BaseModel):
    user_id: str
    message: str

class RouteDecision(BaseModel):
    action: str = Field(description="Must be 'reply' or 'route_to_human'")
    response: str = Field(description="The answer to the user, or the escalation message.")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vectorstore = QdrantVectorStore(client=client, collection_name="automart_chat_history", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(RouteDecision)

def get_ai_decision(user_message: str):
    """A helper function so we don't write the LangChain logic twice."""
    docs = retriever.invoke(user_message)
    context = "\n".join([doc.page_content for doc in docs])
    system_prompt = """You are an AI router for Automart PH. 
    Analyze the user's message and the provided context.
    Rules:
    1. If the context contains the answer, action is 'reply'.
    2. If the user is angry, asks for a human, or context is missing, action is 'route_to_human'.
    Context: {context}"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    chain = prompt | llm
    return chain.invoke({"context": context, "question": user_message})


# --- DOOR 1: THE CUSTOM UI (index.html) ---
@app.post("/chat")
async def handle_custom_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    decision = get_ai_decision(request.message)
    
    if decision.action == "route_to_human":
        # background_tasks.add_task(trigger_human_handoff, request.user_id, request.message)
        pass # Replace pass with the line above if you want Telegram alerts
        
    return {"action": decision.action, "response": decision.response}


# --- DOOR 2: THE ZOHO WEBHOOK ---
@app.post("/zoho-webhook")
async def zoho_webhook(request: Request):
    data = await request.json()
    
    user_message = ""
    message_data = data.get("message")
    if isinstance(message_data, dict):
        user_message = message_data.get("text", "")
    elif isinstance(message_data, str):
        user_message = message_data
        
    if not user_message:
        return {"action": "reply", "replies": ["I am ready to help!"]}

    decision = get_ai_decision(user_message)
    
    if decision.action == "reply":
        return {"action": "reply", "replies": [decision.response]}
    elif decision.action == "route_to_human":
        return {"action": "forward", "replies": [decision.response]}