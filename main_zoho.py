import os
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

load_dotenv()

app = FastAPI(title="Automart AI - Zoho Zobot Edition")

# --- 1. AI SETUP ---
class RouteDecision(BaseModel):
    action: str = Field(description="Must be 'reply' or 'route_to_human'")
    response: str = Field(description="The answer to the user, or the escalation message.")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vectorstore = QdrantVectorStore(client=client, collection_name="automart_chat_history", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(RouteDecision)

# --- 2. THE ZOHO WEBHOOK ENDPOINT ---
@app.post("/zoho-webhook")
async def zoho_webhook(request: Request):
    """
    Zoho SalesIQ expects a SYNCHRONOUS response. 
    It will wait up to 5 seconds for us to return the JSON.
    """
    data = await request.json()
    
    # Safely extract the message text from Zoho's payload
    user_message = ""
    message_data = data.get("message")
    
    if isinstance(message_data, dict):
        user_message = message_data.get("text", "")
    elif isinstance(message_data, str):
        user_message = message_data
        
    # Fallback to prevent crashes if Zoho sends a blank connection ping
    if not user_message:
        return {"action": "reply", "replies": ["I am ready to help!"]}

    # 1. Ask Gemini / Qdrant
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
    decision = chain.invoke({"context": context, "question": user_message})
    
    # --- 3. THE ZOHO HANDOFF PROTOCOL ---
    
    if decision.action == "reply":
        # Zoho's exact format for sending a chat bubble
        return {
            "action": "reply",
            "replies": [decision.response]
        }
        
    elif decision.action == "route_to_human":
        # Zoho's exact format for transferring to a live human operator
        return {
            "action": "forward", 
            "replies": [decision.response]
        }

if __name__ == "__main__":
    # Running on port 8002 to keep things organized
    uvicorn.run("main_zoho:app", host="0.0.0.0", port=8002, reload=True)