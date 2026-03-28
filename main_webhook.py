import os
import requests
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

load_dotenv()

#this file is specifically for chatwoot integration and can be ignored if using any other front end integration

app = FastAPI(title="Automart AI - Webhook Edition")

# --- 1. CONFIGURATION & AI SETUP ---
# We need Chatwoot credentials to "push" messages back to their server
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "https://app.chatwoot.com")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID", "1")

# Standard AI Setup (Same as main.py)
class RouteDecision(BaseModel):
    action: str = Field(description="Must be 'reply' or 'route_to_human'")
    response: str = Field(description="The answer to the user, or the escalation message.")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vectorstore = QdrantVectorStore(client=client, collection_name="automart_chat_history", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(RouteDecision)

# --- 2. THE WEBHOOK RECEIVER ---
@app.post("/webhook")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    WHY THIS ENDPOINT TAKES A RAW REQUEST: 
    Third-party webhooks send massive JSON payloads. Instead of writing a massive 
    Pydantic model for data we don't care about, we just extract what we need.
    """
    data = await request.json()
    
    # Check if the event is a new message being created
    if data.get("event") == "message_created":
        # CRITICAL: We only want to reply to human users ("incoming"), not to messages 
        # sent by human agents or our own bot ("outgoing"), otherwise we create an infinite loop!
        if data.get("message_type") == "incoming":
            
            user_message = data.get("content")
            conversation_id = data.get("conversation", {}).get("id")
            
            # WHY WE USE BACKGROUND TASKS:
            # Chatwoot expects an immediate "200 OK" response. If we make Chatwoot wait 
            # 3 seconds for Gemini to think, Chatwoot assumes our server crashed and will 
            # resend the message, causing duplicate replies. Background tasks prevent this.
            if user_message and conversation_id:
                background_tasks.add_task(process_and_reply, conversation_id, user_message)
    
    # Instantly tell Chatwoot "We received it!"
    return {"status": "success"}

# --- 3. THE ASYNC PROCESSOR ---
def process_and_reply(conversation_id: int, user_message: str):
    """
    This function runs in the background. It asks Gemini for the answer, 
    and then uses the requests library to push that answer back to Chatwoot.
    """
    # Retrieve context from Qdrant
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
    
    # Headers required to authenticate with Chatwoot's API
    headers = {"api_access_token": CHATWOOT_API_TOKEN, "Content-Type": "application/json"}
    
    if decision.action == "reply":
        # POST the AI's answer into the chat widget
        url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
        requests.post(url, headers=headers, json={"content": decision.response})
        
    elif decision.action == "route_to_human":
        # 1. Post the escalation message so the user knows they are being transferred
        msg_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
        requests.post(msg_url, headers=headers, json={"content": decision.response})
        
        # 2. Toggle the conversation status in Chatwoot to "open" (Needs human attention)
        # This triggers the notification "Ding!" on the human agents' dashboard
        status_url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/toggle_status"
        requests.post(status_url, headers=headers, json={"status": "open"})

if __name__ == "__main__":
    # Run on port 8001 so it doesn't conflict with main.py if both are running
    uvicorn.run("main_webhook:app", host="0.0.0.0", port=8001, reload=True)