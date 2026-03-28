# rag_chatbot
# Automart PH - Intelligent RAG Router API

A headless, end-to-end Retrieval-Augmented Generation (RAG) system built to handle high-volume customer support queries. This API ingests historical chat logs and uses an agentic LLM router to dynamically decide whether to answer a user's question using semantic search or seamlessly escalate the conversation to a human agent.

## 🏗 System Architecture

This backend is designed to be fully decoupled from the frontend, allowing for integration via webhooks into existing CRM tools (Zendesk, Intercom) or custom React/Vue widgets.

* **Compute Framework:** FastAPI (Optimized for asynchronous cloud deployments)
* **LLM Engine:** Google Gemini 1.5 Flash (Handles both intent routing and text generation)
* **Embeddings:** Google `text-embedding-004` (Text-Gecko)
* **Vector Database:** Qdrant Cloud (For low-latency, persisted semantic search)
* **Orchestration:** LangChain (Managing the two-tier routing and retrieval pipeline)

## ⚙️ Core Capabilities

1. **Contextual Retrieval:** Converts incoming queries into vector embeddings and retrieves the top-K most relevant historical chat transcripts to formulate accurate, policy-compliant answers.
2. **Intent & Sentiment Routing:** Analyzes user frustration and query complexity. If a user demands a manager or the query falls outside the vector knowledge base, the API automatically triggers a `route_to_human` payload.
3. **Headless Integration:** Exposes a clean `POST /chat` endpoint returning structured JSON, abstracting the AI complexity away from frontend developers.

## 🚀 Quick Start (Local Development)

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
