import os
import json
import chromadb
from typing import List, TypedDict, Dict, Any
from pydantic import BaseModel, Field
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END
from prompts import STRUCTURED_PROMPT_TEMPLATE

# --- Output Pydantic Model ---
class SupportResponse(BaseModel):
    answer: str = Field(..., description="The response string")
    sources: List[str] = Field(default_factory=list, description="Document IDs used for answering")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

# --- LangGraph State ---
class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    final_output: Dict[str, Any]

# Global ChromaDB initialization
PERSIST_DIRECTORY = "chromadb_store"
COLLECTION_NAME = "zepto_policies"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

def get_collection():
    return chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

# --- Node 1: Classify Intent ---
def classify_intent(state: AgentState) -> AgentState:
    query = state["query"]
    mock_mode = os.getenv("MOCK_LLM", "1") != "0"
    
    if mock_mode:
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        query_lower = query.lower()
        if any(kw in query_lower for kw in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # Real LLM Classification path (Optional Extension)
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        res = llm.invoke(f"Classify query as either 'policy_question' or 'general_question': '{query}'. Respond with ONLY the label.")
        intent = "policy_question" if "policy_question" in res.content.lower() else "general_question"

    return {**state, "intent": intent}

# --- Node 2: Retrieve and Answer (Policy Question) ---
def retrieve_and_answer(state: AgentState) -> AgentState:
    query = state["query"]
    collection = get_collection()
    
    # Real Vector Search across both mock and real modes
    results = collection.query(query_texts=[query], n_results=3)
    
    retrieved_chunks = []
    sources = []
    if results and results["documents"]:
        for doc, meta, doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
            retrieved_chunks.append({"id": doc_id, "text": doc, "source": meta["source"]})
            if meta["source"] not in sources:
                sources.append(meta["source"])

    mock_mode = os.getenv("MOCK_LLM", "1") != "0"
    
    if mock_mode:
        top_snippet = retrieved_chunks[0]["text"][:200] if retrieved_chunks else ""
        canned_answer = f"Based on the retrieved context: {top_snippet}"
        output = {
            "answer": canned_answer,
            "sources": sources,
            "confidence": 1.0
        }
    else:
        # Optional Real LLM Path with Retry Logic
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        context_str = "\n".join([f"[{c['source']}]: {c['text']}" for c in retrieved_chunks])
        prompt = STRUCTURED_PROMPT_TEMPLATE.format(context=context_str, query=query)
        
        attempts = 0
        output = None
        while attempts < 3:
            try:
                raw_res = llm.invoke(prompt if attempts == 0 else f"{prompt}\n\nFIX ERROR: Ensure valid JSON strictly adhering to schema.")
                parsed = json.loads(raw_res.content)
                validated = SupportResponse(**parsed)
                output = validated.model_dump()
                break
            except Exception:
                attempts += 1
                
        if not output:
            output = {
                "answer": "Error: Failed to parse valid LLM response schema after retries.",
                "sources": sources,
                "confidence": 0.0
            }

    return {**state, "retrieved_chunks": retrieved_chunks, "final_output": output}

# --- Node 3: Direct Answer (General Question) ---
def direct_answer(state: AgentState) -> AgentState:
    mock_mode = os.getenv("MOCK_LLM", "1") != "0"
    
    if mock_mode:
        output = {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0
        }
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        res = llm.invoke(state["query"])
        output = {
            "answer": res.content,
            "sources": [],
            "confidence": 0.9
        }

    return {**state, "retrieved_chunks": [], "final_output": output}

# --- Router Function ---
def route_intent(state: AgentState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

# --- Build LangGraph StateGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()
