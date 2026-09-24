import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from graph import app_graph, SupportResponse

app = FastAPI(
    title="Zepto Support Assistant API",
    description="GenAI RAG pipeline for Zepto customer support.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the return policy for grocery items?")

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Zepto Support Assistant"}

@app.post("/ask", response_model=SupportResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "final_output": {}
    }
    
    result = app_graph.invoke(initial_state)
    return result["final_output"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
