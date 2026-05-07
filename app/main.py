from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="MetaTag")

class ExtractRequest(BaseModel):
    text: str
    top_k: Optional[int] = None

class EntityResponse(BaseModel):
    entity: str
    type: str
    score: float

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "bert-base-uncased-finetuned-metatag"}

@app.post("/extract")
async def extract(req: ExtractRequest):
    # Placeholder: returns mock entities
    # Real model loads from checkpoints in production
    entities = [
        {"entity": "Eleven", "type": "PER", "score": 0.97},
        {"entity": "Hawkins", "type": "LOC", "score": 0.91},
        {"entity": "friendship", "type": "THEME", "score": 0.88},
        {"entity": "dark", "type": "MOOD", "score": 0.83},
    ]
    return {"text": req.text, "entities": entities}
