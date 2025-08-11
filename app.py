from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import json, os

app = FastAPI()

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lukasmuellerbuehnert.github.io"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Daten laden ---
PATH = "data/faq.jsonl"
docs = [json.loads(l) for l in open(PATH, "r", encoding="utf-8")] if os.path.exists(PATH) else []
def tokenize(s: str): return s.lower().split()
corpus = [tokenize(d["text"]) for d in docs]
bm25 = BM25Okapi(corpus) if docs else None
THRESHOLD = 1.0  # minimaler Score

class Msg(BaseModel):
    message: str

@app.get("/healthz")
def health(): return {"ok": True, "docs": len(docs)}

@app.post("/chat")
def chat(m: Msg):
    if not bm25 or not docs:
        return {"answer": "Keine Wissensbasis geladen.", "sources": []}
    tokens = tokenize(m.message)
    scores = bm25.get_scores(tokens)
    i = int(max(range(len(scores)), key=lambda k: scores[k])) if scores.size else -1
    if i == -1 or scores[i] < THRESHOLD:
        return {"answer": "Weiß ich nicht. Bitte /kontakt nutzen.", "sources": []}
    d = docs[i]
    return {"answer": d["text"], "sources": [d["url"]]}
