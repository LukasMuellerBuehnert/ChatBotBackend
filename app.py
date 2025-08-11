from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import json, os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM = (
  "Du bist ein Website-Assistent. Antworte kurz auf der Sprache, in der sich der Fragesteller ausdrückt, "
  "nur basierend auf den bereitgestellten Auszügen. "
  "Wenn unklar: sag es ehrlich und verweise auf /kontakt."
)

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

def llm_answer(question: str, snippets: list[dict]) -> str:
    ctx = "\n".join([f"- {d['text']} (Quelle: {d['url']})" for d in snippets])
    prompt = f"Frage: {question}\n\nRelevante Auszüge:\n{ctx}\n\nAntwort:"
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"system","content": SYSTEM},{"role":"user","content": prompt}],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()
    
@app.post("/chat")
def chat(m: Msg):
    if not bm25 or not docs:
        return {"answer": "Keine Wissensbasis geladen.", "sources": []}

    tokens = tokenize(m.message)
    scores = bm25.get_scores(tokens)
    if not scores.size:
        return {"answer": "Weiß ich nicht. Bitte /kontakt nutzen.", "sources": []}

    # Top-3 Chunks auswählen
    top_k_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:3]
    if scores[top_k_idx[0]] < THRESHOLD:
        return {"answer": "Weiß ich nicht. Bitte /kontakt nutzen.", "sources": []}
    snippets = [docs[i] for i in top_k_idx]

    # LLM formulieren lassen
    ans = llm_answer(m.message, snippets)
    return {"answer": ans, "sources": [s["url"] for s in snippets]}
