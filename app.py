from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from groq import Groq
import json, os, re

# ---------- Einstellungen ----------
THRESHOLD = 1.0                    # Mindestscore für BM25
ALWAYS_LABELS = ["greeting","thanks","goodbye", "commonsense"]  # feste Intents
GROQ_MODEL = "llama-3.1-8b-instant"

# ---------- Hilfsfunktionen ----------
# fold sorgt dafür das Labels egal ob aus der Database oder vom LLM einheitlich geschrieben sind
def fold(s: str) -> str:
    # einfache Normalisierung (für Label-Matching)
    s = s.lower().strip()
    repl = {"ä":"ae","ö":"oe","ü":"ue","ß":"ss"}
    for a,b in repl.items(): s = s.replace(a,b)
    s = re.sub(r"\s+", " ", s)
    return s

# tokenizer nimmt chunks und zerlegt sie in kleingeschriebene Wörter
def tokenize(s: str):
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)

# ---------- Daten laden ----------
PATH = "data/faq.jsonl"
docs = []
if os.path.exists(PATH):
    with open(PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                # erwartete Felder: url, title, text
                if all(k in d for k in ("url","title","text")):
                    d["_title_fold"] = fold(d["title"])
                    docs.append(d)
            except Exception:
                pass

# Labels aus den Titeln + feste Intents
# Labelliste wird erstellt aus Datensatz und Labels die immer gelten
labels_from_data = {d["_title_fold"]: d for d in docs}
LABELS = list(labels_from_data.keys()) + ALWAYS_LABELS

# ---------- BM25 Index ----------
# tokenizer nimmt chunks und zerlegt sie in kleingeschriebene Wörteroka
corpus = [tokenize(d["text"]) for d in docs]
bm25 = BM25Okapi(corpus) if docs else None

# ---------- LLM-Client ----------
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def classify_lang_intent(q: str):
    """ Gibt (lang, intent) zurück. intent ∈ LABELS oder 'unknown'. """
    # hier wird ein string aus den labels erstellt
    labels_str = ", ".join(LABELS)
    prompt = (
      "Tasks:\n"
      "1) Detect user language (ISO-639-1 like 'de','en',...).\n"
      f"2) Classify the user's question into ONE label from this exact set: [{labels_str}]. "
      "If nothing fits, return 'unknown'.\n"
      'Return ONLY JSON: {"lang":"..","intent":".."}\n'
      f"User: {q}"
    )
    # api call
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":"Answer with valid JSON only."},
                  {"role":"user","content": prompt}],
        # deterministisch
        temperature=0
    ).choices[0].message.content
    try:
        obj = json.loads(r)
        lang = (obj.get("lang") or "en").lower()
        intent_raw = (obj.get("intent") or "unknown").lower().strip()
        intent_fold = fold(intent_raw)
        # wenn LLM den Originaltitel zurückgibt (nicht gefoldet): auch akzeptieren
        if intent_fold in labels_from_data:
            intent = intent_fold
        elif intent_raw in labels_from_data:
            intent = intent_raw
        elif intent_fold in ALWAYS_LABELS or intent_raw in ALWAYS_LABELS:
            intent = intent_fold if intent_fold in ALWAYS_LABELS else intent_raw
        else:
            intent = "unknown"
        return lang, intent
    except Exception:
        return "en", "unknown"



def smalltalk_llm(intent, lang):
    if intent not in {"greeting","thanks","goodbye","commonsense"}:
        return None
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":
           "One very short, friendly sentence in the target language. No emojis unless user used them."},
                  {"role":"user","content": f"Target language: {lang}\nIntent: {intent}"}],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()


def llm_answer(question: str, snippets: list[dict], lang: str) -> str:
    ctx = "\n".join([f"- {d['text']} (Quelle: {d['url']})" for d in snippets])
    sys = (
      "You are a website assistant. Answer briefly in the requested target language. "
      "Only use the provided excerpts; if insufficient, say you don't know and refer to /kontakt."
    )
    prompt = f"Target language: {lang}\nQuestion: {question}\n\nExcerpts:\n{ctx}\n\nAnswer:"
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content": sys},
                  {"role":"user","content": prompt}],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

# ---------- FastAPI ----------


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lukasmuellerbuehnert.github.io"],  # exakt die Origin
    allow_methods=["POST","OPTIONS"],      # OPTIONS explizit erlauben
    allow_headers=["Content-Type"],        # was du wirklich brauchst
    allow_credentials=False,
    max_age=3600,
)

class Msg(BaseModel):
    message: str

@app.get("/healthz")
def health():
    return {"ok": True, "docs": len(docs), "labels": len(LABELS)}

@app.options("/chat")
def options_chat():
    # Preflight sauber beantworten
    return Response(status_code=204)
    
@app.post("/chat")
def chat(m: Msg):
    if not bm25 or not docs:
        return {"answer": "Keine Wissensbasis geladen.", "sources": []}

    # 1) Sprache + Intent also das request parsing
    lang, intent = classify_lang_intent(m.message)
    
    # 2) Smalltalk direkt
    st = smalltalk_llm(intent, lang)
    if st:
        return {"answer": st, "sources":[]}

    # 3) Query bauen: Originalfrage + (Intent-Label) + (passender Titel/Text als Booster)
    query = m.message
    if intent in labels_from_data:
        d = labels_from_data[intent]
        query = f"{m.message} {d['title']} {d['text']}"

    # 4) BM25 → Top-3
    scores = bm25.get_scores(tokenize(query))
    if getattr(scores, "size", 0) == 0:
        return {"answer": "Weiß ich nicht. Bitte /kontakt nutzen.", "sources": []}

    top_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:3]
    if scores[top_idx[0]] < THRESHOLD:
        return {"answer": "Weiß ich nicht. Bitte /kontakt nutzen.", "sources": []}

    snippets = [docs[i] for i in top_idx]

    # 5) Finale Antwort in erkannter Sprache
    ans = llm_answer(m.message, snippets, lang)
    return {"answer": ans, "sources": [s["url"] for s in snippets]}
