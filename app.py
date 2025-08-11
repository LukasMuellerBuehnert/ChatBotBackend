from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS für deine GitHub Pages URL erlauben
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lukasmuellerbuehnert.github.io"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
def chat(payload: dict):
    msg = payload.get("message", "").strip().lower()
    if msg == "hallo":
        return {"answer": "Hallo, wie geht es dir?"}
    else:
        return {"answer": f"Ich habe '{msg}' gehört, aber kenne darauf noch keine Antwort."}
