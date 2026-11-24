import os, base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from companion import VoiceCompanion

app = FastAPI(title="k_rajput Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

companion = VoiceCompanion()
gemini_stt = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
print("Cloud Backend ready!")

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health_check():
    return {"status": "online"}

def transcribe_via_gemini(audio_bytes: bytes, mime: str = "audio/webm") -> str:
    res = gemini_stt.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            "Transcribe the exact spoken words in Marathi, Hindi, or English. Output ONLY the transcript text, nothing else. If silent, output NONE.",
            types.Part.from_bytes(data=audio_bytes, mime_type=mime)
        ]
    )
    txt = (res.text or "").strip()
    return "" if txt.upper() == "NONE" else txt

@app.post("/api/transcribe-song")
async def transcribe_song(audio: UploadFile = File(...), language: str = Form("auto")):
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        return {"status": "empty", "lyrics": ""}
    lyrics = transcribe_via_gemini(audio_bytes, audio.content_type or "audio/webm")
    return {"status": "ok", "lyrics": lyrics, "detected_language": language}

@app.post("/api/companion/voice")
async def companion_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    user_text = transcribe_via_gemini(audio_bytes, audio.content_type or "audio/webm")
    if not user_text:
        return {"status": "no_speech"}
    ai_out = await companion.chat_and_speak(user_text)
    audio_b64 = base64.b64encode(ai_out["audio_bytes"]).decode("utf-8")
    return {"status": "ok", "user_transcript": user_text, "detected_language": "auto", "reply": ai_out["reply"], "audio_base64": audio_b64}

@app.post("/api/companion/text")
async def companion_text(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    ai_out = await companion.chat_and_speak(req.message)
    audio_b64 = base64.b64encode(ai_out["audio_bytes"]).decode("utf-8")
    return {"status": "ok", "reply": ai_out["reply"], "audio_base64": audio_b64}
