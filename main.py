import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transcriber import SongTranscriber
from companion import VoiceCompanion

app = FastAPI(title="k_rajput Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model... (First run downloads model weights)")
transcriber = SongTranscriber()
companion = VoiceCompanion()
print("Backend ready!")

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health_check():
    return {"status": "online"}

@app.post("/api/transcribe-song")
async def transcribe_song(
    audio: UploadFile = File(...),
    language: str = Form("auto")
):
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        return {"status": "empty", "lyrics": ""}
    result = transcriber.transcribe_chunk(audio_bytes, language_hint=language)
    return result

@app.post("/api/companion/voice")
async def companion_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    stt_result = transcriber.transcribe_chunk(audio_bytes, language_hint="auto")
    user_text = stt_result.get("lyrics", "").replace("[", "").replace("?]", "").strip()

    if not user_text:
        return {"status": "no_speech"}

    ai_out = await companion.chat_and_speak(user_text)
    audio_b64 = base64.b64encode(ai_out["audio_bytes"]).decode("utf-8")

    return {
        "status": "ok",
        "user_transcript": user_text,
        "detected_language": stt_result.get("detected_language", "auto"),
        "reply": ai_out["reply"],
        "audio_base64": audio_b64
    }

@app.post("/api/companion/text")
async def companion_text(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    ai_out = await companion.chat_and_speak(req.message)
    audio_b64 = base64.b64encode(ai_out["audio_bytes"]).decode("utf-8")
    return {
        "status": "ok",
        "reply": ai_out["reply"],
        "audio_base64": audio_b64
    }

@app.post("/api/companion/reset")
def reset_companion():
    companion.reset_memory()
    return {"status": "reset"}

