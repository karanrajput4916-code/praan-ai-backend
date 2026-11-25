import os, io, time, edge_tts
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat_history = []
MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash"]

SYSTEM_PROMPT = """You are Praan (प्राण), Karan's warm, witty, supportive personal AI voice companion. Reply naturally in the exact language the user uses (Marathi, Hindi, or English—or natural Hinglish/Marathi-English mix). Keep responses concise, conversational, and spoken-friendly (2-3 short sentences max, no markdown symbols, asterisks, or bullet points since your reply is read aloud via TTS)."""

def detect_voice(text: str) -> str:
    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            if any(w in text for w in ['आहे', 'काय', 'मी', 'तुला', 'मराठी', 'कसं', 'नाही', 'हो', 'चल', 'बोल', 'भाऊ']):
                return 'mr-IN-AarohiNeural'
            return 'hi-IN-SwaraNeural'
    return 'en-IN-NeerjaNeural'

async def chat_and_speak(user_text: str) -> dict:
    global chat_history
    chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    if len(chat_history) > 12:
        chat_history = chat_history[-12:]
    reply = None
    for m in MODELS:
        try:
            res = client.models.generate_content(
                model=m,
                contents=chat_history,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.7)
            )
            if res.text:
                reply = res.text.strip().replace("*", "")
                break
        except Exception:
            time.sleep(0.5)
            continue
    if not reply:
        reply = "सध्या सर्व्हरवर थोडी गर्दी आहे करण, एकदा पुन्हा सांगशील का?"
    chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
    voice = detect_voice(reply)
    communicate = edge_tts.Communicate(reply, voice)
    audio_buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buf.write(chunk["data"])
    return {"reply": reply, "audio_bytes": audio_buf.getvalue()}
