import os, io, edge_tts
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat_history = []

SYSTEM_PROMPT = """You are Praan (प्राण), Karan's warm, witty, supportive personal AI voice companion. Reply naturally in the exact language the user uses (Marathi, Hindi, or English). Keep responses concise and spoken-friendly (2-3 short sentences max, no markdown or asterisks)."""

def detect_voice(text: str) -> str:
    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            if any(w in text for w in ['आहे', 'काय', 'मी', 'तुला', 'मराठी', 'कसं', 'नाही', 'हो', 'चल', 'बोल']):
                return 'mr-IN-AarohiNeural'
            return 'hi-IN-SwaraNeural'
    return 'en-IN-NeerjaNeural'

def get_working_models():
    try:
        avail = [m.name.replace('models/', '') for m in client.models.list() if 'flash' in m.name and 'lite' not in m.name and 'image' not in m.name and 'audio' not in m.name]
        return avail[:4] if avail else ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    except Exception:
        return ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]

async def chat_and_speak(user_text: str) -> dict:
    global chat_history
    chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    if len(chat_history) > 12:
        chat_history = chat_history[-12:]
    reply = None
    last_err = ""
    for m in get_working_models():
        try:
            res = client.models.generate_content(
                model=m,
                contents=chat_history,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.7)
            )
            if res.text:
                reply = res.text.strip().replace("*", "")
                break
        except Exception as e:
            last_err = str(e)
            continue
    if not reply:
        reply = f"API अडचण आली आहे: {last_err[:120]}"
    chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
    try:
        voice = detect_voice(reply)
        communicate = edge_tts.Communicate(reply, voice)
        audio_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buf.write(chunk["data"])
        audio_bytes = audio_buf.getvalue()
    except Exception:
        audio_bytes = b""
    return {"reply": reply, "audio_bytes": audio_bytes}
