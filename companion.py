import os
import io
import re
from google import genai
from google.genai import types
import edge_tts
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are k_rajput, a warm, witty, emotionally intelligent personal AI friend.
Rules:
1. Detect the user's language automatically (Marathi, Hindi, English, or Hinglish/Marathinglish) and reply naturally in the SAME language and script.
2. Keep spoken responses concise (2-4 sentences unless explaining a complex technical topic or telling a story/shayari).
3. Never use markdown asterisks (*), bullet symbols, or emojis in your responses because your text will be read aloud by a Text-to-Speech engine.
4. Be friendly, supportive, funny when appropriate, and genuinely helpful for coding, studies, or casual time-pass chats.
"""

VOICE_MAP = {
    "mr": "mr-IN-AarohiNeural",
    "hi": "hi-IN-k_rajputNeural",
    "en": "en-IN-NeerjaNeural",
}

MODELS_TO_TRY = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

class VoiceCompanion:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in backend/.env")
        self.client = genai.Client(api_key=api_key)
        self.history = []

    def reset_memory(self):
        self.history = []

    def _detect_tts_voice(self, text: str) -> str:
        if re.search(r'[\u0900-\u097F]', text):
            marathi_markers = ["आहे", "आहेत", "नाही", "काय", "कसं", "तुला", "मला", "हो", "ण", "ळ"]
            if any(m in text for m in marathi_markers):
                return VOICE_MAP["mr"]
            return VOICE_MAP["hi"]
        return VOICE_MAP["en"]

    async def chat_and_speak(self, user_text: str) -> dict:
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
        
        reply_text = None
        last_error = None

        for model_name in MODELS_TO_TRY:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=self.history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.8,
                    )
                )
                reply_text = response.text.strip()
                break
            except Exception as e:
                last_error = e
                continue

        if not reply_text:
            self.history.pop()
            raise RuntimeError(f"All Gemini models busy: {last_error}")

        self.history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
        clean_speech = re.sub(r'[*#_`~]', '', reply_text)

        voice = self._detect_tts_voice(clean_speech)
        communicate = edge_tts.Communicate(clean_speech, voice)

        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])

        return {
            "reply": reply_text,
            "voice_used": voice,
            "audio_bytes": audio_stream.getvalue()
        }

