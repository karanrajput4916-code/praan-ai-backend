import os
import io
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

class SongTranscriber:
    def __init__(self):
        # Fast CPU/GPU inference with int8 quantization
        self.model = WhisperModel(MODEL_SIZE, device="auto", compute_type="int8")

    def transcribe_chunk(self, audio_bytes: bytes, language_hint: str = None) -> dict:
        """
        Converts audio bytes into script-preserved lyrics (Marathi/Hindi/English)
        and marks uncertain words with [word?] instead of hallucinating.
        """
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            segments, info = self.model.transcribe(
                audio_buffer,
                language=None if language_hint == "auto" else language_hint,
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=True,
                initial_prompt="Lyrics transcription in original script (मराठी, हिन्दी, English)."
            )

            lines = []
            for segment in segments:
                if segment.no_speech_prob > 0.75:
                    continue

                formatted_words = []
                if segment.words:
                    for w in segment.words:
                        word_text = w.word.strip()
                        if not word_text:
                            continue
                        if w.probability < 0.42:
                            formatted_words.append(f"[{word_text}?]")
                        else:
                            formatted_words.append(word_text)
                    line = " ".join(formatted_words)
                else:
                    line = segment.text.strip()

                if line:
                    lines.append(line)

            return {
                "status": "ok",
                "detected_language": info.language,
                "language_probability": round(info.language_probability, 2),
                "lyrics": "\n".join(lines)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "lyrics": ""
            }