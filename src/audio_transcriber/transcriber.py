"""
Audio Transcriber Module
Student 2 — Responsibility: Transcribe audio (911 calls, witness statements) and extract
structured incident information using Whisper + GPT.
"""

import os
import json
import uuid
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPPORTED_AUDIO = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

FIELD_EXTRACTION_PROMPT = """
You are an AI assistant helping law enforcement extract information from transcribed audio.
The audio may be a 911 call, witness statement, officer field report, or interview.

Given the transcript below, extract structured incident information and return ONLY valid JSON:

Fields:
- date (YYYY-MM-DD if mentioned, else null)
- time (HH:MM 24h if mentioned, else null)
- location (any address, intersection, or place name mentioned)
- incident_type (e.g. Robbery, Assault, Fire, Medical Emergency, etc.)
- description (2-3 sentence summary of what happened)
- suspects (descriptions/names mentioned — list, empty if none)
- victims (names/descriptions — list, empty if none)
- evidence (any physical evidence mentioned — list)
- officer (any officer name mentioned, else null)
- caller_type (911_caller | witness | officer | victim | unknown)
- urgency (low | medium | high | critical)
- status (open | closed | under_investigation)
- confidence_score (float 0.0-1.0)

Return ONLY the JSON. No markdown or explanation.

Transcript:
{transcript}
"""


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Transcribed text string
    """
    from pathlib import Path
    ext = Path(audio_path).suffix.lower()
    if ext not in SUPPORTED_AUDIO:
        print(f"[AUDIO] Unsupported format: {ext}")
        return ""

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="text",
            )
        return response.strip()
    except Exception as e:
        print(f"[AUDIO] Whisper transcription error: {e}")
        return ""


def transcribe_audio_local(audio_path: str) -> str:
    """
    Fallback: transcribe using locally installed whisper package.
    Use when OpenAI API is unavailable.
    """
    try:
        import whisper
        print("[AUDIO] Using local Whisper model (base)...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except ImportError:
        print("[AUDIO] Local whisper not installed. Run: pip install openai-whisper")
        return ""
    except Exception as e:
        print(f"[AUDIO] Local Whisper error: {e}")
        return ""


def extract_fields_from_transcript(transcript: str) -> dict:
    """Use GPT to parse structured incident fields from transcript text."""
    if not transcript.strip():
        return _empty_record()

    prompt = FIELD_EXTRACTION_PROMPT.format(transcript=transcript[:3000])

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[AUDIO] JSON parse error: {e}")
        return _empty_record()
    except Exception as e:
        print(f"[AUDIO] LLM extraction error: {e}")
        return _empty_record()


def process_audio(audio_path: str, use_local_whisper: bool = False) -> dict:
    """
    Full pipeline: Audio file → Whisper transcript → LLM field extraction → structured record.

    Args:
        audio_path: Path to the audio file
        use_local_whisper: If True, uses local whisper package instead of API

    Returns:
        Structured incident dict
    """
    print(f"[AUDIO] Processing: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"[AUDIO] File not found: {audio_path}")
        return _empty_record()

    # Step 1: Transcribe
    if use_local_whisper:
        transcript = transcribe_audio_local(audio_path)
    else:
        transcript = transcribe_audio(audio_path)

    if not transcript:
        print("[AUDIO] Empty transcript — trying local fallback")
        transcript = transcribe_audio_local(audio_path)

    print(f"[AUDIO] Transcript ({len(transcript)} chars): {transcript[:100]}...")

    # Step 2: Extract fields
    record = extract_fields_from_transcript(transcript)

    # Step 3: Add metadata
    record["incident_id"] = f"INC-AUD-{str(uuid.uuid4())[:8].upper()}"
    record["source_modality"] = "audio"
    record["source_file"] = os.path.basename(audio_path)
    record["processed_at"] = datetime.now().isoformat()
    record["transcript"] = transcript

    print(f"[AUDIO] Done — incident_type: {record.get('incident_type', 'Unknown')}")
    return record


def _empty_record() -> dict:
    return {
        "incident_id": f"INC-AUD-{str(uuid.uuid4())[:8].upper()}",
        "source_modality": "audio",
        "date": None,
        "time": None,
        "location": None,
        "incident_type": "Unknown",
        "description": "Transcription/extraction failed",
        "suspects": [],
        "victims": [],
        "evidence": [],
        "officer": None,
        "caller_type": "unknown",
        "urgency": "unknown",
        "status": "open",
        "confidence_score": 0.0,
        "transcript": "",
    }


# ── Demo / standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <path_to_audio>")
        sys.exit(1)
    result = process_audio(sys.argv[1])
    print(json.dumps(result, indent=2))
