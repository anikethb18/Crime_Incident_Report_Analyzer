"""
Text NLP Module
Student 5 — Responsibility: Analyze unstructured text (witness statements, police notes,
social media tips, free-text reports) using NLP techniques.

Pipeline:
  1. Named Entity Recognition (NER) — extract persons, locations, dates, organizations
  2. Sentiment analysis — determine emotional tone (distress, calm, anger)
  3. Topic classification — categorize into incident types
  4. GPT-4o structured extraction — produce unified incident record
"""

import os
import re
import json
import uuid
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPPORTED_TEXT = {".txt", ".md", ".csv", ".json", ".log"}

# ── spaCy NER ─────────────────────────────────────────────────────────────────

def run_ner(text: str) -> dict:
    """
    Run Named Entity Recognition using spaCy (en_core_web_sm).
    Extracts: PERSON, GPE (geopolitical), LOC, ORG, DATE, TIME, CARDINAL.

    Falls back to regex-based extraction if spaCy is unavailable.
    """
    entities = {
        "persons": [],
        "locations": [],
        "organizations": [],
        "dates": [],
        "times": [],
        "other": [],
    }

    try:
        import spacy
        # Load small English model (install: python -m spacy download en_core_web_sm)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[NLP] spaCy model not found. Run: python -m spacy download en_core_web_sm")
            return _regex_fallback_ner(text)

        doc = nlp(text[:10000])  # Limit for performance
        for ent in doc.ents:
            label = ent.label_
            val = ent.text.strip()
            if label == "PERSON":
                if val not in entities["persons"]:
                    entities["persons"].append(val)
            elif label in ("GPE", "LOC", "FAC"):
                if val not in entities["locations"]:
                    entities["locations"].append(val)
            elif label == "ORG":
                if val not in entities["organizations"]:
                    entities["organizations"].append(val)
            elif label == "DATE":
                if val not in entities["dates"]:
                    entities["dates"].append(val)
            elif label == "TIME":
                if val not in entities["times"]:
                    entities["times"].append(val)
            else:
                entities["other"].append(f"{label}: {val}")

        print(f"[NLP] NER complete (spaCy): {sum(len(v) for v in entities.values())} entities")
    except ImportError:
        print("[NLP] spaCy not installed. Run: pip install spacy")
        entities = _regex_fallback_ner(text)

    return entities


def _regex_fallback_ner(text: str) -> dict:
    """Simple regex-based fallback NER when spaCy is unavailable."""
    import re
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'

    dates = re.findall(date_pattern, text, re.IGNORECASE)
    times = re.findall(time_pattern, text)

    print(f"[NLP] NER fallback (regex): {len(dates)} dates, {len(times)} times found")
    return {
        "persons": [],
        "locations": [],
        "organizations": [],
        "dates": list(set(dates)),
        "times": list(set(times)),
        "other": [],
    }


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def run_sentiment(text: str) -> dict:
    """
    Run sentiment analysis using HuggingFace transformers pipeline.
    Uses distilbert-base-uncased-finetuned-sst-2-english (fast, no API cost).

    Falls back to simple lexicon-based approach if transformers unavailable.
    """
    try:
        from transformers import pipeline
        # Use a small, fast model
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        # Chunk long texts
        chunk = text[:500]
        result = sentiment_pipeline(chunk)[0]
        label = result["label"]  # POSITIVE / NEGATIVE
        score = round(result["score"], 3)

        # Map to crime context labels
        if label == "NEGATIVE" and score > 0.9:
            tone = "high_distress"
        elif label == "NEGATIVE":
            tone = "distressed"
        elif label == "POSITIVE" and score > 0.9:
            tone = "calm"
        else:
            tone = "neutral"

        print(f"[NLP] Sentiment: {label} ({score:.2f}) → tone: {tone}")
        return {"label": label, "score": score, "tone": tone}

    except ImportError:
        print("[NLP] transformers not installed. Run: pip install transformers torch")
        return _lexicon_sentiment(text)
    except Exception as e:
        print(f"[NLP] Sentiment error: {e}")
        return _lexicon_sentiment(text)


def _lexicon_sentiment(text: str) -> dict:
    """Simple keyword-based sentiment fallback."""
    high_distress = ["help", "emergency", "scared", "bleeding", "attack", "weapon", "shot", "dying", "please hurry"]
    distress_words = ["hurt", "afraid", "upset", "angry", "dangerous", "threatening", "robbery", "fight"]
    calm_words = ["report", "notify", "inform", "statement", "occurred", "observed"]

    text_lower = text.lower()
    hd_count = sum(1 for w in high_distress if w in text_lower)
    d_count = sum(1 for w in distress_words if w in text_lower)
    c_count = sum(1 for w in calm_words if w in text_lower)

    if hd_count >= 2:
        tone, label, score = "high_distress", "NEGATIVE", 0.95
    elif d_count > c_count:
        tone, label, score = "distressed", "NEGATIVE", 0.75
    elif c_count > 0:
        tone, label, score = "neutral", "POSITIVE", 0.65
    else:
        tone, label, score = "neutral", "POSITIVE", 0.55

    print(f"[NLP] Sentiment (lexicon): {tone}")
    return {"label": label, "score": score, "tone": tone}


# ── Topic Classification ──────────────────────────────────────────────────────

INCIDENT_KEYWORDS = {
    "Robbery": ["robbed", "robbery", "stole", "theft", "took", "money", "cash", "gun", "weapon", "demand"],
    "Assault": ["hit", "punched", "attacked", "assaulted", "fight", "violence", "stabbed", "beat"],
    "Burglary": ["broke in", "break-in", "breaking in", "burglar", "window", "door", "enter", "trespass"],
    "Vandalism": ["spray paint", "graffiti", "damaged", "broken", "smashed", "destroyed", "keyed"],
    "Drug": ["drugs", "narcotics", "dealer", "dealing", "cocaine", "heroin", "marijuana", "suspicious package"],
    "Fraud": ["scam", "fraud", "fake", "impersonate", "phishing", "identity", "credit card"],
    "Homicide": ["killed", "murder", "dead", "body", "shot", "stabbed to death", "homicide"],
    "Missing Person": ["missing", "disappeared", "last seen", "whereabouts", "runaway"],
    "Traffic": ["accident", "crash", "collision", "drunk driving", "hit and run", "speeding"],
    "Disturbance": ["noise", "argument", "dispute", "domestic", "shouting", "disturbance"],
}


def classify_topic(text: str) -> dict:
    """
    Classify text into incident type categories using keyword matching.
    Returns top match and scores for all categories.
    """
    text_lower = text.lower()
    scores = {}
    for category, keywords in INCIDENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score

    top_category = max(scores, key=scores.get) if scores else "Unknown"
    top_score = scores.get(top_category, 0)

    if top_score == 0:
        top_category = "Unknown"

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"[NLP] Topic classification: {top_category} (score={top_score})")
    return {
        "primary_topic": top_category,
        "keyword_score": top_score,
        "all_scores": dict(sorted_scores[:5]),  # Top 5 categories
    }


# ── GPT-4o Structured Extraction ──────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are an AI assistant helping law enforcement extract structured information from
unstructured text. The text may be a witness statement, tipline message, social media post,
officer field note, or written report.

Given the text and pre-computed NLP analysis below, return ONLY valid JSON:

- date (YYYY-MM-DD if mentioned, else null)
- time (HH:MM 24h if mentioned, else null)
- location (most specific address or place name mentioned)
- incident_type (the primary incident type)
- description (2-3 sentence summary)
- suspects (names/descriptions — list, empty if none)
- victims (names/descriptions — list, empty if none)
- evidence (physical or digital evidence mentioned — list)
- officer (any officer name mentioned, else null)
- urgency (low | medium | high | critical)
- status (open | closed | under_investigation)
- confidence_score (float 0.0-1.0)

Return ONLY the JSON. No markdown.

Text:
{text}

Pre-computed NLP:
- Named entities: {entities}
- Sentiment: {sentiment}
- Topic classification: {topic}
"""


def extract_with_llm(text: str, entities: dict, sentiment: dict, topic: dict) -> dict:
    """Use GPT-4o to produce final structured record from text + NLP signals."""
    prompt = EXTRACTION_PROMPT.format(
        text=text[:3000],
        entities=json.dumps(entities),
        sentiment=json.dumps(sentiment),
        topic=json.dumps(topic),
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"[NLP] LLM extraction error: {e}")
        return {}


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_text(text_source) -> str:
    """Load text from a file path or return the string directly."""
    if isinstance(text_source, str) and os.path.exists(text_source):
        ext = Path(text_source).suffix.lower() if hasattr(text_source, "__len__") else ""
        with open(text_source, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif isinstance(text_source, str):
        return text_source  # Raw text string passed directly
    return ""


from pathlib import Path


def process_text(text_source, source_label: str = None) -> dict:
    """
    Full NLP pipeline: text → NER + sentiment + topic → LLM extraction → structured record.

    Args:
        text_source: File path (str) OR raw text string
        source_label: Optional label for the source file name

    Returns:
        Structured incident dict
    """
    is_file = isinstance(text_source, str) and os.path.exists(text_source)
    source_file = os.path.basename(text_source) if is_file else (source_label or "raw_text_input")
    print(f"[NLP] Processing: {source_file}")

    text = load_text(text_source)
    if not text.strip():
        print("[NLP] Empty text input")
        return _empty_record(source_file)

    print(f"[NLP] Text length: {len(text)} chars")

    # Step 1: NER
    entities = run_ner(text)

    # Step 2: Sentiment
    sentiment = run_sentiment(text)

    # Step 3: Topic classification
    topic = classify_topic(text)

    # Step 4: LLM structured extraction
    extracted = extract_with_llm(text, entities, sentiment, topic)

    # Step 5: Build final record
    record = {
        "incident_id": f"INC-TXT-{str(uuid.uuid4())[:8].upper()}",
        "source_modality": "text",
        "source_file": source_file,
        "processed_at": datetime.now().isoformat(),
        "date": extracted.get("date"),
        "time": extracted.get("time"),
        "location": extracted.get("location") or (entities["locations"][0] if entities["locations"] else None),
        "incident_type": extracted.get("incident_type") or topic["primary_topic"],
        "description": extracted.get("description", ""),
        "suspects": extracted.get("suspects", []),
        "victims": extracted.get("victims", []),
        "evidence": extracted.get("evidence", []),
        "officer": extracted.get("officer"),
        "status": extracted.get("status", "open"),
        "confidence_score": extracted.get("confidence_score", 0.5),
        # NLP-specific fields
        "sentiment_tone": sentiment.get("tone", "unknown"),
        "sentiment_score": sentiment.get("score", 0.0),
        "ner_persons": entities.get("persons", []),
        "ner_locations": entities.get("locations", []),
        "ner_dates": entities.get("dates", []),
        "topic_scores": topic.get("all_scores", {}),
        "urgency": extracted.get("urgency", "medium"),
        "raw_text_snippet": text[:300],
    }

    print(f"[NLP] Done — incident_type: {record['incident_type']}, "
          f"sentiment: {record['sentiment_tone']}, "
          f"NER persons: {len(record['ner_persons'])}")
    return record


def _empty_record(source_file: str = "") -> dict:
    return {
        "incident_id": f"INC-TXT-{str(uuid.uuid4())[:8].upper()}",
        "source_modality": "text",
        "source_file": source_file,
        "processed_at": datetime.now().isoformat(),
        "date": None, "time": None, "location": None,
        "incident_type": "Unknown",
        "description": "NLP processing failed",
        "suspects": [], "victims": [], "evidence": [],
        "officer": None, "status": "open", "confidence_score": 0.0,
        "sentiment_tone": "unknown", "sentiment_score": 0.0,
        "ner_persons": [], "ner_locations": [], "ner_dates": [],
        "topic_scores": {}, "urgency": "unknown", "raw_text_snippet": "",
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        result = process_text(sys.argv[1])
    else:
        # Demo with inline text
        sample = """
        On the night of November 20th at around 11:45 PM, I witnessed two men
        attacking a woman near the bus stop on Westheimer Road. One man was wearing
        a red jacket and was about 6 feet tall. The other was shorter with a white
        hoodie. The woman was screaming for help. They grabbed her purse and ran
        toward the parking garage on Montrose. I called 911 immediately.
        My name is Robert Chen and I live at 4200 Main St.
        """
        result = process_text(sample, source_label="witness_statement_demo.txt")

    display = {k: v for k, v in result.items() if k not in ("raw_text_snippet", "topic_scores")}
    print(json.dumps(display, indent=2))
