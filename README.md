# 🔍 Multimodal Crime / Incident Report Analyzer

**Final Assignment | Group of 2**

An end-to-end AI pipeline that ingests unstructured data from **3 modalities** and produces
a unified, structured incident report dataset with a web dashboard for filtering and querying.

---

## 👥 Group Size & Modality Selection

> **Team size: 2 students.**
> Per assignment guidelines (section 7): *"If your group size is less than 3 members, you can work on any 3 types of roles."*
> We implement **PDF, Audio, and Text NLP** — three distinct modalities each requiring its own
> dedicated AI model stack (Tesseract OCR, Whisper, spaCy + DistilBERT).
> All five required deliverables are included: architecture diagram, GitHub-ready code,
> structured dataset, project report, and demonstration.

| Student | Modalities | Role |
|---------|-----------|------|
| Aniketh Bharat | PDF, Audio | PDF extraction + OCR pipeline, Whisper transcription, architecture diagram |
| Sree Vardhan Sunkara | Text NLP, Integration, Dashboard | spaCy NER, DistilBERT sentiment, dataset fusion, Streamlit dashboard |

---

## 📦 Project Structure

```
crime_analyzer/
├── src/
│   ├── pdf_processor/
│   │   └── processor.py        # Modality 1 — pdfplumber + Tesseract OCR + GPT-4o
│   ├── audio_transcriber/
│   │   └── transcriber.py      # Modality 2 — OpenAI Whisper + GPT-4o
│   ├── text_nlp/
│   │   └── analyzer.py         # Modality 3 — spaCy NER + DistilBERT + GPT-4o
│   └── integrator/
│       └── merge.py            # Cross-modality fusion → CSV + JSON dataset
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── data/
│   ├── samples/
│   │   ├── pdf_samples/        # Sample police report text files
│   │   ├── audio_samples/      # Audio input files (.mp3/.wav) + README
│   │   └── text_samples/       # Witness statements, 911 transcripts, tipline messages
│   └── outputs/
│       ├── structured_incidents_latest.csv   ← Final structured dataset (pre-generated)
│       └── structured_incidents_latest.json  ← JSON version (pre-generated)
├── docs/
│   ├── project_report.docx     # Full written report
│   ├── architecture_diagram.svg
│   └── presentation_script.md
├── notebooks/
│   └── demo_pipeline.ipynb     # Jupyter walkthrough (8 cells)
├── run_pipeline.py             # ← MAIN ENTRY POINT
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install system dependencies (for PDF OCR pipeline)
```bash
# Linux / Ubuntu / Google Colab
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Configure API key
```bash
cp .env.example .env
# Open .env and set: OPENAI_API_KEY=your_key_here
```

> **No API key?** Run `--demo` mode below — uses pre-built records, no API calls needed.

---

## 🎬 Step-by-Step Demo

Follow these exact steps to demonstrate the full working pipeline.

### Step 1 — Run the pipeline (no API key needed)

```bash
cd crime_analyzer
python run_pipeline.py --demo
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║   🔍  Multimodal Crime / Incident Report Analyzer            ║
║   3 Modalities: PDF · Audio · Text (NLP)                    ║
╚══════════════════════════════════════════════════════════════╝

[DEMO] Running with 7 pre-built records across 3 modalities.
[MERGE] Fusing 7 modality record(s) → incident rows...
[MERGE] Result: 4 fused incident row(s)
[SAVE] CSV  → data/outputs/fused_incidents_<timestamp>.csv
[SAVE] JSON → data/outputs/fused_incidents_<timestamp>.json

  3-MODALITY FUSION DATASET — INTEGRATION SUMMARY
  Total Incidents (rows)  : 4
  Modalities in dataset   : audio, pdf, text

  ── Combined Severity ──
    high       ██░░░  1     ← Assault (audio urgency=high + text sentiment=high_distress)
    medium     ██░░░  2     ← Burglary (PDF) + Vandalism (PDF + Audio)
    low        █░░░░  1     ← Drug tip (Text only)
```

### Step 2 — Inspect the structured dataset

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/outputs/structured_incidents_latest.csv')
print(df[['incident_id','date','incident_type','combined_severity','sources_present']].to_string())
"
```

**Expected output:**

| incident_id | date | incident_type | combined_severity | sources_present |
|-------------|------|---------------|-------------------|-----------------|
| INC-XXXXXX | 2024-11-19 | Vandalism | medium | audio; pdf |
| INC-XXXXXX | 2024-11-20 | Burglary  | medium | pdf |
| INC-XXXXXX | 2024-11-20 | Assault   | **high** | audio; pdf; text |
| INC-XXXXXX | 2024-11-21 | Drug      | low | text |

### Step 3 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. You will see:

- **KPI row**: 4 total incidents · 3 open · 3 modalities · 83% avg confidence
- **Charts**: incident type bar, modality pie, confidence histogram, severity bar
- **Records table**: all 4 fused incidents
- **Detail panel**: select the **Assault** incident
  - PDF tab → officer report, evidence list, extraction method
  - Audio tab → 911 call transcript, urgency=high, caller_type=911_caller
  - Text NLP tab → spaCy persons/locations, DistilBERT sentiment=high_distress (0.95), topic scores
- **Export button**: download filtered CSV

### Step 4 — Run with real files (requires API key)

```bash
# Single modality
python run_pipeline.py --pdf  data/samples/pdf_samples/HPD_report_burglary.txt
python run_pipeline.py --text data/samples/text_samples/witness_statement_assault.txt
python run_pipeline.py --audio your_911_call.mp3

# All three at once
python run_pipeline.py \
  --pdf  data/samples/pdf_samples/HPD_report_burglary.txt \
  --text data/samples/text_samples/witness_statement_assault.txt \
  --audio your_call.mp3

# Audio without API key (local Whisper model)
python run_pipeline.py --audio your_call.mp3 --local-whisper
```

### Step 5 — Jupyter notebook walkthrough

```bash
jupyter notebook notebooks/demo_pipeline.ipynb
```

Run all 8 cells in order. Cell 5 runs spaCy NER + DistilBERT sentiment live on sample
witness statement text — no API key required.

---

## 🧠 AI Models Used Per Modality

### Modality 1 — PDF Processor (`src/pdf_processor/processor.py`)
| Step | Model / Tool | Category |
|------|-------------|----------|
| Text extraction | `pdfplumber` | Traditional PDF parser |
| Scanned PDF detection | Heuristic (avg chars/page < 50) | Rule-based |
| Page-to-image | `pdf2image` (Poppler) | Traditional |
| Image pre-processing | `Pillow` (greyscale → binarise → sharpen) | Traditional CV |
| OCR | `Tesseract` via `pytesseract` (PSM 6) | Traditional OCR engine |
| Structured extraction | GPT-4o (temperature=0.1, JSON-only) | LLM |

### Modality 2 — Audio Transcriber (`src/audio_transcriber/transcriber.py`)
| Step | Model / Tool | Category |
|------|-------------|----------|
| Speech-to-text (API) | OpenAI Whisper (`whisper-1`) | Deep learning ASR |
| Speech-to-text (local) | `whisper.load_model("base")` | Deep learning ASR (offline) |
| Structured extraction | GPT-4o (temperature=0.1, JSON-only) | LLM |

### Modality 3 — Text NLP Analyzer (`src/text_nlp/analyzer.py`)
| Step | Model / Tool | Category |
|------|-------------|----------|
| Named Entity Recognition | `spaCy` `en_core_web_sm` | Traditional NLP |
| Sentiment analysis | `DistilBERT` (HuggingFace transformers) | Transformer model |
| Topic classification | Keyword scoring across 10 crime categories | Rule-based NLP |
| Structured extraction | GPT-4o (NER + sentiment + topic as context) | LLM |

### Integrator (`src/integrator/merge.py`)
- Assigns explicit `incident_id` to every record (date + type + location matching)
- Pivots per-modality records → **one unified row per real-world incident**
- Computes `combined_severity` from cross-modality signals (audio urgency + text sentiment + PDF confidence)
- Exports timestamped + latest CSV and JSON

---

## 📊 Output Dataset Schema

| Column | Description |
|--------|-------------|
| `incident_id` | Shared ID across modalities (`INC-XXXXXX`) |
| `date` / `time` | Incident date (YYYY-MM-DD) and time (HH:MM) |
| `location` | Full address or scene description |
| `incident_type` | Burglary / Assault / Vandalism / Drug / etc. |
| `combined_description` | Merged narrative from all available modalities |
| `all_suspects` / `all_victims` | Deduplicated and merged across all modalities |
| `all_evidence` | Deduplicated evidence from all modalities |
| `combined_severity` | low / medium / high — fused from audio + text + PDF signals |
| `combined_confidence` | Average confidence score across active modalities |
| `sources_present` | Which modalities contributed (e.g. `audio; pdf; text`) |
| `pdf_*` | PDF-specific columns (extraction method, officer, evidence) |
| `audio_*` | Audio-specific columns (transcript, urgency, caller type) |
| `text_*` | Text NLP columns (sentiment, NER entities, topic scores) |

---

## 📁 Sample Files

| File | Simulates |
|------|-----------| 
| `pdf_samples/HPD_report_burglary.txt` | Police report — Burglary (text-based PDF) |
| `pdf_samples/HPD_report_vandalism.txt` | Police report — Vandalism (OCR path demo) |
| `text_samples/witness_statement_assault.txt` | Witness statement — Assault |
| `text_samples/911_transcript_assault.txt` | 911 call transcript — Assault |
| `text_samples/tipline_drug_complaint.txt` | Anonymous tipline message — Drug |
| `audio_samples/README.txt` | Instructions for adding real .mp3/.wav audio files |

---

## 🖥️ Dashboard Features (`dashboard/app.py`)

- **Sidebar filters**: severity, incident type, status, modality sources, confidence, text search
- **KPI cards**: total incidents, open cases, active modalities, avg confidence
- **Charts**: incident types (bar), modality distribution (pie), confidence (histogram), severity (bar)
- **Records table**: all fused incidents, sortable
- **Detail panel**: per-incident view with PDF / Audio / Text NLP tabs
- **Export**: download filtered results as CSV
