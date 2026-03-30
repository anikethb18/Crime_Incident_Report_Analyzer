# Presentation Script — Multimodal Crime / Incident Report Analyzer
**COMP 4XXX — Final Assignment | Group of 2**

---

## [0:00 – 0:30] Introduction

"Hello. We are a group of two students presenting our final assignment:
the Multimodal Crime / Incident Report Analyzer. Per the assignment guidelines,
groups smaller than three may select any three modality types. We chose PDF,
Audio, and Text NLP — the three modalities that allow us to demonstrate the
richest set of distinct AI models."

---

## [0:30 – 1:30] Problem and Architecture

"Law enforcement agencies deal with incident data in many unstructured forms —
scanned police reports, 911 call recordings, witness statements, and anonymous
tips. Our system takes all of these and produces one clean, structured dataset.

Here is our AI pipeline architecture [show architecture_diagram.svg]:

- **Modality 1 — PDF**: pdfplumber extracts text from digital PDFs.
  For scanned documents, we detect low text yield and switch to a Tesseract OCR
  pipeline: pdf2image converts each page to a 300 DPI image, Pillow pre-processes
  it with greyscale and binarisation, and pytesseract extracts the text.
  GPT-4o then parses the raw text into a structured JSON record.

- **Modality 2 — Audio**: OpenAI Whisper (whisper-1) transcribes 911 calls
  and field recordings. A local whisper.base model is available as an offline
  fallback. GPT-4o then extracts fields: date, time, location, suspects, urgency.

- **Modality 3 — Text NLP**: Three distinct AI steps run in sequence.
  First, spaCy en_core_web_sm performs Named Entity Recognition, extracting
  persons, locations, dates, and organisations. Second, a DistilBERT transformer
  from HuggingFace runs sentiment analysis and maps it to a crime-relevant tone:
  high_distress, distressed, neutral, or calm. Third, a keyword scorer classifies
  the text into one of ten crime categories. All three signals are passed to
  GPT-4o for the final structured extraction."

---

## [1:30 – 2:30] Integration and Dataset

"The Integrator fuses outputs from all three modalities into one dataset.
Each record gets an explicit Incident_ID. Records with the same date,
incident type, and overlapping location keywords are grouped together —
so one real-world assault can be described by a PDF report, a 911 audio
call, and a witness text statement, all in a single row.

The integrator then:
- Deduplicates and merges suspects, victims, and evidence across modalities
- Computes a combined_severity from cross-modality signals:
  audio urgency + text sentiment + PDF confidence
- Outputs structured_incidents_latest.csv and .json

[Show the CSV — point to key columns: incident_id, sources_present, combined_severity]

In our demo dataset, we have four incidents:
- A Burglary with PDF only
- An Assault with PDF + Audio + Text — severity correctly computed as HIGH
- A Vandalism with PDF + Audio
- A Drug complaint with Text only"

---

## [2:30 – 3:15] Live Demo

"Let me show the pipeline running.

[Run in terminal:]
  python run_pipeline.py --demo

[Point to output summary showing 4 rows, sources, severity]

Now let me launch the dashboard:
  streamlit run dashboard/app.py

[Show dashboard — walk through:]
- KPI cards at the top: total incidents, open cases, modalities, avg confidence
- Incident type bar chart — 4 types visible
- Modality pie chart — audio / pdf / text
- Select the Assault incident in the detail view
- Show the three modality tabs: PDF, Audio, Text NLP
- Show the transcript, NER persons, topic scores
- Download filtered CSV button"

---

## [3:15 – 3:45] Results and Reflections

"Our system successfully converts unstructured incident data — PDF reports,
audio recordings, and text statements — into a structured 32-column dataset
ready for analysis.

Key AI models used:
- pdfplumber + Tesseract (real OCR, not just GPT)
- OpenAI Whisper (real speech-to-text)
- spaCy NER (real NLP entity extraction)
- DistilBERT (real transformer sentiment model)
- GPT-4o (LLM reasoning layer on top of traditional models)

The combined approach means each modality uses purpose-built AI for its
specific data type, with GPT-4o serving as the final structured extraction
layer — not the only model.

Thank you."

---

## Notes for Presenter

- Total target length: 3–4 minutes
- Show terminal output for `--demo` run
- Show at least 2 charts in the dashboard
- Show the detail panel for the Assault incident (it has all 3 modalities)
- Point out `sources_present = audio; pdf; text` and `combined_severity = high`
