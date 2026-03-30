"""
Dataset Integrator Module — 3-Modality Fusion
Multimodal Crime / Incident Report Analyzer
Group of 2 — COMP 4XXX Final Assignment

Fuses outputs from three AI-powered modalities into one structured dataset:

  Modality 1 — PDF Processor     : pdfplumber text extraction + Tesseract OCR (scanned PDFs)
  Modality 2 — Audio Transcriber : OpenAI Whisper speech-to-text + GPT-4o extraction
  Modality 3 — Text NLP Analyzer : spaCy NER + DistilBERT sentiment + keyword classification

Integration steps:
  1. Assign explicit Incident_ID to every record (date + type + location grouping)
  2. Pivot per-modality records → one unified row per real-world incident
  3. Compute combined_severity from cross-modality signals (audio urgency + text sentiment + PDF confidence)
  4. Export timestamped + latest CSV and JSON outputs

Output schema (one row per incident):
  row_num | incident_id | date | time | location | incident_type |
  combined_description | officer | overall_status |
  all_suspects | all_victims | all_evidence |
  combined_severity | combined_confidence | sources_present |
  pdf_info | pdf_suspects | pdf_evidence | pdf_officer | pdf_extraction_method | pdf_confidence |
  audio_event | audio_transcript | audio_urgency | audio_caller_type | audio_confidence |
  text_crime_type | text_sentiment | text_sentiment_score |
  text_ner_persons | text_ner_locations | text_ner_dates | text_topic_scores | text_confidence |
  processed_at
"""

import os
import re
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import List


# ── Severity / urgency scoring ────────────────────────────────────────────────

URGENCY_SCORES   = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SENTIMENT_SCORES = {"calm": 0, "neutral": 1, "distressed": 2, "high_distress": 3}


def compute_combined_severity(records: List[dict]) -> str:
    """
    Compute a single severity rating by fusing cross-modality signals:
      - Audio  : urgency field (low / medium / high / critical)
      - Text   : sentiment_tone (calm / neutral / distressed / high_distress)
      - PDF    : confidence score as a proxy (high-confidence report → medium signal)

    Returns: low | medium | high | critical
    """
    scores = []
    for r in records:
        m = r.get("source_modality", "")
        if m == "audio":
            urg = str(r.get("urgency", "")).lower()
            if urg in URGENCY_SCORES:
                scores.append(URGENCY_SCORES[urg])
        elif m == "text":
            sent = str(r.get("sentiment_tone", "")).lower()
            if sent in SENTIMENT_SCORES:
                scores.append(SENTIMENT_SCORES[sent])
        elif m == "pdf":
            conf = float(r.get("confidence_score", 0) or 0)
            scores.append(2 if conf > 0.85 else 1)

    if not scores:
        return "medium"
    avg = sum(scores) / len(scores)
    if avg >= 2.5:
        return "high"
    elif avg >= 1.5:
        return "medium"
    else:
        return "low"


def merge_lists(*lists) -> str:
    """Merge multiple list / semicolon-string fields, deduplicate, return as semicolon string."""
    combined = []
    for val in lists:
        if isinstance(val, list):
            combined.extend([str(v).strip() for v in val if v])
        elif isinstance(val, str) and val:
            combined.extend([s.strip() for s in val.split(";") if s.strip()])
    seen = set()
    result = []
    for item in combined:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return "; ".join(result)


def coalesce(*values):
    """Return the first non-null, non-empty value."""
    for v in values:
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            return v
    return None


# ── Incident ID assignment ─────────────────────────────────────────────────────

def assign_incident_ids(records: List[dict]) -> List[dict]:
    """
    Assign a shared Incident_ID to all records that belong to the same real-world incident.

    Grouping rules (applied in priority order):
      1. Records with the same pre-set 'incident_group_id' → same group.
      2. Same date + same incident_type + at least one shared location keyword → same group.
      3. All other records get their own unique Incident_ID.

    Every record is guaranteed an 'incident_id' key on return.
    """

    def loc_tokens(loc: str) -> set:
        return set(re.findall(r'\b\w{4,}\b', (loc or "").lower()))

    groups: list[list[int]] = []

    for i, rec in enumerate(records):
        pre = rec.get("incident_group_id")
        if pre:
            for group in groups:
                if records[group[0]].get("incident_group_id") == pre:
                    group.append(i)
                    break
            else:
                groups.append([i])
            continue

        rd = rec.get("date") or ""
        rt = (rec.get("incident_type") or "").lower()
        rl = loc_tokens(rec.get("location") or "")

        placed = False
        for group in groups:
            rep = records[group[0]]
            if rep.get("incident_group_id"):
                continue
            if (rd == rep.get("date", "") and rd != ""
                    and rt == (rep.get("incident_type") or "").lower() and rt != ""
                    and len(rl & loc_tokens(rep.get("location") or "")) >= 1):
                group.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])

    for group in groups:
        gid = f"INC-{str(uuid.uuid4())[:6].upper()}"
        for idx in group:
            records[idx]["incident_id"] = gid

    return records


# ── Pivot to one row per incident ─────────────────────────────────────────────

def pivot_to_incident_rows(records: List[dict]) -> pd.DataFrame:
    """Pivot all modality records into ONE row per incident after grouping."""
    records = assign_incident_ids(records)

    groups: dict[str, list[dict]] = {}
    for rec in records:
        gid = rec.get("incident_id", "UNGROUPED")
        groups.setdefault(gid, []).append(rec)

    rows = []
    for gid, grecs in groups.items():
        by_mod = {r["source_modality"]: r for r in grecs}
        pdf = by_mod.get("pdf",   {})
        aud = by_mod.get("audio", {})
        txt = by_mod.get("text",  {})

        # ── Shared core fields (first non-null wins) ──────────────────────
        date     = coalesce(pdf.get("date"),  aud.get("date"),  txt.get("date"))
        time_val = coalesce(pdf.get("time"),  aud.get("time"),  txt.get("time"))
        location = coalesce(pdf.get("location"), aud.get("location"), txt.get("location"))
        inc_type = coalesce(pdf.get("incident_type"), aud.get("incident_type"),
                            txt.get("incident_type"))
        officer  = coalesce(pdf.get("officer"), aud.get("officer"))
        status   = coalesce(pdf.get("status"), aud.get("status"), txt.get("status")) or "open"

        # ── Cross-modality fused fields ───────────────────────────────────
        all_suspects = merge_lists(
            pdf.get("suspects", ""), aud.get("suspects", ""), txt.get("suspects", ""))
        all_victims  = merge_lists(
            pdf.get("victims",  ""), aud.get("victims",  ""), txt.get("victims",  ""))
        all_evidence = merge_lists(
            pdf.get("evidence", ""), aud.get("evidence", ""), txt.get("evidence", ""))
        descriptions       = [r.get("description", "") for r in grecs if r.get("description")]
        combined_desc      = " | ".join(dict.fromkeys(descriptions))
        combined_severity  = compute_combined_severity(grecs)
        confs              = [float(r.get("confidence_score", 0) or 0) for r in grecs]
        combined_confidence = round(sum(confs) / len(confs), 3) if confs else 0.0
        sources_present    = "; ".join(sorted(r["source_modality"] for r in grecs))

        # ── PDF columns ───────────────────────────────────────────────────
        pdf_info     = f"Report — {inc_type} at {location}" if pdf else ""
        pdf_suspects = merge_lists(pdf.get("suspects", ""))
        pdf_evidence = merge_lists(pdf.get("evidence", ""))
        pdf_officer  = pdf.get("officer") or ""
        pdf_method   = pdf.get("extraction_method", "") if pdf else ""
        pdf_conf     = float(pdf.get("confidence_score", 0) or 0)

        # ── Audio columns ─────────────────────────────────────────────────
        audio_event     = (f"{inc_type} — {aud.get('urgency','unknown')} urgency"
                           if aud else "")
        audio_transcript = (aud.get("transcript") or "")[:500]
        audio_urgency    = aud.get("urgency") or ""
        audio_caller     = aud.get("caller_type") or ""
        audio_conf       = float(aud.get("confidence_score", 0) or 0)

        # ── Text NLP columns ──────────────────────────────────────────────
        ts_raw = txt.get("topic_scores", "")
        if isinstance(ts_raw, dict):
            ts_raw = json.dumps(ts_raw)

        text_crime_type   = (f"Primary: {inc_type}" if txt else "")
        text_sentiment    = txt.get("sentiment_tone") or ""
        text_sentiment_sc = float(txt.get("sentiment_score", 0) or 0)
        text_ner_persons  = merge_lists(txt.get("ner_persons", ""))
        text_ner_locs     = merge_lists(txt.get("ner_locations", ""))
        text_ner_dates    = merge_lists(txt.get("ner_dates", ""))
        text_topic_scores = ts_raw or ""
        text_conf         = float(txt.get("confidence_score", 0) or 0)

        rows.append({
            # ── Incident identity ─────────────────────────────────────────
            "incident_id":           gid,
            "date":                  date,
            "time":                  time_val,
            "location":              location,
            "incident_type":         inc_type,
            "combined_description":  combined_desc,
            "officer":               officer,
            "overall_status":        status,
            "sources_present":       sources_present,
            # ── Cross-modality fused fields ───────────────────────────────
            "all_suspects":          all_suspects,
            "all_victims":           all_victims,
            "all_evidence":          all_evidence,
            "combined_severity":     combined_severity,
            "combined_confidence":   combined_confidence,
            # ── PDF modality ──────────────────────────────────────────────
            "pdf_info":              pdf_info,
            "pdf_suspects":          pdf_suspects,
            "pdf_evidence":          pdf_evidence,
            "pdf_officer":           pdf_officer,
            "pdf_extraction_method": pdf_method,
            "pdf_confidence":        pdf_conf,
            # ── Audio modality ────────────────────────────────────────────
            "audio_event":           audio_event,
            "audio_transcript":      audio_transcript,
            "audio_urgency":         audio_urgency,
            "audio_caller_type":     audio_caller,
            "audio_confidence":      audio_conf,
            # ── Text NLP modality ─────────────────────────────────────────
            "text_crime_type":       text_crime_type,
            "text_sentiment":        text_sentiment,
            "text_sentiment_score":  text_sentiment_sc,
            "text_ner_persons":      text_ner_persons,
            "text_ner_locations":    text_ner_locs,
            "text_ner_dates":        text_ner_dates,
            "text_topic_scores":     text_topic_scores,
            "text_confidence":       text_conf,
            # ── Metadata ──────────────────────────────────────────────────
            "processed_at":          datetime.now().isoformat(),
        })

    df = pd.DataFrame(rows)
    df.insert(0, "row_num", range(1, len(df) + 1))
    df = df.sort_values("date", na_position="last").reset_index(drop=True)
    df["row_num"] = range(1, len(df) + 1)
    return df


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, output_dir: str = "data/outputs") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path    = os.path.join(output_dir, f"fused_incidents_{ts}.csv")
    json_path   = os.path.join(output_dir, f"fused_incidents_{ts}.json")
    csv_latest  = os.path.join(output_dir, "structured_incidents_latest.csv")
    json_latest = os.path.join(output_dir, "structured_incidents_latest.json")

    df.to_csv(csv_path,   index=False)
    df.to_csv(csv_latest, index=False)
    print(f"[SAVE] CSV  → {csv_path}")

    records = df.to_dict(orient="records")
    for rec in records:
        if isinstance(rec.get("text_topic_scores"), str) and rec["text_topic_scores"]:
            try:
                rec["text_topic_scores"] = json.loads(rec["text_topic_scores"])
            except Exception:
                pass

    with open(json_path,   "w") as f: json.dump(records, f, indent=2, default=str)
    with open(json_latest, "w") as f: json.dump(records, f, indent=2, default=str)
    print(f"[SAVE] JSON → {json_path}")
    return {"csv": csv_path, "json": json_path,
            "latest_csv": csv_latest, "latest_json": json_latest}


# ── Summary report ────────────────────────────────────────────────────────────

def generate_summary_report(df: pd.DataFrame) -> str:
    all_sources: set = set()
    for val in df["sources_present"].dropna():
        all_sources.update(s.strip() for s in val.split(";"))

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║   3-MODALITY FUSION DATASET — INTEGRATION SUMMARY           ║",
        "╚══════════════════════════════════════════════════════════════╝",
        f"  Total Incidents (rows)  : {len(df)}",
        f"  Modalities in dataset   : {', '.join(sorted(all_sources))}",
        "",
        "  ── Combined Severity ──────────────────────────",
    ]
    for sev in ["high", "medium", "low"]:
        count = int((df["combined_severity"] == sev).sum())
        bar = "█" * count + "░" * max(0, 5 - count)
        lines.append(f"    {sev:<10} {bar}  {count}")

    lines += ["", "  ── Incident Types ────────────────────────────"]
    for itype, count in df["incident_type"].value_counts().items():
        lines.append(f"    {str(itype):<30} {count}")

    lines += ["", "  ── Sources per Incident ───────────────────────"]
    for sp, count in df["sources_present"].value_counts().items():
        lines.append(f"    {str(sp):<45} {count}")

    avg_conf = df["combined_confidence"].mean()
    lines += [
        "",
        f"  Avg Combined Confidence  : {avg_conf:.2f}",
        f"  Date Range               : {df['date'].dropna().min()} → {df['date'].dropna().max()}",
        "══════════════════════════════════════════════════════════════",
        "",
    ]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_integration(records: List[dict], output_dir: str = "data/outputs") -> dict:
    """
    3-Modality fusion pipeline.

    Steps:
      1. Assign Incident_IDs — group records by date + type + shared location keyword
      2. Pivot each group into ONE unified row with all modality columns
      3. Compute combined_severity from audio urgency + text sentiment + PDF confidence
      4. Save to timestamped + latest CSV and JSON

    Args:
        records    : List of per-modality dicts from pdf_processor, audio_transcriber, text_nlp
        output_dir : Output directory (default: data/outputs)

    Returns:
        dict with keys: paths, summary, record_count, dataframe
    """
    if not records:
        print("[MERGE] No records provided.")
        return {"paths": {}, "summary": "", "record_count": 0, "dataframe": pd.DataFrame()}

    print(f"[MERGE] Fusing {len(records)} modality record(s) → incident rows...")
    df = pivot_to_incident_rows(records)
    print(f"[MERGE] Result: {len(df)} fused incident row(s)")

    paths   = save_outputs(df, output_dir)
    summary = generate_summary_report(df)
    print(summary)
    return {"paths": paths, "summary": summary, "record_count": len(df), "dataframe": df}


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = [
        {
            "source_modality": "pdf", "source_file": "HPD_Report_Burglary.pdf",
            "date": "2024-11-20", "time": "09:45",
            "location": "4521 Richmond Ave, Houston, TX",
            "incident_type": "Burglary", "status": "under_investigation",
            "description": "Forced entry through rear window. Electronics stolen.",
            "suspects": ["Unknown male, 5'10\", dark hoodie"],
            "victims": ["Maria Rodriguez"],
            "evidence": ["Broken window", "Boot prints", "Stolen laptop"],
            "officer": "Officer James Kim", "confidence_score": 0.93,
            "extraction_method": "text",
        },
        {
            "source_modality": "audio", "source_file": "911_call_assault.mp3",
            "date": "2024-11-20", "time": "23:12",
            "location": "Westheimer Rd & Montrose Blvd, Houston, TX",
            "incident_type": "Assault", "status": "open",
            "description": "Two men fighting outside Club Luxe. Bottle used as weapon.",
            "suspects": ["Male in red jacket ~6ft", "Male in white shirt ~5'8\""],
            "victims": ["Male, 30s, bleeding from temple"],
            "evidence": ["Broken glass bottle"],
            "caller_type": "911_caller", "urgency": "high", "confidence_score": 0.87,
            "transcript": "There is a fight outside Club Luxe. One guy hit the other with a bottle.",
        },
        {
            "source_modality": "text", "source_file": "witness_statement.txt",
            "date": "2024-11-20", "time": "23:20",
            "location": "Westheimer Rd & Montrose Blvd, Houston, TX",
            "incident_type": "Assault", "status": "open",
            "description": "Witness saw two men attack a woman and steal her purse near Club Luxe.",
            "suspects": ["Red jacket male ~6ft", "White hoodie male ~5'8\""],
            "victims": ["Woman, mid-30s, black leather purse stolen"],
            "evidence": ["Stolen black leather purse"],
            "confidence_score": 0.85, "urgency": "high",
            "sentiment_tone": "high_distress", "sentiment_score": 0.95,
            "ner_persons": ["Robert Chen"],
            "ner_locations": ["Westheimer Road", "Montrose Blvd"],
            "ner_dates": ["November 20th"],
            "topic_scores": {"Assault": 4, "Robbery": 3},
        },
    ]
    result = run_integration(demo, output_dir="data/outputs")
    print(result["dataframe"][["incident_id", "incident_type",
                                "combined_severity", "sources_present"]].to_string(index=False))
