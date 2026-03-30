"""
dashboard/app.py — Streamlit Dashboard for the 3-Modality Fused Incident Dataset
Launch: streamlit run dashboard/app.py
"""
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Crime Incident Analyzer",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.big-title { font-size: 1.8rem; font-weight: 700; color: #1F3864; }
.sub       { font-size: 1rem; color: #888; margin-bottom: 1rem; }
.kpi-box   { background: #F0F4FA; border-radius: 8px; padding: 1rem;
             text-align: center; border-left: 4px solid #2E75B6; }
.kpi-num   { font-size: 2rem; font-weight: 700; color: #1F3864; }
.kpi-label { font-size: 0.85rem; color: #666; }
</style>""", unsafe_allow_html=True)

DATA_PATHS = [
    "data/outputs/structured_incidents_latest.csv",
    "../data/outputs/structured_incidents_latest.csv",
    os.path.join(os.path.dirname(__file__), "..", "data", "outputs",
                 "structured_incidents_latest.csv"),
]


@st.cache_data(ttl=30)
def load_data():
    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["combined_confidence", "audio_confidence",
                        "pdf_confidence", "text_confidence"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df, path
    return None, None


def sidebar_filters(df):
    st.sidebar.header("🔎 Filters")

    # Severity
    sev_opts = sorted(df["combined_severity"].dropna().unique()) \
        if "combined_severity" in df.columns else []
    sel_sev = st.sidebar.multiselect("Severity", sev_opts, default=list(sev_opts))

    # Incident type
    type_opts = sorted(df["incident_type"].dropna().unique()) \
        if "incident_type" in df.columns else []
    sel_types = st.sidebar.multiselect("Incident Type", type_opts, default=list(type_opts))

    # Status
    status_opts = sorted(df["overall_status"].dropna().unique()) \
        if "overall_status" in df.columns else []
    sel_status = st.sidebar.multiselect("Status", status_opts, default=list(status_opts))

    # Sources
    src_opts = sorted({s.strip() for val in df["sources_present"].dropna()
                       for s in val.split(";")}) if "sources_present" in df.columns else []
    sel_src = st.sidebar.multiselect("Modality Sources", src_opts, default=list(src_opts))

    # Confidence
    conf_min = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05) \
        if "combined_confidence" in df.columns else 0.0

    # Text search
    search = st.sidebar.text_input("🔍 Search (location / suspects / description)")

    # Apply filters
    mask = pd.Series([True] * len(df))
    if sel_sev   and "combined_severity" in df.columns:
        mask &= df["combined_severity"].isin(sel_sev)
    if sel_types and "incident_type" in df.columns:
        mask &= df["incident_type"].isin(sel_types)
    if sel_status and "overall_status" in df.columns:
        mask &= df["overall_status"].isin(sel_status)
    if "combined_confidence" in df.columns:
        mask &= df["combined_confidence"].fillna(0) >= conf_min
    if sel_src and "sources_present" in df.columns:
        mask &= df["sources_present"].apply(
            lambda v: any(s in (v or "") for s in sel_src))
    if search:
        q = search.lower()
        text_mask = pd.Series([False] * len(df))
        for col in ["location", "all_suspects", "combined_description",
                    "incident_type", "all_evidence"]:
            if col in df.columns:
                text_mask |= df[col].fillna("").str.lower().str.contains(q)
        mask &= text_mask

    return df[mask].copy()


def kpi_card(col, number, label):
    col.markdown(
        f'<div class="kpi-box"><div class="kpi-num">{number}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def show_charts(df):
    c1, c2 = st.columns(2)

    # Bar — incident types
    if "incident_type" in df.columns:
        counts = df["incident_type"].value_counts().reset_index()
        counts.columns = ["Incident Type", "Count"]
        fig = px.bar(counts, x="Incident Type", y="Count",
                     color="Incident Type",
                     title="Incidents by Type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=320)
        c1.plotly_chart(fig, use_container_width=True)

    # Pie — sources present
    if "sources_present" in df.columns:
        src_counts: dict = {}
        for val in df["sources_present"].dropna():
            for s in val.split(";"):
                s = s.strip()
                src_counts[s] = src_counts.get(s, 0) + 1
        if src_counts:
            src_df = pd.DataFrame(src_counts.items(), columns=["Modality", "Count"])
            fig2 = px.pie(src_df, names="Modality", values="Count",
                          title="Records by Modality",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(height=320)
            c2.plotly_chart(fig2, use_container_width=True)

    # Confidence histogram
    if "combined_confidence" in df.columns and df["combined_confidence"].notna().any():
        fig3 = px.histogram(
            df, x="combined_confidence", nbins=10,
            title="Combined Confidence Distribution",
            labels={"combined_confidence": "Confidence"},
            color_discrete_sequence=["#2E75B6"],
        )
        fig3.update_layout(height=280)
        st.plotly_chart(fig3, use_container_width=True)

    # Severity bar
    if "combined_severity" in df.columns:
        sev_order = ["high", "medium", "low"]
        sev_counts = (df["combined_severity"]
                      .value_counts()
                      .reindex(sev_order, fill_value=0)
                      .reset_index())
        sev_counts.columns = ["Severity", "Count"]
        fig4 = px.bar(sev_counts, x="Severity", y="Count",
                      color="Severity",
                      color_discrete_map={"high": "#d62728", "medium": "#ff7f0e", "low": "#2ca02c"},
                      title="Incidents by Combined Severity")
        fig4.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig4, use_container_width=True)


def show_detail(row):
    """Expand panel for a selected incident row."""
    st.markdown(f"### 🗂 Incident Detail — `{row.get('incident_id', '')}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Incident Type", row.get("incident_type", "—"))
    c2.metric("Severity",      row.get("combined_severity", "—"))
    c3.metric("Confidence",    f"{float(row.get('combined_confidence', 0)):.0%}")

    st.markdown(f"**📍 Location:** {row.get('location', '—')}")
    st.markdown(f"**📅 Date/Time:** {row.get('date', '—')} {row.get('time', '')}".strip())
    st.markdown(f"**👮 Officer:** {row.get('officer', '—')}")
    st.markdown(f"**📋 Status:** {row.get('overall_status', '—')}")
    st.markdown(f"**🔗 Sources:** `{row.get('sources_present', '—')}`")

    if row.get("combined_description"):
        st.markdown("**📝 Description:**")
        st.info(row["combined_description"])

    ta, tb, tc = st.columns(3)
    if row.get("all_suspects"):
        ta.markdown("**🚨 Suspects:**")
        for s in row["all_suspects"].split(";"):
            if s.strip(): ta.markdown(f"- {s.strip()}")
    if row.get("all_victims"):
        tb.markdown("**🏥 Victims:**")
        for v in row["all_victims"].split(";"):
            if v.strip(): tb.markdown(f"- {v.strip()}")
    if row.get("all_evidence"):
        tc.markdown("**🔬 Evidence:**")
        for e in row["all_evidence"].split(";"):
            if e.strip(): tc.markdown(f"- {e.strip()}")

    # Modality tabs
    tabs = st.tabs(["📄 PDF", "🎙️ Audio", "📝 Text NLP"])

    with tabs[0]:
        if row.get("pdf_info"):
            st.markdown(f"**Info:** {row.get('pdf_info')}")
            st.markdown(f"**Extraction Method:** `{row.get('pdf_extraction_method', '—')}`")
            st.markdown(f"**Officer:** {row.get('pdf_officer', '—')}")
            st.markdown(f"**Confidence:** {float(row.get('pdf_confidence', 0)):.0%}")
        else:
            st.info("No PDF data for this incident.")

    with tabs[1]:
        if row.get("audio_event"):
            st.markdown(f"**Event Summary:** {row.get('audio_event')}")
            st.markdown(f"**Urgency:** `{row.get('audio_urgency', '—')}`")
            st.markdown(f"**Caller Type:** `{row.get('audio_caller_type', '—')}`")
            st.markdown(f"**Confidence:** {float(row.get('audio_confidence', 0)):.0%}")
            if row.get("audio_transcript"):
                st.markdown("**Transcript:**")
                st.code(row["audio_transcript"], language=None)
        else:
            st.info("No audio data for this incident.")

    with tabs[2]:
        if row.get("text_crime_type"):
            st.markdown(f"**Crime Classification:** {row.get('text_crime_type')}")
            st.markdown(f"**Sentiment Tone:** `{row.get('text_sentiment', '—')}` "
                        f"(score: {float(row.get('text_sentiment_score', 0)):.2f})")
            if row.get("text_ner_persons"):
                st.markdown(f"**NER Persons:** {row['text_ner_persons']}")
            if row.get("text_ner_locations"):
                st.markdown(f"**NER Locations:** {row['text_ner_locations']}")
            if row.get("text_ner_dates"):
                st.markdown(f"**NER Dates:** {row['text_ner_dates']}")
            st.markdown(f"**Confidence:** {float(row.get('text_confidence', 0)):.0%}")
            ts = row.get("text_topic_scores")
            if ts:
                try:
                    scores = json.loads(ts) if isinstance(ts, str) else ts
                    st.markdown("**Topic Keyword Scores:**")
                    score_df = pd.DataFrame(
                        sorted(scores.items(), key=lambda x: x[1], reverse=True),
                        columns=["Category", "Score"]
                    )
                    st.dataframe(score_df, hide_index=True, use_container_width=False)
                except Exception:
                    st.text(str(ts))
        else:
            st.info("No text NLP data for this incident.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="big-title">🔍 Multimodal Crime / Incident Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub">3 Modalities: PDF · Audio · Text NLP &nbsp;|&nbsp; '
                'COMP 4XXX Final Assignment</div>', unsafe_allow_html=True)

    df_raw, data_path = load_data()
    if df_raw is None:
        st.error("No dataset found. Run `python run_pipeline.py --demo` first to generate data.")
        st.code("python run_pipeline.py --demo", language="bash")
        st.stop()

    st.caption(f"Loaded from: `{data_path}` — {len(df_raw)} total records")

    df = sidebar_filters(df_raw)

    # KPI row
    open_count = int((df["overall_status"] == "open").sum()) \
        if "overall_status" in df.columns else 0
    src_count = len({s.strip() for val in df["sources_present"].dropna()
                     for s in val.split(";")}) if "sources_present" in df.columns else 0
    avg_conf = df["combined_confidence"].mean() if "combined_confidence" in df.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, len(df),              "Total Incidents")
    kpi_card(k2, open_count,           "Open Cases")
    kpi_card(k3, src_count,            "Active Modalities")
    kpi_card(k4, f"{avg_conf:.0%}",    "Avg Confidence")

    st.markdown("---")
    show_charts(df)

    # Records table + detail
    st.markdown("---")
    st.subheader("📋 Incident Records")

    display_cols = [c for c in [
        "row_num", "incident_id", "date", "incident_type",
        "combined_severity", "location", "overall_status",
        "sources_present", "combined_confidence",
    ] if c in df.columns]

    st.dataframe(
        df[display_cols].rename(columns={"row_num": "#"}),
        use_container_width=True,
        hide_index=True,
    )

    # Detail view
    st.markdown("---")
    st.subheader("🔎 Incident Detail View")
    if len(df) == 0:
        st.info("No records match the current filters.")
    else:
        incident_ids = df["incident_id"].tolist() if "incident_id" in df.columns else []
        selected_id = st.selectbox("Select Incident ID", incident_ids)
        if selected_id:
            row = df[df["incident_id"] == selected_id].iloc[0].to_dict()
            show_detail(row)

    # Export
    st.markdown("---")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_bytes,
        file_name="filtered_incidents.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
