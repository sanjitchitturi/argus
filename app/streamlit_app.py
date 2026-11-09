"""
Argus — Streamlit Inspection Dashboard
"""

import json
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Argus Inspection Workspace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.warning(
    "DEMO MODE: outputs may be synthetic. This prototype is not validated for industrial inspection. Verify model configuration and review every finding."
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1c2537;
    --accent: #00d4ff;
    --red: #ef4444;
    --amber: #f59e0b;
    --green: #10b981;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }

.argus-header {
    background: var(--surface);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.argus-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
}
.argus-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-card {
    background: var(--surface);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.defect-card {
    background: var(--surface2);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 3px solid;
}
.defect-high  { border-color: var(--red); }
.defect-medium { border-color: var(--amber); }
.defect-low   { border-color: var(--green); }

.severity-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-high   { background: rgba(239,68,68,0.2);  color: #f87171; }
.badge-medium { background: rgba(245,158,11,0.2); color: #fbbf24; }
.badge-low    { background: rgba(16,185,129,0.2); color: #34d399; }

.report-box {
    background: var(--surface);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    white-space: pre-wrap;
    line-height: 1.7;
    max-height: 500px;
    overflow-y: auto;
    color: #94a3b8;
}

.caption-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-style: italic;
    color: #cbd5e1;
}

.vqa-item {
    background: var(--surface);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}
.vqa-q { color: var(--accent); font-size: 0.85rem; font-weight: 600; }
.vqa-a { color: #94a3b8; font-size: 0.9rem; margin-top: 0.3rem; }

.status-running { color: var(--accent); }
.status-done    { color: var(--green); }
.status-error   { color: var(--red); }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #1e3a5f;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ──────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "results" not in st.session_state:
    st.session_state.results = None
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False


# ── Pipeline Loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(yolo_model, vlm_model, llm_model, device):
    from src.pipeline import ArgusPipeline

    return ArgusPipeline(
        yolo_model=yolo_model,
        vlm_model=vlm_model,
        llm_model=llm_model,
        device=device,
    )


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="argus-header">
    <div class="argus-title">⬡ ARGUS</div>
    <div class="argus-sub">Vision-Language Inspection Workspace &nbsp;·&nbsp; v1.0</div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    yolo_model = st.selectbox(
        "Detection Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "custom"],
        index=0,
    )
    vlm_model = st.selectbox(
        "Vision-Language Model",
        ["blip2", "llava"],
        index=0,
    )
    llm_model = st.selectbox(
        "Report Generator",
        ["template", "mistral", "llama", "phi"],
        index=0,
        help="'template' is fast; others require GPU and HuggingFace download",
    )
    device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)
    confidence = st.slider("Detection Confidence", 0.1, 0.95, 0.35, 0.05)

    st.markdown("---")
    st.markdown("### 💬 Custom Questions")
    default_questions = [
        "What defect is visible?",
        "Where is the defect located?",
        "How severe is the defect?",
        "What is the surface condition?",
    ]
    questions_text = st.text_area(
        "One question per line",
        value="\n".join(default_questions),
        height=130,
    )
    custom_questions = [q.strip() for q in questions_text.splitlines() if q.strip()]

    st.markdown("---")
    st.markdown("### 📊 MLflow Tracking")
    track_mlflow = st.toggle("Enable MLflow", value=False)

    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.75rem;'>Argus · inspectable visual review</div>",
        unsafe_allow_html=True,
    )


# ── Main Panel ─────────────────────────────────────────────────────────────
tab_inspect, tab_evaluate, tab_history = st.tabs(["🔍 Inspect", "📈 Evaluate", "🗃️ History"])


# ── TAB 1: Inspect ─────────────────────────────────────────────────────────
with tab_inspect:
    col_upload, col_results = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Input Image", use_container_width=True)

            image_id = st.text_input(
                "Image ID (optional)",
                value="",
                placeholder="e.g. PART-2024-001",
            )
            inspector_name = st.text_input("Inspector", value="Argus AI")

            run_btn = st.button("⚡ Run Inspection", type="primary", use_container_width=True)

            if run_btn:
                img_id = image_id.strip() or None
                with st.spinner("Loading pipeline..."):
                    pipeline = load_pipeline(yolo_model, vlm_model, llm_model, device)
                    pipeline.detector.confidence = confidence

                progress = st.progress(0, text="Starting pipeline...")
                t_start = time.perf_counter()

                progress.progress(10, "🔍 Running defect detection...")
                det_result = pipeline.detector.detect(pil_img)
                progress.progress(40, "🧠 Generating image caption...")
                cap_result = pipeline.vlm.caption(pil_img)
                progress.progress(65, "💬 Running visual Q&A...")
                vqa_results = pipeline.vlm.batch_vqa(pil_img, custom_questions)
                progress.progress(85, "📝 Generating inspection report...")
                report_result = pipeline.reporter.generate(
                    detections=det_result["detections"],
                    caption=cap_result["caption"],
                    vqa_answers=vqa_results,
                    image_id=img_id or "ARGUS-LIVE",
                    inspector=inspector_name,
                )
                progress.progress(100, "✅ Complete!")
                total_ms = (time.perf_counter() - t_start) * 1000

                st.session_state.results = {
                    "image_id": img_id or "ARGUS-LIVE",
                    "detection": det_result,
                    "caption": cap_result,
                    "vqa": vqa_results,
                    "report": report_result,
                    "total_ms": round(total_ms),
                    "pil_img": pil_img,
                    "pipeline": pipeline,
                }

                if track_mlflow:
                    try:
                        import mlflow

                        with mlflow.start_run(run_name=img_id or "argus_run"):
                            mlflow.log_param("yolo_model", yolo_model)
                            mlflow.log_param("vlm_model", vlm_model)
                            mlflow.log_param("llm_model", llm_model)
                            mlflow.log_metric("num_defects", det_result["num_defects"])
                            mlflow.log_metric("detection_ms", det_result["inference_ms"])
                            mlflow.log_metric("caption_ms", cap_result["inference_ms"])
                        st.success("MLflow run logged ✓")
                    except Exception as e:
                        st.warning(f"MLflow logging failed: {e}")

    # ── Results Column ─────────────────────────────────────────────────────
    with col_results:
        if st.session_state.results:
            res = st.session_state.results
            det = res["detection"]
            n = det["num_defects"]

            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{n}</div>
                    <div class="metric-label">Defects</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                high_n = sum(1 for d in det["detections"] if d["severity"] == "HIGH")
                color = "#ef4444" if high_n else "#10b981"
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color}">{high_n}</div>
                    <div class="metric-label">High Severity</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{det["inference_ms"]:.0f}</div>
                    <div class="metric-label">Detect ms</div></div>""",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{res["total_ms"]}</div>
                    <div class="metric-label">Total ms</div></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Annotated image
            try:
                annotated = res["pipeline"].visualize(res["pil_img"])
                st.image(annotated, caption="Annotated Detections", use_container_width=True)
            except Exception:
                pass

            # Detections
            st.markdown("#### 🎯 Detected Defects")
            if not det["detections"]:
                st.success("✓ No defects detected")
            else:
                for d in det["detections"]:
                    sev = d["severity"].lower()
                    badge_cls = f"badge-{sev}"
                    card_cls = f"defect-{sev}"
                    st.markdown(
                        f"""
                    <div class="defect-card {card_cls}">
                        <strong>{d["class"].replace("_", " ").title()}</strong>
                        &nbsp;<span class="severity-badge {badge_cls}">{d["severity"]}</span>
                        <br>
                        <small style="color:#64748b">
                        conf={d["confidence"]:.2%} &nbsp;|&nbsp;
                        area={d["area_px"]:,}px² &nbsp;|&nbsp;
                        bbox=({d["bbox"]["x1"]},{d["bbox"]["y1"]})→({d["bbox"]["x2"]},{d["bbox"]["y2"]})
                        </small>
                    </div>""",
                        unsafe_allow_html=True,
                    )

            # Caption
            st.markdown("#### 📸 Image Caption")
            st.markdown(
                f'<div class="caption-box">{res["caption"]["caption"]}</div>',
                unsafe_allow_html=True,
            )

            # VQA
            st.markdown("#### 💬 Visual Q&A")
            for qa in res["vqa"]:
                st.markdown(
                    f"""
                <div class="vqa-item">
                    <div class="vqa-q">Q: {qa["question"]}</div>
                    <div class="vqa-a">A: {qa["answer"]}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            # Report
            st.markdown("#### 📋 Inspection Report")
            st.markdown(
                f'<div class="report-box">{res["report"]["report_text"]}</div>',
                unsafe_allow_html=True,
            )

            # Download buttons
            st.markdown("#### 💾 Export")
            col_j, col_t = st.columns(2)
            with col_j:
                export = {
                    "image_id": res["image_id"],
                    "detection": res["detection"],
                    "caption": res["caption"],
                    "vqa": res["vqa"],
                    "report": res["report"],
                }
                st.download_button(
                    "⬇ Download JSON",
                    data=json.dumps(export, indent=2, default=str),
                    file_name=f"{res['image_id']}_inspection.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_t:
                st.download_button(
                    "⬇ Download Report",
                    data=res["report"]["report_text"],
                    file_name=f"{res['image_id']}_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.info("Upload an image and click **Run Inspection** to begin.")


# ── TAB 2: Evaluate ────────────────────────────────────────────────────────
with tab_evaluate:
    st.markdown("### 📈 Evaluation Metrics")
    st.markdown(
        "Upload prediction and ground-truth JSON files to compute "
        "**mAP** (detection), **BLEU** (captioning), and **ROUGE** (report quality)."
    )

    col_p, col_g = st.columns(2)
    with col_p:
        pred_file = st.file_uploader("Predictions JSON", type="json", key="pred_upload")
    with col_g:
        gt_file = st.file_uploader("Ground Truth JSON", type="json", key="gt_upload")

    if st.button("▶ Run Evaluation", type="primary") and pred_file and gt_file:
        preds = json.load(pred_file)
        gts = json.load(gt_file)

        from src.evaluation.metrics import run_full_evaluation

        with st.spinner("Computing metrics..."):
            metrics = run_full_evaluation(
                predictions=preds.get("predictions", []),
                ground_truths=gts.get("ground_truths", []),
                captions=preds.get("captions", []),
                reference_captions=gts.get("reference_captions", [[]]),
                reports=preds.get("reports", []),
                reference_reports=gts.get("reference_reports", [""]),
            )

        st.success("Evaluation complete!")

        # Detection metrics
        st.markdown("#### 🎯 Detection Metrics")
        dm = metrics["detection"]
        cols = st.columns(4)
        for col, (k, v) in zip(
            cols,
            [
                ("mAP", dm["mAP"]),
                ("Precision", dm["precision"]),
                ("Recall", dm["recall"]),
                ("IoU Thresh", dm["iou_threshold"]),
            ],
        ):
            col.metric(k, f"{v:.4f}")

        if dm.get("per_class_AP"):
            st.markdown("**Per-class AP:**")
            st.json(dm["per_class_AP"])

        # BLEU
        st.markdown("#### 📝 Captioning — BLEU Scores")
        bleu = metrics["captioning"]
        cols = st.columns(4)
        for col, (k, v) in zip(cols, bleu.items()):
            col.metric(k, f"{v:.4f}")

        # ROUGE
        st.markdown("#### 📋 Report Quality — ROUGE Scores")
        rouge = metrics["report_quality"]
        for rt, scores in rouge.items():
            st.markdown(f"**{rt.upper()}**")
            cols = st.columns(3)
            for col, (k, v) in zip(cols, scores.items()):
                col.metric(k.capitalize(), f"{v:.4f}")

        st.download_button(
            "⬇ Download Metrics JSON",
            data=json.dumps(metrics, indent=2),
            file_name="argus_metrics.json",
            mime="application/json",
        )


# ── TAB 3: History ─────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 🗃️ Past Inspections")
    from src.utils.config import EXPERIMENTS_DIR

    result_files = sorted(
        Path(EXPERIMENTS_DIR).glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not result_files:
        st.info("No inspection records yet. Run an inspection to populate history.")
    else:
        rows = []
        for f in result_files[:100]:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                rows.append(
                    {
                        "Image ID": data.get("image_id", f.stem),
                        "Timestamp": data.get("timestamp", "")[:19].replace("T", " "),
                        "Defects": data.get("detection", {}).get("num_defects", 0),
                        "File": f.name,
                    }
                )
            except Exception:
                pass

        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        selected = st.selectbox("View inspection detail", ["—"] + [r["Image ID"] for r in rows])
        if selected != "—":
            file = next((f for f in result_files if selected in f.stem), None)
            if file:
                with open(file) as fp:
                    detail = json.load(fp)
                st.json(detail)
