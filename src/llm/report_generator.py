"""
Argus LLM Module — Inspection Report Generator
Uses HuggingFace pipeline (Mistral / LLaMA / Falcon) to generate
structured inspection reports from detection + VLM outputs.
"""

from __future__ import annotations
import time
import textwrap
from datetime import datetime, timezone

from loguru import logger

try:
    import os

    if os.getenv("ARGUS_MODE", "demo") == "demo":
        raise ImportError("Explicit demo mode")
    from transformers import pipeline
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — LLM in mock/template mode")


REPORT_SYSTEM_PROMPT = """You are an expert industrial quality-control engineer.
You receive structured inspection data and produce a formal, concise inspection report.
Structure your report with these sections:
1. Executive Summary
2. Defect Analysis
3. Visual Description
4. Risk Assessment
5. Recommended Actions
6. Conclusion

Be professional, precise, and actionable. Avoid filler text."""


class InspectionReportGenerator:
    """
    LLM-powered inspection report generator.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Defaults to Mistral-7B-Instruct.
        Use 'template' to skip LLM and use rule-based template.
    device : str
        'cpu', 'cuda', or 'mps'.
    max_new_tokens : int
        Maximum tokens for generated report.
    load_in_8bit : bool
        8-bit quantisation (CUDA only).
    """

    SHORTCUTS = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "falcon": "tiiuae/falcon-7b-instruct",
        "phi": "microsoft/phi-2",
    }

    def __init__(
        self,
        model_name: str = "template",
        device: str = "cpu",
        max_new_tokens: int = 600,
        load_in_8bit: bool = False,
    ):
        self.model_id = self.SHORTCUTS.get(model_name, model_name)
        self.use_template = model_name == "template"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.pipe = None

        if not self.use_template:
            self._load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers unavailable — falling back to template")
            self.use_template = True
            return
        try:
            logger.info(f"Loading LLM: {self.model_id}")
            kwargs = {}
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                device=None if self.load_in_8bit else (0 if self.device == "cuda" else -1),
                **kwargs,
            )
            logger.info("LLM pipeline ready")
        except Exception as exc:
            logger.error(f"LLM load failed: {exc} — using template fallback")
            self.use_template = True

    # ── Report Generation ─────────────────────────────────────────────────

    def generate(
        self,
        detections: list[dict],
        caption: str,
        vqa_answers: list[dict],
        image_id: str = "IMG-001",
        inspector: str = "Argus AI",
    ) -> dict:
        """
        Generate a full inspection report.

        Parameters
        ----------
        detections  : output from DefectDetector.detect()['detections']
        caption     : output from VLM caption()['caption']
        vqa_answers : output from VLM batch_vqa()
        image_id    : identifier for the inspected image
        inspector   : name of inspector (human or AI)

        Returns
        -------
        dict with keys: report_text, metadata, prompt_used
        """
        t0 = time.perf_counter()

        context = self._build_context(detections, caption, vqa_answers, image_id)

        if self.use_template or self.pipe is None:
            report_text = self._template_report(
                detections, caption, vqa_answers, image_id, inspector
            )
            method = "template"
        else:
            report_text = self._llm_report(context)
            method = self.model_id

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "report_text": report_text,
            "metadata": {
                "image_id": image_id,
                "inspector": inspector,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generation_method": method,
                "inference_ms": elapsed_ms,
                "num_defects": len(detections),
            },
            "prompt_used": context,
        }

    # ── Context Builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_context(detections, caption, vqa_answers, image_id) -> str:
        det_lines = (
            "\n".join(
                f"  - {d['class'].upper()} | confidence={d['confidence']:.2%} "
                f"| severity={d['severity']} | bbox=({d['bbox']['x1']},{d['bbox']['y1']})→"
                f"({d['bbox']['x2']},{d['bbox']['y2']}) | area={d['area_px']}px²"
                for d in detections
            )
            or "  None detected"
        )

        vqa_lines = (
            "\n".join(f"  Q: {r['question']}\n  A: {r['answer']}" for r in vqa_answers)
            or "  No VQA data"
        )

        return textwrap.dedent(f"""
            INSPECTION CONTEXT
            ==================
            Image ID    : {image_id}
            Timestamp   : {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

            DETECTED DEFECTS
            ----------------
            {det_lines}

            IMAGE CAPTION
            -------------
            {caption}

            VISUAL Q&A
            ----------
            {vqa_lines}
        """).strip()

    # ── LLM Report ────────────────────────────────────────────────────────

    def _llm_report(self, context: str) -> str:
        prompt = f"{REPORT_SYSTEM_PROMPT}\n\n{context}\n\nINSPECTION REPORT:\n"
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        generated = outputs[0]["generated_text"]
        return generated[len(prompt) :].strip()

    # ── Template Report ───────────────────────────────────────────────────

    @staticmethod
    def _template_report(
        detections: list[dict],
        caption: str,
        vqa_answers: list[dict],
        image_id: str,
        inspector: str,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        n = len(detections)
        high = [d for d in detections if d["severity"] == "HIGH"]
        medium = [d for d in detections if d["severity"] == "MEDIUM"]
        classes = list({d["class"] for d in detections})

        # VQA lookup helpers
        def vqa_get(keyword: str) -> str:
            for r in vqa_answers:
                if keyword.lower() in r["question"].lower():
                    return r["answer"]
            return "Not determined."

        risk = "CRITICAL" if high else ("MODERATE" if medium else "LOW")

        lines = [
            "=" * 62,
            "          ARGUS INDUSTRIAL INSPECTION REPORT",
            "=" * 62,
            f"  Report ID   : ARGUS-{image_id}-{ts[:10].replace('-', '')}",
            f"  Image ID    : {image_id}",
            f"  Inspector   : {inspector}",
            f"  Timestamp   : {ts}",
            f"  Risk Level  : {risk}",
            "=" * 62,
            "",
            "1. EXECUTIVE SUMMARY",
            "-" * 40,
        ]

        if n == 0:
            lines.append(
                "No defects were detected in the inspected image. The surface "
                "appears to be within acceptable quality parameters."
            )
        else:
            lines.append(
                f"Inspection of image {image_id} identified {n} defect(s) "
                f"classified as: {', '.join(classes)}. "
                f"{'IMMEDIATE action is required.' if high else 'Monitoring is recommended.'}"
            )

        lines += [
            "",
            "2. DEFECT ANALYSIS",
            "-" * 40,
        ]
        if not detections:
            lines.append("  ✓ No defects found.")
        for i, d in enumerate(detections, 1):
            lines += [
                f"  Defect #{i}",
                f"    Type       : {d['class'].replace('_', ' ').title()}",
                f"    Confidence : {d['confidence']:.2%}",
                f"    Severity   : {d['severity']}",
                f"    Location   : ({d['bbox']['x1']}, {d['bbox']['y1']}) → "
                f"({d['bbox']['x2']}, {d['bbox']['y2']})",
                f"    Area       : {d['area_px']:,} px²",
                "",
            ]

        lines += [
            "3. VISUAL DESCRIPTION",
            "-" * 40,
            f"  {caption}",
            "",
            "4. RISK ASSESSMENT",
            "-" * 40,
            f"  Risk Level    : {risk}",
            f"  Defect Count  : {n}",
            f"  High Severity : {len(high)}",
            f"  Medium Severity: {len(medium)}",
            f"  Location      : {vqa_get('where')}",
            f"  Severity Desc : {vqa_get('severe')}",
            "",
            "5. RECOMMENDED ACTIONS",
            "-" * 40,
        ]

        if not detections:
            lines.append("  • Continue routine inspection schedule.")
            lines.append("  • No corrective action required at this time.")
        else:
            if high:
                lines.append("  • ⛔ HALT production — HIGH severity defect(s) detected.")
                lines.append("  • Escalate to senior QC engineer immediately.")
                lines.append("  • Quarantine affected component batch.")
            if medium:
                lines.append("  • ⚠️  Schedule detailed inspection within 24 hours.")
                lines.append("  • Flag component for secondary review.")
            lines.append("  • Document defect coordinates and capture additional angles.")
            lines.append("  • Cross-reference with historical defect database.")
            lines.append("  • Update inspection log and notify production supervisor.")

        lines += [
            "",
            "6. CONCLUSION",
            "-" * 40,
            f"  This automated inspection report was generated by Argus AI "
            f"using YOLOv8 defect detection and vision-language analysis. "
            f"{'All findings require human engineer validation before final disposition.' if n > 0 else 'Surface passed automated quality checks.'}",
            "",
            "=" * 62,
            f"  Generated by Argus v1.0 | {ts}",
            "=" * 62,
        ]

        return "\n".join(lines)
