"""
Evaluation module: rubric scoring, hallucination detection, and persistent storage.
Saves evaluation results to CSV and hallucination flags to JSON for research reproducibility.
"""

import csv
import json
import os
import re
from datetime import datetime
from typing import Any

from kpi import KPIResult

OUTPUT_DIR = "outputs"
EVAL_CSV = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
HALLUCINATION_JSON = os.path.join(OUTPUT_DIR, "hallucination_flags.json")

# Tolerance for numeric comparison (handles rounding, e.g. 25.0 vs 25)
NUMERIC_TOLERANCE = 0.01


def extract_numbers_from_text(text: str) -> set[float]:
    """
    Extract numeric values from text for hallucination check.
    Handles: 1,234.56 | 1234 | 25% (extracts 25) | 25.5
    """
    # Match numbers with optional commas and decimals
    pattern = r"\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+"
    matches = re.findall(pattern, text)
    nums = set()
    for m in matches:
        try:
            nums.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return nums


def get_kpi_numbers(kpis: KPIResult) -> set[float]:
    """Collect all allowed numeric values from the computed KPIs."""
    nums = set()
    nums.add(kpis.total_sales)
    for d in kpis.date_range:
        if "-" in d:
            try:
                nums.add(float(d.split("-")[0]))
            except (ValueError, IndexError):
                pass
    for m in kpis.monthly_trend:
        nums.add(m["sales"])
    for c in kpis.top_categories:
        nums.add(c["sales"])
    for p in kpis.top_products:
        nums.add(p["sales"])
    for r in kpis.regional_distribution:
        nums.add(r["sales"])
        nums.add(r["share_pct"])
    for a in kpis.anomalies:
        nums.add(a["sales"])
        nums.add(a["z_score"])
    return nums


def _matches_kpi(value: float, kpi_nums: set[float]) -> bool:
    """Check if value is within tolerance of any KPI number."""
    return any(abs(value - k) < NUMERIC_TOLERANCE for k in kpi_nums)


def detect_potential_hallucinations(llm_text: str, kpis: KPIResult) -> list[str]:
    """
    Flag numbers in LLM output not present in KPI data.
    Filters section numbers (1–6) and allows formatting tolerance.
    Returns list of suspect values as strings.
    """
    kpi_nums = get_kpi_numbers(kpis)
    llm_nums = extract_numbers_from_text(llm_text)
    suspects = []
    for n in llm_nums:
        if n == int(n) and 1 <= n <= 6:
            continue  # Likely section/list numbering
        if not _matches_kpi(n, kpi_nums):
            suspects.append(str(n))
    return suspects


def save_hallucination_flags(
    dataset_name: str,
    suspicious_numbers: list[str],
    llm_report_preview: str = "",
) -> str:
    """
    Save hallucination flags to outputs/hallucination_flags.json.
    Append new entry; file contains list of flag objects.
    Returns path to saved file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().isoformat()
    entry = {
        "timestamp": ts,
        "dataset_name": dataset_name,
        "suspicious_numbers": suspicious_numbers,
        "llm_report_preview": llm_report_preview[:500] if llm_report_preview else "",
    }
    existing = []
    if os.path.isfile(HALLUCINATION_JSON):
        try:
            with open(HALLUCINATION_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []
    existing.append(entry)
    with open(HALLUCINATION_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    return HALLUCINATION_JSON


def save_evaluation_to_csv(
    dataset_name: str,
    clarity: int,
    correctness: int,
    usefulness: int,
    consistency: int,
    comments: str,
    suspicious_count: int = 0,
) -> str:
    """
    Append evaluation row to outputs/evaluation_results.csv.
    Creates file with header if missing.
    Returns path to CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": ts,
        "dataset_name": dataset_name,
        "clarity": clarity,
        "correctness": correctness,
        "usefulness": usefulness,
        "consistency": consistency,
        "comments": comments,
        "suspicious_numbers_count": suspicious_count,
    }
    file_exists = os.path.isfile(EVAL_CSV)
    with open(EVAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "dataset_name",
                "clarity",
                "correctness",
                "usefulness",
                "consistency",
                "comments",
                "suspicious_numbers_count",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return EVAL_CSV


RUBRIC_LABELS = {
    "clarity": "Clarity",
    "correctness": "Correctness",
    "usefulness": "Usefulness",
    "consistency": "Consistency",
}

RUBRIC_DESCRIPTIONS = {
    "clarity": "Report is easy to read and understand.",
    "correctness": "Facts and numbers align with the data.",
    "usefulness": "Report provides actionable business value.",
    "consistency": "Structure and style are coherent throughout.",
}


def get_default_scores() -> dict[str, int]:
    """Default rubric scores (middle value)."""
    return {k: 3 for k in RUBRIC_LABELS}


def format_rubric_section(scores: dict[str, int]) -> str:
    """Format scores as a readable section."""
    lines = []
    for key, label in RUBRIC_LABELS.items():
        val = scores.get(key, 3)
        lines.append(f"- {label}: {val}/5")
    return "\n".join(lines)
