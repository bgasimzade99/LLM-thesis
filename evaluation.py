"""
Evaluation: rubric scoring, hallucination detection, persistent storage.
"""

import csv
import json
import os
import re
from datetime import datetime

from kpi import KPIResult

OUTPUT_DIR = "outputs"
EVAL_CSV = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
HALLUCINATION_JSON = os.path.join(OUTPUT_DIR, "hallucination_flags.json")
NUMERIC_TOLERANCE = 0.02


def extract_numbers(text: str) -> list[float]:
    pattern = r"\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+"
    matches = re.findall(pattern, text)
    nums = []
    for m in matches:
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return nums


def get_allowed_numbers(kpi_summary_text: str) -> set[float]:
    return set(extract_numbers(kpi_summary_text))


def get_reported_numbers(llm_report_text: str) -> list[float]:
    return extract_numbers(llm_report_text)


def _value_matches(a: float, b: float, tol: float = NUMERIC_TOLERANCE) -> bool:
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(a), abs(b)) < tol or abs(a - b) < tol


def detect_suspicious_numbers(
    kpi_summary_text: str,
    llm_report_text: str,
) -> list[str]:
    allowed = get_allowed_numbers(kpi_summary_text)
    reported = get_reported_numbers(llm_report_text)
    suspicious = []
    for n in reported:
        if n == int(n) and 1 <= n <= 6:
            continue
        if not any(_value_matches(n, a) for a in allowed):
            suspicious.append(str(n))
    return suspicious


def detect_potential_hallucinations(llm_text: str, kpis: KPIResult) -> list[str]:
    from prompts import build_kpi_summary
    kpi_summary = build_kpi_summary(kpis)
    return detect_suspicious_numbers(kpi_summary, llm_text)


def save_hallucination_flags(
    run_id: str,
    scenario: str,
    scenario_value: str,
    suspicious_numbers: list[str],
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().isoformat()
    entry = {
        "run_id": run_id,
        "timestamp": ts,
        "scenario": scenario,
        "scenario_value": scenario_value,
        "suspicious_numbers": suspicious_numbers,
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
    run_id: str,
    dataset_name: str,
    rows_used: int,
    scenario: str,
    scenario_value: str,
    clarity: int,
    correctness: int,
    usefulness: int,
    consistency: int,
    comments: str,
    hallucination_count: int,
    hallucination_values: str,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "run_id": run_id,
        "timestamp": ts,
        "dataset_name": dataset_name,
        "rows_used": rows_used,
        "scenario": scenario,
        "scenario_value": scenario_value,
        "clarity": clarity,
        "correctness": correctness,
        "usefulness": usefulness,
        "consistency": consistency,
        "comments": comments,
        "hallucination_count": hallucination_count,
        "hallucination_values": hallucination_values,
    }
    fieldnames = list(row.keys())
    file_exists = os.path.isfile(EVAL_CSV)
    with open(EVAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def format_rubric_section(scores: dict[str, int]) -> str:
    lines = []
    for key, label in RUBRIC_LABELS.items():
        val = scores.get(key, 3)
        lines.append(f"- {label}: {val}/5")
    return "\n".join(lines)
