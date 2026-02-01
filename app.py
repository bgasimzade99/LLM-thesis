"""
LLM-based Decision Support System (DSS) for Business Reporting.
Streamlit UI. Bachelor thesis prototype.
Research-critical: evaluation, hallucination detection, reproducible outputs.
"""

import os
from datetime import datetime

import streamlit as st
import pandas as pd

from kpi import compute_kpis, KPIResult
from prompts import SYSTEM_PROMPT, build_user_prompt, build_kpi_summary
from llm_client import generate_report, get_usage
from template_report import generate_template_report
from evaluation import (
    detect_potential_hallucinations,
    save_hallucination_flags,
    save_evaluation_to_csv,
    RUBRIC_LABELS,
    RUBRIC_DESCRIPTIONS,
    format_rubric_section,
)

OUTPUT_DIR = "outputs"
LLM_REPORTS_DIR = os.path.join(OUTPUT_DIR, "llm_reports")
TEMPLATE_REPORTS_DIR = os.path.join(OUTPUT_DIR, "template_reports")
PROMPTS_DIR = os.path.join(OUTPUT_DIR, "prompts")


def ensure_output_dirs():
    """Create output subdirectories for reproducible saving."""
    for d in [OUTPUT_DIR, LLM_REPORTS_DIR, TEMPLATE_REPORTS_DIR, PROMPTS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_reports_reproducible(
    llm_report: str,
    template_report: str,
    user_prompt: str,
    system_prompt: str,
    dataset_name: str,
) -> tuple[str, str, str]:
    """
    Save reports and prompt to timestamped files for reproducibility.
    Returns (llm_path, template_path, prompt_path).
    """
    ensure_output_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)[:50]

    llm_path = os.path.join(LLM_REPORTS_DIR, f"{ts}_{safe_name}_llm.txt")
    template_path = os.path.join(TEMPLATE_REPORTS_DIR, f"{ts}_{safe_name}_template.txt")
    prompt_path = os.path.join(PROMPTS_DIR, f"{ts}_{safe_name}_prompt.txt")

    with open(llm_path, "w", encoding="utf-8") as f:
        f.write(llm_report)
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template_report)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n=== USER PROMPT ===\n{user_prompt}")

    return llm_path, template_path, prompt_path


def init_session_state():
    """Initialize session state keys for report persistence."""
    if "reports_generated" not in st.session_state:
        st.session_state.reports_generated = False
    if "kpis" not in st.session_state:
        st.session_state.kpis = None
    if "llm_report" not in st.session_state:
        st.session_state.llm_report = None
    if "template_report" not in st.session_state:
        st.session_state.template_report = None
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = "unknown"
    if "suspects" not in st.session_state:
        st.session_state.suspects = []
    if "saved_paths" not in st.session_state:
        st.session_state.saved_paths = None


def main():
    st.set_page_config(page_title="LLM DSS - Business Report", layout="wide")
    st.title("LLM-based Decision Support System")
    st.caption("Bachelor thesis prototype — Compare LLM vs template-based reports")

    init_session_state()

    # --- Data source ---
    train_csv = os.path.join(os.path.dirname(__file__), "train.csv")
    has_train = os.path.isfile(train_csv)

    if has_train:
        data_source = st.radio("Data source", ["Use train.csv", "Upload own CSV"], horizontal=True)
    else:
        data_source = "Upload own CSV"

    if data_source == "Use train.csv" and has_train:
        df = pd.read_csv(train_csv)
        dataset_name = "train.csv"
        st.caption(f"Using **train.csv** ({len(df):,} rows)")
    else:
        uploaded = st.file_uploader("Upload Superstore Sales CSV", type=["csv"])
        if not uploaded:
            st.info("Upload a CSV with columns: Order Date, Category, Sub-Category, Product Name, Region, Sales.")
            return
        df = pd.read_csv(uploaded)
        dataset_name = uploaded.name or "uploaded.csv"
        st.caption(f"Using **{dataset_name}** ({len(df):,} rows)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Rows: {len(df):,}")

    # --- Generate reports ---
    if st.button("Generate Business Report", type="primary"):
        with st.spinner("Computing KPIs..."):
            kpis = compute_kpis(df)

        with st.spinner("Generating template report..."):
            template_report = generate_template_report(kpis)

        with st.spinner("Generating LLM report..."):
            user_prompt = build_user_prompt(kpis)
            llm_report, err = generate_report(SYSTEM_PROMPT, user_prompt)

        if err:
            st.error(err)
            llm_report = None

        suspects = detect_potential_hallucinations(llm_report, kpis) if llm_report else []

        # Reproducible save (auto-save on generation)
        ensure_output_dirs()
        llm_p, template_p, prompt_p = save_reports_reproducible(
            llm_report or "(not generated)",
            template_report,
            user_prompt,
            SYSTEM_PROMPT,
            dataset_name,
        )
        if suspects and llm_report:
            save_hallucination_flags(dataset_name, suspects, llm_report[:500])

        st.session_state.saved_paths = (llm_p, template_p, prompt_p)
        st.session_state.reports_generated = True
        st.session_state.kpis = kpis
        st.session_state.llm_report = llm_report
        st.session_state.template_report = template_report
        st.session_state.user_prompt = user_prompt
        st.session_state.dataset_name = dataset_name
        st.session_state.suspects = suspects
        st.rerun()

    # --- Display reports (side-by-side) ---
    if not st.session_state.reports_generated:
        return

    kpis = st.session_state.kpis
    llm_report = st.session_state.llm_report
    saved_paths = st.session_state.saved_paths
    template_report = st.session_state.template_report
    user_prompt = st.session_state.user_prompt
    dataset_name = st.session_state.dataset_name
    suspects = st.session_state.suspects

    # Saved files info
    if saved_paths:
        with st.expander("Saved outputs (reproducible)", expanded=False):
            st.write(f"- LLM report: `{saved_paths[0]}`")
            st.write(f"- Template report: `{saved_paths[1]}`")
            st.write(f"- Prompt: `{saved_paths[2]}`")

    # KPI Summary (proves we did NOT send raw rows)
    with st.expander("KPI Summary (input to LLM — no raw dataset rows)", expanded=False):
        kpi_summary = build_kpi_summary(kpis)
        st.code(kpi_summary, language=None)
        st.caption("This structured summary is the only data sent to the LLM.")

    # Side-by-side comparison
    st.subheader("Report Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### LLM Report")
        if llm_report:
            st.markdown(llm_report.replace("\n", "  \n"))
            usage = get_usage()
            st.caption(f"Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out | Requests: {usage.request_count}")
            st.download_button(
                "Download LLM Report (txt)",
                data=llm_report,
                file_name=f"llm_report_{dataset_name.replace('.csv','')}.txt",
                mime="text/plain",
                key="dl_llm",
            )
        else:
            st.warning("LLM report not available.")

    with col2:
        st.markdown("#### Template Report")
        st.markdown(template_report.replace("\n", "  \n"))
        st.download_button(
            "Download Template Report (txt)",
            data=template_report,
            file_name=f"template_report_{dataset_name.replace('.csv','')}.txt",
            mime="text/plain",
            key="dl_template",
        )

    # --- Hallucination check ---
    st.subheader("Hallucination Check")
    if suspects:
        st.warning(
            f"Potential invented numbers (not in KPI data): **{', '.join(suspects[:15])}**"
            + (f" … (+{len(suspects)-15} more)" if len(suspects) > 15 else "")
        )
    else:
        st.success("No obvious numeric hallucinations detected.")

    # --- Evaluation rubric ---
    st.subheader("Evaluation Rubric")
    st.caption("Rate the LLM report on each criterion (1–5).")

    scores = {}
    ev_cols = st.columns(4)
    for i, (key, label) in enumerate(RUBRIC_LABELS.items()):
        with ev_cols[i]:
            scores[key] = st.slider(
                label,
                min_value=1,
                max_value=5,
                value=3,
                help=RUBRIC_DESCRIPTIONS[key],
                key=f"slider_{key}",
            )

    comments = st.text_area(
        "Evaluator comments",
        placeholder="Optional notes on the LLM report quality, hallucinations, or comparison with template...",
        key="eval_comments",
    )

    if st.button("Save Evaluation to CSV"):
        save_evaluation_to_csv(
            dataset_name=dataset_name,
            clarity=scores["clarity"],
            correctness=scores["correctness"],
            usefulness=scores["usefulness"],
            consistency=scores["consistency"],
            comments=comments,
            suspicious_count=len(suspects),
        )
        st.success("Evaluation saved to outputs/evaluation_results.csv")

    st.markdown("**Current scores:** " + format_rubric_section(scores).replace("\n", " | "))


if __name__ == "__main__":
    main()
