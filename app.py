"""
LLM-based DSS for Business Reporting.
"""

import os
from datetime import datetime

import streamlit as st
import pandas as pd

from kpi import compute_kpis, KPIResult, normalize_columns
from prompts import SYSTEM_PROMPT, build_user_prompt, build_kpi_summary
from llm_client import generate_report, get_usage
from template_report import generate_template_report
from evaluation import (
    detect_suspicious_numbers,
    save_hallucination_flags,
    save_evaluation_to_csv,
    RUBRIC_LABELS,
    RUBRIC_DESCRIPTIONS,
    format_rubric_section,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LLM_REPORTS_DIR = os.path.join(OUTPUT_DIR, "llm_reports")
TEMPLATE_REPORTS_DIR = os.path.join(OUTPUT_DIR, "template_reports")
PROMPTS_DIR = os.path.join(OUTPUT_DIR, "prompts")


def ensure_output_dirs():
    for d in [OUTPUT_DIR, LLM_REPORTS_DIR, TEMPLATE_REPORTS_DIR, PROMPTS_DIR]:
        os.makedirs(d, exist_ok=True)


def load_default_dataset() -> tuple[pd.DataFrame | None, str]:
    for path in [
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(BASE_DIR, "train.csv"),
    ]:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            return df, os.path.basename(path)
    return None, ""


def _col(df: pd.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def apply_scenario_filter(df: pd.DataFrame, scenario: str, scenario_value: str) -> pd.DataFrame:
    if scenario == "Full dataset" or not scenario_value:
        return df
    df = normalize_columns(df)
    if scenario == "By Region":
        col = _col(df, "Region")
        if col:
            return df[df[col].astype(str) == scenario_value].copy()
    if scenario == "By Category":
        col = _col(df, "Category")
        if col:
            return df[df[col].astype(str) == scenario_value].copy()
    return df


def get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _metadata_header(run_id: str, dataset_name: str, rows_used: int, scenario: str, scenario_value: str) -> str:
    return f"""---
run_id: {run_id}
dataset_name: {dataset_name}
rows_used: {rows_used}
scenario: {scenario}
scenario_value: {scenario_value}
---

"""


def save_reports_reproducible(
    run_id: str,
    llm_report: str,
    template_report: str,
    user_prompt: str,
    system_prompt: str,
    dataset_name: str,
    rows_used: int,
    scenario: str,
    scenario_value: str,
) -> tuple[str, str, str]:
    ensure_output_dirs()
    header = _metadata_header(run_id, dataset_name, rows_used, scenario, scenario_value)

    prompt_path = os.path.join(PROMPTS_DIR, f"prompt_{run_id}.txt")
    llm_path = os.path.join(LLM_REPORTS_DIR, f"llm_report_{run_id}.md")
    template_path = os.path.join(TEMPLATE_REPORTS_DIR, f"template_report_{run_id}.md")

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n=== USER PROMPT ===\n{user_prompt}")

    with open(llm_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(llm_report)

    with open(template_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(template_report)

    return llm_path, template_path, prompt_path


def init_session_state():
    defaults = {
        "reports_generated": False,
        "run_id": None,
        "kpis": None,
        "llm_report": None,
        "template_report": None,
        "user_prompt": None,
        "kpi_summary": None,
        "dataset_name": "unknown",
        "rows_used": 0,
        "scenario": "Full dataset",
        "scenario_value": "",
        "suspects": [],
        "saved_paths": None,
        "llm_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(page_title="LLM DSS - Business Report", layout="wide")
    st.title("LLM-based Decision Support System")

    ensure_output_dirs()
    init_session_state()

    default_df, default_name = load_default_dataset()
    has_default = default_df is not None

    if has_default:
        data_source = st.radio("Data source", ["Use default dataset", "Upload CSV"], horizontal=True)
    else:
        data_source = "Upload CSV"

    if data_source == "Use default dataset" and has_default:
        df = default_df.copy()
        dataset_name = default_name
        st.caption(f"Using **{dataset_name}** ({len(df):,} rows)")
    else:
        uploaded = st.file_uploader("Upload Superstore Sales CSV", type=["csv"])
        if not uploaded:
            st.info("Upload a CSV or use the default dataset.")
            return
        df = pd.read_csv(uploaded)
        dataset_name = uploaded.name or "uploaded.csv"
        st.caption(f"Using **{dataset_name}** ({len(df):,} rows)")

    st.subheader("Scenario")
    df_norm = normalize_columns(df.copy())
    region_col = _col(df_norm, "Region")
    cat_col = _col(df_norm, "Category")

    scenario_options = ["Full dataset"]
    if region_col:
        scenario_options.append("By Region")
    if cat_col:
        scenario_options.append("By Category")

    scenario = st.selectbox("Filter data", scenario_options, key="scenario_select")

    scenario_value = ""
    if scenario == "By Region" and region_col:
        regions = sorted(df_norm[region_col].dropna().astype(str).unique().tolist())
        scenario_value = st.selectbox("Region", regions, key="region_select")
    elif scenario == "By Category" and cat_col:
        categories = sorted(df_norm[cat_col].dropna().astype(str).unique().tolist())
        scenario_value = st.selectbox("Category", categories, key="category_select")

    df_filtered = apply_scenario_filter(df, scenario, scenario_value)
    rows_used = len(df_filtered)
    st.caption(f"Rows: **{rows_used:,}**")

    st.subheader("Dataset Preview")
    st.dataframe(df_filtered.head(20), use_container_width=True)

    if st.button("Generate Business Report", type="primary"):
        with st.spinner("Computing KPIs..."):
            kpis = compute_kpis(df_filtered)

        with st.spinner("Generating template report..."):
            template_report = generate_template_report(kpis)

        kpi_summary = build_kpi_summary(kpis)
        user_prompt = build_user_prompt(kpis)

        with st.spinner("Generating LLM report..."):
            llm_report, err = generate_report(SYSTEM_PROMPT, user_prompt, env_dir=BASE_DIR)

        llm_error = err
        if err:
            st.warning(f"LLM: {err}")
            if "not found" in err.lower() or "missing" in err.lower():
                with st.expander("Debug: .env path"):
                    st.code(f"BASE_DIR: {BASE_DIR}\n.env: {os.path.join(BASE_DIR, '.env')}\nExists: {os.path.isfile(os.path.join(BASE_DIR, '.env'))}")
            llm_report = None
        else:
            llm_error = None

        suspects = []
        if llm_report:
            suspects = detect_suspicious_numbers(kpi_summary, llm_report)

        run_id = get_run_id()
        llm_p, template_p, prompt_p = save_reports_reproducible(
            run_id=run_id,
            llm_report=llm_report or "(LLM not generated)",
            template_report=template_report,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            dataset_name=dataset_name,
            rows_used=rows_used,
            scenario=scenario,
            scenario_value=scenario_value,
        )

        if suspects and llm_report:
            save_hallucination_flags(run_id, scenario, scenario_value, suspects)

        st.session_state.reports_generated = True
        st.session_state.run_id = run_id
        st.session_state.kpis = kpis
        st.session_state.kpi_summary = kpi_summary
        st.session_state.llm_report = llm_report
        st.session_state.template_report = template_report
        st.session_state.user_prompt = user_prompt
        st.session_state.dataset_name = dataset_name
        st.session_state.rows_used = rows_used
        st.session_state.scenario = scenario
        st.session_state.scenario_value = scenario_value
        st.session_state.suspects = suspects
        st.session_state.llm_error = llm_error
        st.session_state.saved_paths = (llm_p, template_p, prompt_p)
        st.rerun()

    if not st.session_state.reports_generated:
        return

    run_id = st.session_state.run_id
    kpis = st.session_state.kpis
    kpi_summary = st.session_state.kpi_summary
    llm_report = st.session_state.llm_report
    template_report = st.session_state.template_report
    saved_paths = st.session_state.saved_paths
    dataset_name = st.session_state.dataset_name
    rows_used = st.session_state.rows_used
    scenario = st.session_state.scenario
    scenario_value = st.session_state.scenario_value
    suspects = st.session_state.suspects
    llm_error = st.session_state.get("llm_error")

    if saved_paths:
        with st.expander("Saved outputs", expanded=False):
            st.write(f"Run ID: `{run_id}`")
            st.write(f"- LLM: `{saved_paths[0]}`")
            st.write(f"- Template: `{saved_paths[1]}`")
            st.write(f"- Prompt: `{saved_paths[2]}`")

    with st.expander("KPI Summary", expanded=False):
        st.code(kpi_summary, language=None)

    st.subheader("Report Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### LLM Report")
        if llm_report:
            st.markdown(llm_report.replace("\n", "  \n"))
            usage = get_usage()
            st.caption(f"Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
            st.download_button(
                "Download LLM Report",
                data=llm_report,
                file_name=f"llm_report_{run_id}.md",
                mime="text/markdown",
                key="dl_llm",
            )
        else:
            if llm_error and "429" in llm_error:
                st.warning("LLM report not available. OpenAI quota exceeded — add credits at platform.openai.com.")
            elif llm_error:
                st.warning(f"LLM report not available. {llm_error[:200]}...")
            else:
                st.warning("LLM report not available. Add OPENAI_API_KEY to .env.")

    with col2:
        st.markdown("#### Template Report")
        st.markdown(template_report.replace("\n", "  \n"))
        st.download_button(
            "Download Template Report",
            data=template_report,
            file_name=f"template_report_{run_id}.md",
            mime="text/markdown",
            key="dl_template",
        )

    st.subheader("Hallucination Check")
    if suspects:
        st.warning(
            f"Potential invented numbers: **{', '.join(suspects[:15])}**"
            + (f" (+{len(suspects)-15} more)" if len(suspects) > 15 else "")
        )
    else:
        st.success("No numeric hallucinations detected.")

    st.subheader("Evaluation Rubric")
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

    comments = st.text_area("Evaluator comments", placeholder="Optional notes...", key="eval_comments")

    if st.button("Save Evaluation"):
        save_evaluation_to_csv(
            run_id=run_id,
            dataset_name=dataset_name,
            rows_used=rows_used,
            scenario=scenario,
            scenario_value=scenario_value,
            clarity=scores["clarity"],
            correctness=scores["correctness"],
            usefulness=scores["usefulness"],
            consistency=scores["consistency"],
            comments=comments,
            hallucination_count=len(suspects),
            hallucination_values="; ".join(suspects[:50]),
        )
        st.success("Saved to outputs/evaluation_results.csv")

    st.markdown("**Scores:** " + format_rubric_section(scores).replace("\n", " | "))


if __name__ == "__main__":
    main()
