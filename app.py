import json
import os
import re
from datetime import date, datetime

import streamlit as st
import pandas as pd

from kpi import compute_kpis, normalize_columns, get_kpi_definitions
from kpi_definitions import KPI_DEFINITIONS
from decision_engine import run_decision_engine, RULE_DEFINITIONS
from rule_based_report import generate_rule_based_report, get_rule_definitions
from prompts import SYSTEM_PROMPT, build_user_prompt, build_decision_summary
from llm_client import generate_report, get_usage, check_ollama_available, DEFAULT_MODEL
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
LLM_REPORTS_RAW_DIR = os.path.join(OUTPUT_DIR, "llm_reports_raw")
TEMPLATE_REPORTS_DIR = os.path.join(OUTPUT_DIR, "template_reports")
PROMPTS_DIR = os.path.join(OUTPUT_DIR, "prompts")
KPI_TABLES_DIR = os.path.join(OUTPUT_DIR, "kpi_tables")
DEFINITIONS_DIR = os.path.join(OUTPUT_DIR, "definitions")
DECISIONS_DIR = os.path.join(OUTPUT_DIR, "decisions")
OLLAMA_ENDPOINT = "http://localhost:11434"
MAX_REPORT_WORDS = 400


def ensure_output_dirs():
    for d in [
        OUTPUT_DIR,
        LLM_REPORTS_DIR,
        LLM_REPORTS_RAW_DIR,
        TEMPLATE_REPORTS_DIR,
        PROMPTS_DIR,
        KPI_TABLES_DIR,
        DEFINITIONS_DIR,
        DECISIONS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def _decision_result_to_structured(decision_result) -> dict:
    """Build JSON-serializable view of DecisionResult (no raw KPI data)."""
    dr = decision_result
    out = {
        "fired_rule_ids": list(dr.fired_rule_ids),
        "recommendations": list(dr.recommendations),
        "rule_explanations": list(dr.rule_explanations),
        "trend": None,
        "anomalies": [],
        "concentration": [],
    }
    if dr.trend:
        t = dr.trend
        out["trend"] = {
            "label": t.label,
            "mom_change_pct": t.mom_change_pct,
            "prev_month": t.prev_month,
            "latest_month": t.latest_month,
            "prev_sales": t.prev_sales,
            "latest_sales": t.latest_sales,
        }
    for a in dr.anomalies:
        out["anomalies"].append({
            "month": a.month,
            "sales": a.sales,
            "z_score": a.z_score,
            "type": a.type,
            "label": a.label,
        })
    for c in dr.concentration:
        out["concentration"].append({
            "entity_type": c.entity_type,
            "entity_name": c.entity_name,
            "share_pct": c.share_pct,
            "label": c.label,
        })
    return out


def save_decision_json(
    run_id: str,
    scenario: str,
    scenario_value: str,
    decision_result,
) -> str:
    """Write decision output to outputs/decisions/decision_<run_id>.json."""
    ensure_output_dirs()
    structured = _decision_result_to_structured(decision_result)
    payload = {
        "run_id": run_id,
        "scenario": scenario,
        "scenario_value": scenario_value,
        "fired_rule_ids": structured["fired_rule_ids"],
        "decisions": {
            "trend": structured["trend"],
            "anomalies": structured["anomalies"],
            "concentration": structured["concentration"],
        },
        "recommendations": structured["recommendations"],
        "rule_explanations": structured["rule_explanations"],
    }
    path = os.path.join(DECISIONS_DIR, f"decision_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def save_definitions_and_tables(run_id: str, kpi_table_df: pd.DataFrame) -> None:
    """Save KPI table, KPI definitions, and rule definitions for reproducibility."""
    ensure_output_dirs()
    kpi_table_df.to_csv(os.path.join(KPI_TABLES_DIR, f"kpi_table_{run_id}.csv"), index=False)
    with open(os.path.join(DEFINITIONS_DIR, "kpi_definitions.json"), "w", encoding="utf-8") as f:
        json.dump(KPI_DEFINITIONS, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DEFINITIONS_DIR, "rule_definitions.json"), "w", encoding="utf-8") as f:
        json.dump(get_rule_definitions(), f, indent=2, ensure_ascii=False)


def sanitize_markdown_artifacts(text: str) -> str:
    s = text.replace("**", "").replace("__", "")
    for sym in ("$", "€", "£"):
        s = s.replace(sym, "")
    return s


def _report_with_label_badges(text: str) -> str:
    """Wrap decision labels (NEGATIVE, ATTENTION, POSITIVE, STABLE, OK) in colored badges for UI display."""
    # Colors: negative=red, attention=amber, positive=green, stable=neutral, ok=green
    style = {
        "NEGATIVE": "color:#b91c1c;font-weight:700;",
        "ATTENTION": "color:#b45309;font-weight:700;",
        "POSITIVE": "color:#15803d;font-weight:700;",
        "STABLE": "color:#4b5563;font-weight:700;",
        "OK": "color:#15803d;font-weight:600;",
    }

    def replace_label(m: re.Match) -> str:
        label = m.group(1)
        s = style.get(label, "font-weight:700;")
        return f'<span style="{s}">{label}</span>'

    return re.sub(r"\b(NEGATIVE|POSITIVE|STABLE|ATTENTION|OK)\b", replace_label, text)


def _show_implicit_computation_warning(llm_report: str, kpi_summary: str) -> None:
    problematic = ["ranging", "on average", "overall increase"]
    indicators = ["average", "percent change", "Min anomaly", "Max anomaly"]
    llm_lower = llm_report.lower()
    kpi_lower = kpi_summary.lower()
    has_problematic = any(p in llm_lower for p in problematic)
    has_indicator = any(i.lower() in kpi_lower for i in indicators)
    if has_problematic and not has_indicator:
        st.warning("Possible implicit computation: report uses range/average wording not backed by KPI input.")


def truncate_report(text: str, max_words: int = MAX_REPORT_WORDS) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    truncated = " ".join(words[:max_words])
    return truncated + "\n\n[Report truncated to 400 words.]", True


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


def apply_time_filter(
    df: pd.DataFrame,
    date_col: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Filter rows to those with Order_Date in [start_date, end_date] (inclusive)."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask = (df[date_col] >= start_ts) & (df[date_col] < end_ts)
    return df.loc[mask].copy()


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
    llm_report_raw: str,
    llm_report_sanitized: str,
    template_report: str,
    user_prompt: str,
    system_prompt: str,
    dataset_name: str,
    rows_used: int,
    scenario: str,
    scenario_value: str,
) -> tuple[str, str, str, str]:
    ensure_output_dirs()
    header = _metadata_header(run_id, dataset_name, rows_used, scenario, scenario_value)

    prompt_path = os.path.join(PROMPTS_DIR, f"prompt_{run_id}.txt")
    llm_raw_path = os.path.join(LLM_REPORTS_RAW_DIR, f"llm_report_raw_{run_id}.md")
    llm_path = os.path.join(LLM_REPORTS_DIR, f"llm_report_{run_id}.md")
    template_path = os.path.join(TEMPLATE_REPORTS_DIR, f"template_report_{run_id}.md")
    rule_based_path = os.path.join(TEMPLATE_REPORTS_DIR, f"rule_based_report_{run_id}.md")

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n=== USER PROMPT ===\n{user_prompt}")

    with open(llm_raw_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(llm_report_raw)

    with open(llm_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(llm_report_sanitized)

    with open(template_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(template_report)

    with open(rule_based_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(template_report)

    return llm_path, llm_raw_path, template_path, prompt_path


def init_session_state():
    defaults = {
        "reports_generated": False,
        "run_id": None,
        "kpis": None,
        "kpi_table_df": None,
        "llm_report": None,
        "template_report": None,
        "user_prompt": None,
        "decision_summary": None,
        "decision_result": None,
        "dataset_name": "unknown",
        "rows_used": 0,
        "scenario": "Full dataset",
        "scenario_value": "",
        "saved_time_from": None,
        "saved_time_to": None,
        "suspects": [],
        "saved_paths": None,
        "saved_enable_llm": True,
        "llm_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _inject_css():
    st.markdown("""
    <style>
    .stApp { max-width: 100%; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdown"]) > div { padding-top: 0.5rem; }
    .main-header { font-size: 1.75rem; font-weight: 600; color: #1e3a5f; margin-bottom: 0.25rem; }
    .pipeline-badge { display: inline-block; background: #e8eef4; color: #1e3a5f; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin-right: 0.25rem; margin-bottom: 0.25rem; }
    section[data-testid="stSidebar"] .stMarkdown { font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="LLM DSS - Business Report", layout="wide", initial_sidebar_state="expanded")
    _inject_css()

    ensure_output_dirs()
    init_session_state()

    # ----- Sidebar: Settings -----
    df = None
    dataset_name = ""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.divider()

        default_df, default_name = load_default_dataset()
        has_default = default_df is not None
        if has_default:
            data_source = st.radio("Data source", ["Use default dataset", "Upload CSV"], key="data_source")
        else:
            data_source = "Upload CSV"

        if data_source == "Use default dataset" and has_default:
            df = default_df.copy()
            dataset_name = default_name
            st.caption(f"📁 **{dataset_name}** ({len(df):,} rows)")
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="file_uploader")
            if uploaded:
                df = pd.read_csv(uploaded)
                dataset_name = uploaded.name or "uploaded.csv"
                st.caption(f"📁 **{dataset_name}** ({len(df):,} rows)")
            else:
                st.info("Upload a CSV or use the default dataset.")

        if df is not None:
            st.divider()
            st.markdown("**Scenario**")
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
            df_filtered = normalize_columns(df_filtered.copy())
            date_col = _col(df_filtered, "Order_Date", "Order Date")
            time_from = None
            time_to = None
            st.markdown("**Time period**")
            if date_col:
                df_dates = df_filtered.copy()
                df_dates[date_col] = pd.to_datetime(df_dates[date_col], dayfirst=True, errors="coerce")
                df_dates = df_dates.dropna(subset=[date_col])
                if not df_dates.empty:
                    min_d = df_dates[date_col].min().date()
                    max_d = df_dates[date_col].max().date()
                    time_from = st.date_input("From", value=min_d, min_value=min_d, max_value=max_d, key="time_from")
                    time_to = st.date_input("To", value=max_d, min_value=min_d, max_value=max_d, key="time_to")
                    if time_from and time_to and time_from <= time_to:
                        df_filtered = apply_time_filter(df_filtered, date_col, time_from, time_to)
            else:
                st.caption("No date column")

            rows_used = len(df_filtered)
            st.caption(f"**{rows_used:,}** rows" + (f" · {time_from} → {time_to}" if time_from and time_to else ""))

        st.divider()
        st.markdown("**LLM (optional)**")
        enable_llm = st.checkbox("Enable LLM explanation", value=True, key="enable_llm")
        model_name = st.text_input("Ollama model", value=DEFAULT_MODEL, placeholder="llama3", key="model_input")
        model = model_name or DEFAULT_MODEL
        ollama_available = check_ollama_available()
        if ollama_available:
            st.success("Ollama reachable")
        else:
            st.error("Ollama unreachable")

        with st.expander("KPI definitions", expanded=False):
            for name, defn in get_kpi_definitions().items():
                st.markdown(f"**{name}**")
                f = defn.get("formula", "")
                st.caption(f[:60] + "…" if len(f) > 60 else f)
        with st.expander("Rule definitions", expanded=False):
            for rule_id, defn in get_rule_definitions().items():
                c = defn.get("condition", "")
                st.caption(f"**{rule_id}**: {c[:50]}…")

    # ----- Main area -----
    st.title("LLM-based Decision Support System")
    st.caption(
        "Deterministic pipeline: **KPI** → **Rules** → **Baseline report** → (optional) **LLM explanation**"
    )

    if df is None:
        st.info("👈 Upload a CSV in the sidebar or select **Use default dataset** if available.")
        return

    st.subheader("Dataset preview")
    st.dataframe(df_filtered.head(20), width="stretch")

    if st.button("Generate Business Report", type="primary"):
        # Explicit pipeline: df -> compute_kpis() -> run_decision_engine() -> generate_rule_based_report()
        with st.spinner("Step 1: Deterministic KPI Computation (No AI)..."):
            comp_result = compute_kpis(df_filtered)
            kpis = comp_result.kpis
            kpi_table_df = comp_result.kpi_table_df
            kpi_summary_text = comp_result.kpi_summary_text

        with st.spinner("Step 2: Rule-Based Decision Engine (No AI)..."):
            decision_result = run_decision_engine(kpis)

        with st.spinner("Step 3: Rule-Based Baseline Report (No AI)..."):
            template_report = generate_rule_based_report(decision_result)

        decision_summary = build_decision_summary(decision_result)
        user_prompt = build_user_prompt(decision_result)

        llm_report_raw = None
        err = None
        if enable_llm:
            with st.spinner("Step 4: LLM-Based Explanation Layer (Optional)..."):
                llm_report_raw, err = generate_report(SYSTEM_PROMPT, user_prompt, model=model)

        llm_error = err
        if err:
            st.warning(f"LLM: {err}")
            llm_report_raw = None
        else:
            llm_error = None

        llm_report_sanitized = ""
        if llm_report_raw:
            llm_report_truncated, _ = truncate_report(llm_report_raw)
            llm_report_sanitized = sanitize_markdown_artifacts(llm_report_truncated)
        else:
            llm_report_sanitized = "(LLM not generated)"

        suspects = []
        if llm_report_raw:
            suspects = detect_suspicious_numbers(decision_summary, llm_report_raw)

        run_id = get_run_id()
        save_definitions_and_tables(run_id, kpi_table_df)
        save_decision_json(run_id, scenario, scenario_value, decision_result)
        llm_p, llm_raw_p, template_p, prompt_p = save_reports_reproducible(
            run_id=run_id,
            llm_report_raw=llm_report_raw or "(LLM not generated)",
            llm_report_sanitized=llm_report_sanitized,
            template_report=template_report,
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            dataset_name=dataset_name,
            rows_used=rows_used,
            scenario=scenario,
            scenario_value=scenario_value,
        )

        if suspects and llm_report_raw:
            save_hallucination_flags(run_id, scenario, scenario_value, suspects)

        st.session_state.reports_generated = True
        st.session_state.run_id = run_id
        st.session_state.kpis = kpis
        st.session_state.kpi_table_df = kpi_table_df
        st.session_state.decision_summary = decision_summary
        st.session_state.decision_result = decision_result
        st.session_state.llm_report = llm_report_sanitized
        st.session_state.template_report = template_report
        st.session_state.user_prompt = user_prompt
        st.session_state.dataset_name = dataset_name
        st.session_state.rows_used = rows_used
        st.session_state.scenario = scenario
        st.session_state.scenario_value = scenario_value
        st.session_state.saved_time_from = time_from
        st.session_state.saved_time_to = time_to
        st.session_state.suspects = suspects
        st.session_state.llm_error = llm_error
        st.session_state.saved_enable_llm = enable_llm
        st.session_state.saved_paths = (llm_p, llm_raw_p, template_p, prompt_p)
        st.rerun()

    if not st.session_state.reports_generated:
        return

    run_id = st.session_state.run_id
    kpis = st.session_state.kpis
    kpi_table_df = st.session_state.kpi_table_df
    decision_summary = st.session_state.decision_summary
    decision_result = st.session_state.decision_result
    llm_report = st.session_state.llm_report
    template_report = st.session_state.template_report
    saved_paths = st.session_state.saved_paths
    dataset_name = st.session_state.dataset_name
    rows_used = st.session_state.rows_used
    scenario = st.session_state.scenario
    scenario_value = st.session_state.scenario_value
    suspects = st.session_state.suspects
    llm_error = st.session_state.get("llm_error")

    time_from_s = st.session_state.get("saved_time_from")
    time_to_s = st.session_state.get("saved_time_to")
    period_caption = ""
    if time_from_s and time_to_s:
        period_caption = f" Period: **{time_from_s}** to **{time_to_s}**."

    st.subheader("Report results")
    tab_kpi, tab_baseline, tab_llm = st.tabs(["📊 KPI table", "📋 Rule-based report", "🤖 LLM report"])

    with tab_kpi:
        st.caption("Deterministic KPI output (Step 1)." + period_caption)
        if kpi_table_df is not None and not kpi_table_df.empty:
            st.dataframe(kpi_table_df, width="stretch")
            st.caption("Saved to outputs/kpi_tables/")
        else:
            st.info("KPI table will appear after report generation.")

    with tab_baseline:
        st.caption("Deterministic baseline from the rule engine (Steps 2–3)." + period_caption)
        report_display = template_report.replace("\n", "  \n")
        report_display = _report_with_label_badges(report_display)
        st.markdown(report_display, unsafe_allow_html=True)
        st.download_button(
            "Download Rule-Based Report",
            data=template_report,
            file_name=f"rule_based_report_{run_id}.md",
            mime="text/markdown",
            key="dl_template",
        )

    with tab_llm:
        st.caption("Natural-language explanation of the same rule outcomes (Step 4, optional)." + period_caption)
        st.caption("_LLM explanation is generated only from rule-based decision output._")
        st.warning(
            "LLM output does not influence KPI values or decisions. It only verbalizes deterministic outcomes."
        )
        if llm_report:
            st.code(llm_report, language=None)
            usage = get_usage()
            st.caption(f"Requests: {usage.request_count}")
            st.download_button(
                "Download LLM Report",
                data=llm_report,
                file_name=f"llm_report_{run_id}.md",
                mime="text/markdown",
                key="dl_llm",
            )
        else:
            if not st.session_state.get("saved_enable_llm", True):
                st.info("LLM explanation layer was disabled for this run. Only the rule-based baseline report was generated.")
            elif llm_error:
                st.warning(f"LLM report not available. {llm_error[:250]}...")
            else:
                st.warning("LLM report not available. Ensure Ollama is running (ollama serve).")

    st.divider()
    st.subheader("Details & evaluation")

    col_halluc, col_eval = st.columns([1, 1])
    with col_halluc:
        st.markdown("**Hallucination check**")
        if suspects:
            st.warning(
                f"Potential invented numbers: **{', '.join(suspects[:10])}**"
                + (f" (+{len(suspects)-10} more)" if len(suspects) > 10 else "")
            )
        else:
            st.success("No numeric hallucinations detected.")

    with col_eval:
        st.markdown("**Evaluation rubric**")
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
        st.caption("_Scores are assigned by human evaluator for comparative analysis._")
        comments = st.text_area("Evaluator comments", placeholder="Optional notes...", key="eval_comments")
        if st.button("Save Evaluation", key="save_eval_btn"):
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
        st.caption(format_rubric_section(scores).replace("\n", " · "))

    _show_implicit_computation_warning(llm_report or "", decision_summary)

    with st.expander("Fired rules (decision engine)", expanded=False):
        if decision_result is not None and decision_result.fired_rule_ids:
            for rid in decision_result.fired_rule_ids:
                defn = RULE_DEFINITIONS.get(rid, {})
                st.markdown(f"**{rid}** — {defn.get('condition', 'N/A')}")
                st.caption(f"Threshold: {defn.get('threshold', 'N/A')} · {defn.get('why_threshold', '')[:80]}…")
                st.divider()
        else:
            st.caption("No rules fired for this run.")

    with st.expander("Decision output (structured JSON)", expanded=False):
        if decision_result is not None:
            st.json(_decision_result_to_structured(decision_result))
        else:
            st.caption("Decision output will appear after report generation.")

    if saved_paths:
        with st.expander("Saved file paths", expanded=False):
            st.code(
                f"Run ID: {run_id}\n"
                f"KPI table: {os.path.join(KPI_TABLES_DIR, f'kpi_table_{run_id}.csv')}\n"
                f"Decision: {os.path.join(DECISIONS_DIR, f'decision_{run_id}.json')}\n"
                f"Rule-based: {os.path.join(TEMPLATE_REPORTS_DIR, f'rule_based_report_{run_id}.md')}\n"
                f"LLM report: {saved_paths[0]}",
                language=None,
            )

    with st.expander("Decision input (to LLM)", expanded=False):
        st.code(decision_summary, language=None)
        st.caption("Input to LLM: rule outcomes only. No raw rows.")


if __name__ == "__main__":
    main()
