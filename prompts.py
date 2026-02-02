"""
Prompt Design — LLM receives ONLY rule-based decision output (Type 3 thesis).

The LLM is an explanation layer ONLY. It must:
- Interpret rule outcomes (e.g., "negative trend detected", "attention required")
- NOT decide whether performance is good or bad — the rule engine already did
- NOT perform calculations or introduce new numbers
"""

from decision_engine import DecisionResult


# -----------------------------------------------------------------------------
# SYSTEM PROMPT — LLM as explanation layer only
# -----------------------------------------------------------------------------
# The LLM does NOT make decisions. It receives pre-computed decision labels
# (positive/negative/attention) and explains them in natural language.
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an explanation layer for a business reporting system.
You will receive RULE-BASED DECISION OUTCOMES — not raw data.

Your role:
- Interpret the pre-computed decisions (positive / negative / attention) in readable language
- Explain what the rule engine determined and why (from the rule explanations)
- Do NOT decide whether performance is good or bad — the system already classified outcomes
- Do NOT perform any calculations or introduce new numbers
- Use ONLY the values and labels provided in the input

CRITICAL:
- Use ONLY the numbers and facts in the input. Do NOT invent, estimate, or hallucinate figures.
- If a value is not in the input, say "not available in the decision input".
- Do NOT add currency symbols ($, €, £).
- Do NOT compute averages, percent changes, or ranges — use only those in the input.
- Do NOT use subjective adjectives like "significant" unless the decision label explicitly supports it.
- Phrase explanations as: "The rule engine detected X" or "The system classified this as Y".

Style:
- Write plain, readable text. No extra formatting (bullets, bold).
- Use conservative language. Avoid absolute terms unless data supports 100%.
- Each recommendation must reference a specific decision outcome (e.g., "Given the attention flag on region X").
- The report should read as an explanation of rule outcomes, not as independent analysis."""


def build_decision_summary(decision_result: DecisionResult) -> str:
    """
    Build the structured input for the LLM.
    Contains ONLY: KPI values, decision labels, rule explanations.
    No raw row-level data. The LLM interprets these outcomes.
    """
    kpis = decision_result.kpis
    lines = []

    lines.append("## Decision Input (Rule-Based Outcomes)")
    lines.append("")
    lines.append("### Dataset Overview")
    lines.append(f"- Total sales: {kpis.total_sales}")
    lines.append(f"- Date range: {kpis.date_range[0]} to {kpis.date_range[1]}")
    lines.append(f"- Distinct order periods: {kpis.total_orders}")
    lines.append("")

    lines.append("### Trend Decision")
    if decision_result.trend:
        t = decision_result.trend
        lines.append(f"- Label: {t.label.upper()}")
        lines.append(f"- Month-over-month change: {t.mom_change_pct:+.1f}%")
        lines.append(f"- From {t.prev_month} ({t.prev_sales}) to {t.latest_month} ({t.latest_sales})")
        lines.append(f"- Rule explanation: {t.rule_explanation}")
    else:
        lines.append("- Insufficient data for trend decision.")
    lines.append("")

    lines.append("### Top 5 Categories by Sales")
    for c in kpis.top_categories:
        lines.append(f"- {c['category']}: {c['sales']}")
    lines.append("")

    lines.append("### Top 5 Products by Sales")
    for p in kpis.top_products:
        lines.append(f"- {p['product']}: {p['sales']}")
    lines.append("")

    lines.append("### Sales by Region")
    for r in kpis.regional_distribution:
        lines.append(f"- {r['region']}: {r['sales']} ({r['share_pct']}%)")
    lines.append("")

    lines.append("### Concentration Decisions")
    for c in decision_result.concentration:
        lines.append(f"- {c.entity_type.title()} '{c.entity_name}': share {c.share_pct:.1f}% — Label: {c.label.upper()}")
        lines.append(f"  {c.rule_explanation}")
    if not decision_result.concentration:
        lines.append("- No concentration decisions.")
    lines.append("")

    lines.append("### Anomaly Decisions")
    if decision_result.anomalies:
        for a in decision_result.anomalies:
            lines.append(f"- {a.month}: {a.type} — sales {a.sales}, z-score {a.z_score}")
            lines.append(f"  Label: {a.label.upper()} — {a.rule_explanation}")
    else:
        lines.append("- No anomalies detected.")
    lines.append("")

    if kpis.peak_month:
        lines.append("### Extremes")
        lines.append(f"- Peak month: {kpis.peak_month} (value: {kpis.peak_value})")
        lines.append(f"- Lowest month: {kpis.lowest_month} (value: {kpis.lowest_value})")
    lines.append("")

    return "\n".join(lines)


def build_user_prompt(decision_result: DecisionResult) -> str:
    """
    Build user prompt for LLM. Input is ONLY the decision summary.
    The LLM must explain rule outcomes, not analyze raw data.
    """
    summary = build_decision_summary(decision_result)
    return f"""Based on the following RULE-BASED DECISION OUTCOMES, write a short executive report (max 400 words).


Your task: Explain what the system determined. Do NOT make your own judgments about good/bad.
Use the decision labels (positive, negative, attention) as given. Explain them in plain language.

Use these exact section headers:
- Executive Report (title)
- Overview
- Trends
- Top Drivers
- Regional Insights
- Anomalies
- Recommendations

For Recommendations:
- Provide exactly 2 recommendations.
- Each must reference a specific decision outcome (e.g., "Given the attention flag on ...", "The negative trend in ...").
- Do NOT invent causes or external factors.

Remember: Use ONLY the values below. Do not invent or estimate. You are explaining rule outcomes, not analyzing data.

---
Decision Input:
{summary}
"""


# Backward compatibility: build_kpi_summary was used for hallucination detection.
# Now we use decision summary which contains the same numbers.
def build_kpi_summary(decision_result: DecisionResult) -> str:
    """
    Alias for build_decision_summary. Used by evaluation/hallucination check.
    """
    return build_decision_summary(decision_result)
