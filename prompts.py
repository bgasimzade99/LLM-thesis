"""
Prompt generation module. Converts computed KPIs into structured natural language
for LLM consumption. Explicitly instructs the LLM to avoid hallucinations.
"""

from kpi import KPIResult


SYSTEM_PROMPT = """You are a business analyst writing a concise executive report.
You will receive structured KPI data. Your task is to synthesize it into a brief, readable report.
CRITICAL: Use ONLY the numbers and facts provided in the input. Do NOT invent, estimate, or hallucinate any figures.
If a value is not in the input, do not mention it. Stick strictly to the given data."""


def build_kpi_summary(kpis: KPIResult) -> str:
    """Convert KPIs into a concise, structured text summary for the LLM."""
    lines = []

    lines.append("## Dataset Overview")
    lines.append(f"- Total sales: {kpis.total_sales}")
    lines.append(f"- Date range: {kpis.date_range[0]} to {kpis.date_range[1]}")
    lines.append(f"- Distinct order periods: {kpis.total_orders}")
    lines.append("")

    lines.append("## Monthly Sales Trend")
    for m in kpis.monthly_trend[-12:]:  # Last 12 months if available
        lines.append(f"- {m['month']}: {m['sales']}")
    lines.append("")

    lines.append("## Top 5 Categories by Sales")
    for c in kpis.top_categories:
        lines.append(f"- {c['category']}: {c['sales']}")
    lines.append("")

    lines.append("## Top 5 Products by Sales")
    for p in kpis.top_products:
        lines.append(f"- {p['product']}: {p['sales']}")
    lines.append("")

    lines.append("## Sales by Region")
    for r in kpis.regional_distribution:
        lines.append(f"- {r['region']}: {r['sales']} ({r['share_pct']}%)")
    lines.append("")

    if kpis.anomalies:
        lines.append("## Detected Anomalies")
        for a in kpis.anomalies:
            lines.append(f"- {a['month']}: {a['type']} (sales: {a['sales']}, z-score: {a['z_score']})")
        lines.append("")
    else:
        lines.append("## Detected Anomalies")
        lines.append("- None detected.")
        lines.append("")

    return "\n".join(lines)


def build_user_prompt(kpis: KPIResult) -> str:
    """Build the full user prompt with KPI summary and instructions."""
    summary = build_kpi_summary(kpis)
    return f"""Based on the following KPI data, write a short executive business report (max 400 words).

Sections to include:
1. Overview - brief summary of total performance
2. Trends - key patterns in monthly sales
3. Top Drivers - most important categories and products
4. Regional Insights - geographic performance
5. Anomalies - any notable spikes or drops (if applicable)
6. Recommendations - 1-2 actionable suggestions based on the data

Remember: Use ONLY the numbers provided below. Do not invent or estimate any figures.

---
KPI Data:
{summary}
"""
