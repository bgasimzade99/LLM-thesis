from kpi import KPIResult


SYSTEM_PROMPT = """You are a business analyst writing a concise executive report.
You will receive structured KPI data. Your task is to synthesize it into a brief, readable report.
CRITICAL: Use ONLY the numbers and facts provided in the input. Do NOT invent, estimate, or hallucinate any figures.
If a value is not in the input, say "not available in the KPI input". Stick strictly to the given data.
Do NOT add currency symbols ($, €, £). Use numbers exactly as provided.
Do NOT claim "highest", "lowest", or "peak" unless the KPI input explicitly includes peak/lowest info.
Do NOT infer units (avoid phrases like "units sold"). Use "total sales amount" or "sales value".
Do NOT introduce time ranges beyond the provided date range. Use numbers exactly as provided.
Do NOT add extra formatting (bullets, bold markers). Write plain readable text.
Do NOT compute ranges, averages, or percent changes unless those exact values are in the KPI input.
Avoid words like "ranging", "on average", "overall increase" unless the KPI input includes the computed metrics.
Do not use subjective adjectives unless directly supported by numeric thresholds in the input.
List anomalies explicitly instead of summarizing them qualitatively.

Style and structure:
- The report MUST differ in structure and phrasing from a typical template-based report. Use narrative, analytical language instead of enumerating KPI tables. Avoid mirroring sentence structure or section wording from template reports.
- Use conservative academic language. Avoid absolute terms ("entire", "all", "dominant", "majority") unless the KPI input explicitly states 100%. Prefer "represents the largest share" or "accounts for the highest observed value".
- Replace generic adjectives with data-grounded descriptions: e.g. instead of "significant increase", write "sales peaked in November 2018 at X".
- Each recommendation must be explicitly tied to observed patterns and reference at least one KPI value or anomaly.
- The report should read as an analytical interpretation of the KPIs, not as a restatement of the KPI tables.
- Do NOT change any numeric values. Do NOT infer causes beyond the data."""


def build_kpi_summary(kpis: KPIResult) -> str:
    lines = []
    lines.append("## Dataset Overview")
    lines.append(f"- Total sales: {kpis.total_sales}")
    lines.append(f"- Date range: {kpis.date_range[0]} to {kpis.date_range[1]}")
    lines.append(f"- Distinct order periods: {kpis.total_orders}")
    lines.append("")
    lines.append("## Monthly Sales Trend")
    for m in kpis.monthly_trend[-12:]:
        lines.append(f"- {m['month']}: {m['sales']}")
    if kpis.peak_month:
        lines.append(f"- Peak month: {kpis.peak_month} (value: {kpis.peak_value})")
        lines.append(f"- Lowest month: {kpis.lowest_month} (value: {kpis.lowest_value})")
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
        lines.append(f"- Anomaly count: {len(kpis.anomalies)}")
        min_a = min(kpis.anomalies, key=lambda x: x["z_score"])
        max_a = max(kpis.anomalies, key=lambda x: x["z_score"])
        lines.append(f"- Min anomaly z-score: {min_a['z_score']} ({min_a['month']})")
        lines.append(f"- Max anomaly z-score: {max_a['z_score']} ({max_a['month']})")
        for a in kpis.anomalies:
            lines.append(f"- {a['month']}: {a['type']} (sales: {a['sales']}, z-score: {a['z_score']})")
        lines.append("")
    else:
        lines.append("## Detected Anomalies")
        lines.append("- None detected.")
        lines.append("")
    return "\n".join(lines)


def build_user_prompt(kpis: KPIResult) -> str:
    summary = build_kpi_summary(kpis)
    return f"""Based on the following KPI data, write a short executive business report (max 400 words).

Use these exact section headers:
- Executive Report (title)
- Overview
- Trends
- Top Drivers
- Regional Insights
- Anomalies
- Recommendations

For Recommendations:
- Provide exactly 2 recommendations as bullet points.
- Each recommendation must be explicitly tied to observed patterns and reference at least one KPI value or anomaly (e.g., category sales, region share, anomaly month and value).
- Avoid generic phrases like "monitor" unless tied to a specific anomaly month and value.

Remember: Use ONLY the numbers provided below. Do not invent or estimate any figures. Do not infer causes beyond the data.
If a value is not in the input, say "not available in the KPI input".

---
KPI Data:
{summary}
"""
