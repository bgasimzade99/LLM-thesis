"""
KPI Definitions — Explicit source, formula, and business justification.

Each KPI is defined with:
- required_columns: dataset columns needed for computation
- formula: short text describing the computation
- business_question: what executive question it answers
- why_selected: justification for inclusion in the report
- output_type: scalar | table | series

All KPIs are derived ONLY from dataset columns. No industry benchmarks.
"""

KPI_DEFINITIONS = {
    "total_sales": {
        "name": "total_sales",
        "required_columns": ["Sales"],
        "formula": "sum(Sales)",
        "business_question": "What is overall sales performance?",
        "why_selected": "Standard executive report overview KPI; primary revenue metric.",
        "interpretation_note": "High values indicate strong aggregate performance; low values may warrant attention.",
        "output_type": "scalar",
    },
    "date_range": {
        "name": "date_range",
        "required_columns": ["Order_Date"],
        "formula": "min(Order_Date), max(Order_Date)",
        "business_question": "Over what time span does the data cover?",
        "why_selected": "Context for all time-based analysis; required for trend interpretation.",
        "interpretation_note": "Wider range provides more context; narrow range may limit generalizability.",
        "output_type": "scalar",
    },
    "total_orders": {
        "name": "total_orders",
        "required_columns": ["Order_Date"],
        "formula": "nunique(Order_Date) or len(rows)",
        "business_question": "How many distinct order periods exist?",
        "why_selected": "Volume context for sales; complements total_sales.",
        "interpretation_note": "Higher count indicates more data points; lower count may reduce reliability.",
        "output_type": "scalar",
    },
    "monthly_sales_trend": {
        "name": "monthly_sales_trend",
        "required_columns": ["Order_Date", "Sales"],
        "formula": "groupby(Order_Date.dt.to_period('M')).sum(Sales)",
        "business_question": "How do sales evolve over time?",
        "why_selected": "Required for trend rules and anomaly detection; standard time-series KPI.",
        "interpretation_note": "Upward trend may indicate growth; downward trend may indicate concern.",
        "output_type": "series",
    },
    "top_categories_by_sales": {
        "name": "top_categories_by_sales",
        "required_columns": ["Category", "Sales"],
        "formula": "groupby(Category).sum(Sales).head(5)",
        "business_question": "Which product categories drive the most revenue?",
        "why_selected": "Key driver analysis; supports concentration rule.",
        "interpretation_note": "Top-ranked categories drive performance; high concentration may imply dependency risk.",
        "output_type": "table",
    },
    "top_products_by_sales": {
        "name": "top_products_by_sales",
        "required_columns": ["Product_Name", "Sales"],
        "formula": "groupby(Product_Name).sum(Sales).head(5)",
        "business_question": "Which specific products drive the most revenue?",
        "why_selected": "Product-level insight; granular complement to category.",
        "interpretation_note": "High values indicate strong product performance; ranking supports prioritization.",
        "output_type": "table",
    },
    "sales_by_region": {
        "name": "sales_by_region",
        "required_columns": ["Region", "Sales"],
        "formula": "groupby(Region).sum(Sales)",
        "business_question": "How is sales distributed geographically?",
        "why_selected": "Geographic performance; supports concentration rule.",
        "interpretation_note": "Balanced distribution may indicate resilience; high concentration may warrant attention.",
        "output_type": "table",
    },
    "region_share": {
        "name": "region_share",
        "required_columns": ["Region", "Sales"],
        "formula": "100 * region_sales / total_sales",
        "business_question": "What percentage of sales does each region contribute?",
        "why_selected": "Concentration risk analysis; dataset-relative share.",
        "interpretation_note": "High share for one region indicates concentration; lower shares indicate spread.",
        "output_type": "table",
    },
    "month_over_month_change": {
        "name": "month_over_month_change",
        "required_columns": ["Order_Date", "Sales"],
        "formula": "100 * (latest_month_sales - prev_month_sales) / prev_month_sales",
        "business_question": "Is sales trending up or down between consecutive months?",
        "why_selected": "Short-term trend signal; feeds into trend rule (positive/negative/stable).",
        "interpretation_note": "Positive change may indicate improvement; negative change may indicate decline.",
        "output_type": "scalar",
    },
    "anomaly_detection": {
        "name": "anomaly_detection",
        "required_columns": ["Order_Date", "Sales"],
        "formula": "z_score = (x - mean(monthly_sales)) / std(monthly_sales); flag if |z| > 1.5",
        "business_question": "Which months deviate significantly from the observed pattern?",
        "why_selected": "Dataset-relative outlier detection; statistical heuristic, not industry benchmark.",
        "interpretation_note": "Flagged months deviate from the observed pattern and may warrant review.",
        "output_type": "table",
    },
    "peak_month": {
        "name": "peak_month",
        "required_columns": ["Order_Date", "Sales"],
        "formula": "argmax(monthly_sales)",
        "business_question": "When did sales peak?",
        "why_selected": "Extreme value for reporting; highlights best period.",
        "interpretation_note": "Identifies the strongest period in the observed range.",
        "output_type": "scalar",
    },
    "lowest_month": {
        "name": "lowest_month",
        "required_columns": ["Order_Date", "Sales"],
        "formula": "argmin(monthly_sales)",
        "business_question": "When did sales bottom out?",
        "why_selected": "Extreme value for reporting; highlights weakest period.",
        "interpretation_note": "Identifies the weakest period; low values may indicate potential concern.",
        "output_type": "scalar",
    },
}


def get_all_required_columns() -> list[str]:
    """Union of all required columns across KPIs. Used for validation."""
    seen: set[str] = set()
    for defn in KPI_DEFINITIONS.values():
        for col in defn["required_columns"]:
            seen.add(col)
    return sorted(seen)
