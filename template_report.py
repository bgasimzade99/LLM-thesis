"""
Template-based report generator.
"""

from kpi import KPIResult


def generate_template_report(kpis: KPIResult) -> str:
    sections = []

    sections.append("## 1. Overview")
    sections.append(
        f"Total sales across the period amount to {kpis.total_sales}. "
        f"The dataset covers orders from {kpis.date_range[0]} to {kpis.date_range[1]}."
    )
    sections.append("")

    sections.append("## 2. Trends")
    if len(kpis.monthly_trend) >= 2:
        latest = kpis.monthly_trend[-1]
        prev = kpis.monthly_trend[-2]
        change = latest["sales"] - prev["sales"]
        pct = (change / prev["sales"] * 100) if prev["sales"] else 0
        direction = "increased" if change >= 0 else "decreased"
        sections.append(
            f"Monthly sales {direction} from {prev['month']} ({prev['sales']}) to "
            f"{latest['month']} ({latest['sales']}). Change: {change:,.2f} ({pct:+.1f}%)."
        )
    else:
        sections.append("Insufficient months for trend analysis.")
    sections.append("")

    sections.append("## 3. Top Drivers")
    if kpis.top_categories:
        top_cat = kpis.top_categories[0]
        sections.append(f"Top category by sales: {top_cat['category']} ({top_cat['sales']}).")
    if kpis.top_products:
        top_prod = kpis.top_products[0]
        sections.append(f"Top product: {top_prod['product']} ({top_prod['sales']}).")
    if kpis.top_categories or kpis.top_products:
        sections.append("Full top-5 lists are available in the KPI tables.")
    else:
        sections.append("No category or product data available.")
    sections.append("")

    sections.append("## 4. Regional Insights")
    if kpis.regional_distribution:
        top_region = kpis.regional_distribution[0]
        sections.append(
            f"Strongest region: {top_region['region']} with {top_region['sales']} "
            f"({top_region['share_pct']}% of total sales)."
        )
        for r in kpis.regional_distribution[1:]:
            sections.append(f"- {r['region']}: {r['sales']} ({r['share_pct']}%)")
    else:
        sections.append("No regional data available.")
    sections.append("")

    sections.append("## 5. Anomalies")
    if kpis.anomalies:
        for a in kpis.anomalies:
            sections.append(
                f"- {a['month']}: {a['type'].upper()} — sales {a['sales']} "
                f"(z-score: {a['z_score']})"
            )
    else:
        sections.append("No significant anomalies detected in monthly sales.")
    sections.append("")

    sections.append("## 6. Recommendations")
    if kpis.top_categories and kpis.regional_distribution:
        top_cat = kpis.top_categories[0]["category"]
        top_reg = kpis.regional_distribution[0]["region"]
        sections.append(
            f"1. Focus inventory and promotions on {top_cat} in {top_reg}, "
            "the top-performing category and region."
        )
    if kpis.anomalies:
        drop_months = [a for a in kpis.anomalies if a["type"] == "drop"]
        if drop_months:
            m = drop_months[0]["month"]
            sections.append(f"2. Investigate the sales drop in {m} for root causes.")
    if not (kpis.top_categories and kpis.regional_distribution) and not kpis.anomalies:
        sections.append("1. Expand the dataset for more robust recommendations.")
        sections.append("2. Monitor monthly trends for emerging patterns.")

    return "\n".join(sections)
