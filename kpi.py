"""
KPI Computation Module — Deterministic layer only.

All KPIs are derived exclusively from dataset columns: Order_Date (time),
Sales, Product_Name (product), Category, Region.
Validates required columns against KPI_DEFINITIONS before computation.
"""

import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass, field

from kpi_definitions import KPI_DEFINITIONS, get_all_required_columns


def get_kpi_definitions() -> dict[str, dict[str, Any]]:
    """Returns KPI definitions for documentation and traceability."""
    return dict(KPI_DEFINITIONS)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    col_map = {
        "Order Date": "Order_Date",
        "Order_Date": "Order_Date",
        "Category": "Category",
        "Sub-Category": "Sub_Category",
        "Sub_Category": "Sub_Category",
        "Product Name": "Product_Name",
        "Product_Name": "Product_Name",
        "Region": "Region",
        "Sales": "Sales",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


normalize_columns = _normalize_columns


# Column name mapping: definition uses canonical names, dataset may use variants
CANONICAL_COL_MAP = {
    "Order_Date": ["Order_Date", "Order Date"],
    "Sales": ["Sales"],
    "Category": ["Category"],
    "Product_Name": ["Product_Name", "Product Name"],
    "Region": ["Region"],
}


def _get_col(df: pd.DataFrame, canonical: str) -> str:
    """Resolve canonical column name to actual column in df."""
    candidates = CANONICAL_COL_MAP.get(canonical, [canonical])
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Required column not found. Tried: {candidates}")


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required columns for KPI computation exist.
    Raises ValueError with clear message listing missing columns.
    """
    df_norm = _normalize_columns(df.copy())
    required = get_all_required_columns()
    missing: list[str] = []
    for col in required:
        if col not in df_norm.columns:
            candidates = CANONICAL_COL_MAP.get(col, [col])
            if not any(c in df_norm.columns for c in candidates):
                missing.append(col)
    if missing:
        raise ValueError(
            f"Missing required columns for KPI computation: {missing}. "
            f"Available columns: {list(df_norm.columns)}"
        )


@dataclass
class KPIResult:
    """
    Output of deterministic KPI computation.
    All values derived from dataset columns: time, sales, product, region.
    """
    monthly_trend: list[dict[str, Any]] = field(default_factory=list)
    top_categories: list[dict[str, Any]] = field(default_factory=list)
    top_products: list[dict[str, Any]] = field(default_factory=list)
    regional_distribution: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    total_sales: float = 0.0
    total_orders: int = 0
    date_range: tuple[str, str] = ("", "")
    peak_month: str = ""
    peak_value: float = 0.0
    lowest_month: str = ""
    lowest_value: float = 0.0
    month_over_month_change_pct: float | None = None


@dataclass
class KPIComputationResult:
    """
    Full output of compute_kpis: KPI values, summary text, structured table.
    """
    kpis: KPIResult
    kpi_summary_text: str
    kpi_table_df: pd.DataFrame


def _build_kpi_summary_text(kpis: KPIResult) -> str:
    """Build plain text summary of KPIs for prompt and documentation."""
    lines = []
    lines.append("## KPI Summary")
    lines.append(f"- Total sales: {kpis.total_sales}")
    lines.append(f"- Date range: {kpis.date_range[0]} to {kpis.date_range[1]}")
    lines.append(f"- Distinct order periods: {kpis.total_orders}")
    if kpis.month_over_month_change_pct is not None:
        lines.append(f"- Month-over-month change: {kpis.month_over_month_change_pct:+.1f}%")
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
    lines.append("## Anomalies (|z| > 1.5)")
    if kpis.anomalies:
        for a in kpis.anomalies:
            lines.append(f"- {a['month']}: {a['type']} (sales: {a['sales']}, z-score: {a['z_score']})")
    else:
        lines.append("- None detected.")
    return "\n".join(lines)


def _build_kpi_table_df(kpis: KPIResult) -> pd.DataFrame:
    """Build structured KPI table for UI and CSV export."""
    rows: list[dict[str, Any]] = []

    rows.append({"kpi_name": "total_sales", "dimension": "", "value": kpis.total_sales, "extra": ""})
    rows.append({
        "kpi_name": "date_range",
        "dimension": "",
        "value": f"{kpis.date_range[0]} to {kpis.date_range[1]}",
        "extra": "",
    })
    rows.append({"kpi_name": "total_orders", "dimension": "", "value": kpis.total_orders, "extra": ""})
    if kpis.month_over_month_change_pct is not None:
        rows.append({
            "kpi_name": "month_over_month_change",
            "dimension": "",
            "value": kpis.month_over_month_change_pct,
            "extra": "%",
        })

    for m in kpis.monthly_trend:
        rows.append({"kpi_name": "monthly_sales_trend", "dimension": m["month"], "value": m["sales"], "extra": ""})

    for c in kpis.top_categories:
        rows.append({"kpi_name": "top_categories_by_sales", "dimension": c["category"], "value": c["sales"], "extra": ""})

    for p in kpis.top_products:
        rows.append({"kpi_name": "top_products_by_sales", "dimension": p["product"], "value": p["sales"], "extra": ""})

    for r in kpis.regional_distribution:
        rows.append({
            "kpi_name": "sales_by_region",
            "dimension": r["region"],
            "value": r["sales"],
            "extra": f"{r['share_pct']}%",
        })

    for a in kpis.anomalies:
        rows.append({
            "kpi_name": "anomaly_detection",
            "dimension": a["month"],
            "value": a["sales"],
            "extra": f"z={a['z_score']} {a['type']}",
        })

    if kpis.peak_month:
        rows.append({"kpi_name": "peak_month", "dimension": kpis.peak_month, "value": kpis.peak_value, "extra": ""})
    if kpis.lowest_month:
        rows.append({"kpi_name": "lowest_month", "dimension": kpis.lowest_month, "value": kpis.lowest_value, "extra": ""})

    return pd.DataFrame(rows)


def compute_kpis(df: pd.DataFrame) -> KPIComputationResult:
    """
    Compute all KPIs deterministically from dataset.
    Validates required columns against KPI_DEFINITIONS.
    Returns KPIResult, kpi_summary_text, and kpi_table_df.
    """
    df = _normalize_columns(df)
    _validate_required_columns(df)

    date_col = _get_col(df, "Order_Date")
    cat_col = _get_col(df, "Category")
    prod_col = _get_col(df, "Product_Name")
    region_col = _get_col(df, "Region")
    sales_col = _get_col(df, "Sales")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])

    total_sales = float(df[sales_col].sum())
    total_orders = int(df[date_col].nunique()) if date_col else len(df)
    min_date = df[date_col].min().strftime("%Y-%m-%d")
    max_date = df[date_col].max().strftime("%Y-%m-%d")
    date_range = (min_date, max_date)

    df["_month"] = df[date_col].dt.to_period("M")
    monthly = df.groupby("_month")[sales_col].sum().reset_index()
    monthly["_month"] = monthly["_month"].astype(str)
    monthly_trend = [
        {"month": row["_month"], "sales": round(float(row[sales_col]), 2)}
        for _, row in monthly.iterrows()
    ]
    monthly_trend.sort(key=lambda x: x["month"])

    cat_sales = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False).head(5)
    top_categories = [
        {"category": str(c), "sales": round(float(s), 2)}
        for c, s in cat_sales.items()
    ]

    prod_sales = df.groupby(prod_col)[sales_col].sum().sort_values(ascending=False).head(5)
    top_products = [
        {"product": str(p), "sales": round(float(s), 2)}
        for p, s in prod_sales.items()
    ]

    region_sales = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
    regional_distribution = [
        {"region": str(r), "sales": round(float(s), 2), "share_pct": round(100 * s / total_sales, 2)}
        for r, s in region_sales.items()
    ]

    anomalies = _detect_anomalies(monthly_trend)

    peak_month, peak_value = "", 0.0
    lowest_month, lowest_value = "", 0.0
    mom_change_pct: float | None = None

    if monthly_trend:
        by_sales = sorted(monthly_trend, key=lambda x: x["sales"])
        lowest_month, lowest_value = by_sales[0]["month"], by_sales[0]["sales"]
        peak_month, peak_value = by_sales[-1]["month"], by_sales[-1]["sales"]

        if len(monthly_trend) >= 2:
            latest = monthly_trend[-1]
            prev = monthly_trend[-2]
            mom_change_pct = (latest["sales"] - prev["sales"]) / prev["sales"] * 100 if prev["sales"] else 0.0

    kpis = KPIResult(
        monthly_trend=monthly_trend,
        top_categories=top_categories,
        top_products=top_products,
        regional_distribution=regional_distribution,
        anomalies=anomalies,
        total_sales=round(total_sales, 2),
        total_orders=total_orders,
        date_range=date_range,
        peak_month=peak_month,
        peak_value=round(peak_value, 2),
        lowest_month=lowest_month,
        lowest_value=round(lowest_value, 2),
        month_over_month_change_pct=round(mom_change_pct, 2) if mom_change_pct is not None else None,
    )

    kpi_summary_text = _build_kpi_summary_text(kpis)
    kpi_table_df = _build_kpi_table_df(kpis)

    return KPIComputationResult(kpis=kpis, kpi_summary_text=kpi_summary_text, kpi_table_df=kpi_table_df)


def _detect_anomalies(monthly_trend: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Detect months that deviate significantly from the observed distribution.
    Uses z-score (dataset-relative): |z| > 1.5 indicates outlier.
    Threshold 1.5 is a heuristic for moderate outliers; not an industry benchmark.
    """
    if len(monthly_trend) < 3:
        return []
    sales = np.array([m["sales"] for m in monthly_trend])
    mean_s = np.mean(sales)
    std_s = np.std(sales)
    if std_s == 0:
        return []
    z_scores = (sales - mean_s) / std_s
    anomalies = []
    for m, z in zip(monthly_trend, z_scores):
        if abs(z) > 1.5:
            anomalies.append({
                "month": m["month"],
                "sales": m["sales"],
                "type": "spike" if z > 0 else "drop",
                "z_score": round(float(z), 2),
            })
    return anomalies
