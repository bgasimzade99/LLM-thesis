import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass, field


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


def _get_col(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Required column not found. Tried: {candidates}")


@dataclass
class KPIResult:
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


def compute_kpis(df: pd.DataFrame) -> KPIResult:
    df = _normalize_columns(df)

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
    if monthly_trend:
        by_sales = sorted(monthly_trend, key=lambda x: x["sales"])
        lowest_month, lowest_value = by_sales[0]["month"], by_sales[0]["sales"]
        peak_month, peak_value = by_sales[-1]["month"], by_sales[-1]["sales"]

    return KPIResult(
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
    )


def _detect_anomalies(monthly_trend: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
