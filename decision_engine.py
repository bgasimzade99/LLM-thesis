"""
Rule-Based Decision Engine — Core deterministic logic (Type 3 thesis).

All decisions (positive / negative / attention) are made HERE, before any LLM.
The LLM receives ONLY the output of this engine and may only explain it.

THRESHOLD POLICY (academic clarity):
- All thresholds are dataset-relative heuristics. They are NOT industry benchmarks.
- Their purpose is controlled comparison and reproducible baseline logic, not optimal
  decision-making. This allows fair evaluation of the LLM explanation layer.
"""

from dataclasses import dataclass, field
from typing import Any

from kpi import KPIResult


# -----------------------------------------------------------------------------
# RULE DEFINITIONS — Explicit thresholds with justification
# -----------------------------------------------------------------------------
# All thresholds are dataset-relative heuristics; NOT industry benchmarks.
# Purpose: controlled comparison and reproducible baseline, not optimal decision-making.
# -----------------------------------------------------------------------------

RULE_DEFINITIONS = {
    "mom_positive": {
        "id": "mom_positive",
        "condition": "Month-over-month sales change >= +10%",
        "threshold": 10.0,
        "unit": "percent",
        "why_threshold": "Heuristic for meaningful upward change; avoids labeling tiny fluctuations. Dataset-relative.",
        "message_template": "Month-over-month change {pct:+.1f}% >= +{threshold}%. Label: positive trend.",
        "business_rationale": "Reporting systems flag sustained upward trends so management can reinforce successful drivers.",
    },
    "mom_negative": {
        "id": "mom_negative",
        "condition": "Month-over-month sales change <= -10%",
        "threshold": -10.0,
        "unit": "percent",
        "why_threshold": "Heuristic for meaningful downward change; dataset-relative.",
        "message_template": "Month-over-month change {pct:+.1f}% <= {threshold}%. Label: negative trend.",
        "business_rationale": "Reporting systems surface declines early so stakeholders can investigate causes.",
    },
    "mom_stable": {
        "id": "mom_stable",
        "condition": "Month-over-month sales change between -10% and +10%",
        "threshold": None,
        "unit": "percent",
        "why_threshold": "Within normal fluctuation band; dataset-relative.",
        "message_template": "Month-over-month change {pct:+.1f}% within ±10%. Label: stable.",
        "business_rationale": "Reporting systems distinguish stable periods from material trends to avoid alarm fatigue.",
    },
    "anomaly_z": {
        "id": "anomaly_z",
        "condition": "|z-score| > 1.5 for monthly sales",
        "threshold": 1.5,
        "unit": "z_score",
        "why_threshold": "Common heuristic for moderate statistical outliers (~87% within ±1.5). Dataset-relative.",
        "message_template": "Anomaly in {month}: z-score {z_score} exceeds |{threshold}|. Label: attention required.",
        "business_rationale": "Reporting systems highlight months that deviate from the observed pattern for review.",
    },
    "region_concentration": {
        "id": "region_concentration",
        "condition": "Top region share >= 30%",
        "threshold": 30.0,
        "unit": "percent",
        "why_threshold": "Indicates over-reliance on single region; heuristic, not industry benchmark.",
        "message_template": "Top region '{region}' share {share:.1f}% >= {threshold}%. Label: attention.",
        "business_rationale": "Reporting systems flag geographic concentration so diversification can be considered.",
    },
    "category_key_driver": {
        "id": "category_key_driver",
        "condition": "Top category share >= 35%",
        "threshold": 35.0,
        "unit": "percent",
        "why_threshold": "Indicates dominant category; key driver note. Heuristic, dataset-relative.",
        "message_template": "Top category '{category}' share {share:.1f}% >= {threshold}%. Label: attention.",
        "business_rationale": "Reporting systems call out dominant categories so dependency risk is visible.",
    },
}


# Thresholds derived from RULE_DEFINITIONS (for computation)
MOM_POSITIVE_THRESHOLD_PCT = RULE_DEFINITIONS["mom_positive"]["threshold"]
MOM_NEGATIVE_THRESHOLD_PCT = RULE_DEFINITIONS["mom_negative"]["threshold"]
ANOMALY_Z_THRESHOLD = RULE_DEFINITIONS["anomaly_z"]["threshold"]
CONCENTRATION_REGION_THRESHOLD_PCT = RULE_DEFINITIONS["region_concentration"]["threshold"]
CONCENTRATION_CATEGORY_THRESHOLD_PCT = RULE_DEFINITIONS["category_key_driver"]["threshold"]


@dataclass
class TrendDecision:
    """Outcome of trend rule: positive, negative, or stable."""
    label: str  # "positive" | "negative" | "stable"
    mom_change_pct: float
    prev_month: str
    latest_month: str
    prev_sales: float
    latest_sales: float
    rule_explanation: str


@dataclass
class AnomalyDecision:
    """Outcome of anomaly rule: attention required."""
    month: str
    sales: float
    z_score: float
    type: str  # "spike" | "drop"
    label: str  # "attention"
    rule_explanation: str


@dataclass
class ConcentrationDecision:
    """Outcome of concentration rule: attention or ok."""
    entity_type: str  # "category" | "region"
    entity_name: str
    share_pct: float
    label: str  # "attention" | "ok"
    rule_explanation: str


@dataclass
class DecisionResult:
    """
    Output of rule-based decision engine.
    Contains: KPI values, decision labels, rule explanations, fired rules, recommendations.
    This is the ONLY input the LLM receives — not raw KPIs.
    """
    kpis: KPIResult
    trend: TrendDecision | None
    anomalies: list[AnomalyDecision]
    concentration: list[ConcentrationDecision]
    # Rule IDs that fired (for traceability)
    fired_rule_ids: list[str] = field(default_factory=list)
    # Recommendations generated ONLY by deterministic rules (no freehand interpretation)
    recommendations: list[str] = field(default_factory=list)
    # Flattened text of all rule explanations for traceability
    rule_explanations: list[str] = field(default_factory=list)


def run_decision_engine(kpis: KPIResult) -> DecisionResult:
    """
    Apply deterministic rules to KPIs. Returns decisions + explanations + fired_rule_ids + recommendations.
    The LLM may only interpret these outcomes; it does NOT decide.
    All thresholds are heuristics for consistent baseline comparison, NOT industry benchmarks.
    """
    rule_explanations: list[str] = []
    fired_rule_ids: list[str] = []
    recommendations: list[str] = []
    trend_decision: TrendDecision | None = None
    anomaly_decisions: list[AnomalyDecision] = []
    concentration_decisions: list[ConcentrationDecision] = []

    # -------------------------------------------------------------------------
    # RULE 1: Month-over-month trend
    # -------------------------------------------------------------------------
    # Thresholds ±10% are heuristics (not industry benchmarks); dataset-relative.
    if len(kpis.monthly_trend) >= 2:
        latest = kpis.monthly_trend[-1]
        prev = kpis.monthly_trend[-2]
        change = latest["sales"] - prev["sales"]
        pct = (change / prev["sales"] * 100) if prev["sales"] else 0.0

        if pct >= MOM_POSITIVE_THRESHOLD_PCT:
            label = "positive"
            rule_id = "mom_positive"
            rule_explanation = (
                f"Month-over-month change {pct:+.1f}% >= +{MOM_POSITIVE_THRESHOLD_PCT}% "
                f"(heuristic threshold). Label: positive trend."
            )
        elif pct <= MOM_NEGATIVE_THRESHOLD_PCT:
            label = "negative"
            rule_id = "mom_negative"
            rule_explanation = (
                f"Month-over-month change {pct:+.1f}% <= {MOM_NEGATIVE_THRESHOLD_PCT}% "
                f"(heuristic threshold). Label: negative trend."
            )
            recommendations.append("Investigate drivers behind the MoM decline and consider corrective actions.")
        else:
            label = "stable"
            rule_id = "mom_stable"
            rule_explanation = (
                f"Month-over-month change {pct:+.1f}% within ±{MOM_POSITIVE_THRESHOLD_PCT}%. "
                f"Label: stable."
            )

        fired_rule_ids.append(rule_id)
        trend_decision = TrendDecision(
            label=label,
            mom_change_pct=pct,
            prev_month=prev["month"],
            latest_month=latest["month"],
            prev_sales=prev["sales"],
            latest_sales=latest["sales"],
            rule_explanation=rule_explanation,
        )
        rule_explanations.append(rule_explanation)

    # -------------------------------------------------------------------------
    # RULE 2: Anomaly attention
    # -------------------------------------------------------------------------
    # Threshold |z| > 1.5 is heuristic (dataset-relative); not industry benchmark.
    if kpis.anomalies and "anomaly_z" not in fired_rule_ids:
        fired_rule_ids.append("anomaly_z")
        recommendations.append("Review the anomalous month(s) for root causes and one-off effects.")
    for a in kpis.anomalies:
        rule_explanation = (
            f"Anomaly in {a['month']}: z-score {a['z_score']} exceeds |{ANOMALY_Z_THRESHOLD}| "
            f"(dataset-relative). Label: attention required."
        )
        anomaly_decisions.append(
            AnomalyDecision(
                month=a["month"],
                sales=a["sales"],
                z_score=a["z_score"],
                type=a["type"],
                label="attention",
                rule_explanation=rule_explanation,
            )
        )
        rule_explanations.append(rule_explanation)

    # -------------------------------------------------------------------------
    # RULE 3: Concentration risk (category)
    # -------------------------------------------------------------------------
    # Threshold 35% is heuristic (dataset-relative); not industry benchmark.
    if kpis.top_categories and kpis.total_sales > 0:
        top_cat = kpis.top_categories[0]
        share = 100 * top_cat["sales"] / kpis.total_sales
        if share >= CONCENTRATION_CATEGORY_THRESHOLD_PCT:
            fired_rule_ids.append("category_key_driver")
            recommendations.append("Maintain focus on top category while monitoring dependency risk.")
            rule_explanation = (
                f"Top category '{top_cat['category']}' share {share:.1f}% >= "
                f"{CONCENTRATION_CATEGORY_THRESHOLD_PCT}% (heuristic). Label: attention."
            )
            concentration_decisions.append(
                ConcentrationDecision(
                    entity_type="category",
                    entity_name=top_cat["category"],
                    share_pct=share,
                    label="attention",
                    rule_explanation=rule_explanation,
                )
            )
            rule_explanations.append(rule_explanation)
        else:
            rule_explanation = (
                f"Top category share {share:.1f}% below {CONCENTRATION_CATEGORY_THRESHOLD_PCT}%. "
                f"Label: ok."
            )
            concentration_decisions.append(
                ConcentrationDecision(
                    entity_type="category",
                    entity_name=top_cat["category"],
                    share_pct=share,
                    label="ok",
                    rule_explanation=rule_explanation,
                )
            )

    # -------------------------------------------------------------------------
    # RULE 4: Concentration risk (region)
    # -------------------------------------------------------------------------
    # Threshold 30% is heuristic (dataset-relative); not industry benchmark.
    if kpis.regional_distribution and kpis.total_sales > 0:
        top_reg = kpis.regional_distribution[0]
        share = top_reg["share_pct"]
        if share >= CONCENTRATION_REGION_THRESHOLD_PCT:
            fired_rule_ids.append("region_concentration")
            recommendations.append("Consider diversification across regions to reduce concentration risk.")
            rule_explanation = (
                f"Top region '{top_reg['region']}' share {share:.1f}% >= "
                f"{CONCENTRATION_REGION_THRESHOLD_PCT}% (heuristic). Label: attention."
            )
            concentration_decisions.append(
                ConcentrationDecision(
                    entity_type="region",
                    entity_name=top_reg["region"],
                    share_pct=share,
                    label="attention",
                    rule_explanation=rule_explanation,
                )
            )
            rule_explanations.append(rule_explanation)
        else:
            rule_explanation = (
                f"Top region share {share:.1f}% below {CONCENTRATION_REGION_THRESHOLD_PCT}%. "
                f"Label: ok."
            )
            concentration_decisions.append(
                ConcentrationDecision(
                    entity_type="region",
                    entity_name=top_reg["region"],
                    share_pct=share,
                    label="ok",
                    rule_explanation=rule_explanation,
                )
            )

    return DecisionResult(
        kpis=kpis,
        trend=trend_decision,
        anomalies=anomaly_decisions,
        concentration=concentration_decisions,
        fired_rule_ids=fired_rule_ids,
        recommendations=recommendations,
        rule_explanations=rule_explanations,
    )
