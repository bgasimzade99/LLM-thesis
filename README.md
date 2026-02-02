# LLM-based DSS for Business Reporting

Bachelor thesis prototype (Type 3 systems engineering). Compares LLM-generated reports with rule-based reports. Uses Ollama for local LLM (no API keys).

## Why this is a Type 3 Thesis

- **System architecture ownership:** The student designs and implements the full pipeline (KPI computation, rule engine, baseline report, optional LLM layer). The system is not a black-box AI product.
- **Deterministic core:** All KPIs and all decisions (positive / negative / attention) are computed by deterministic code. No AI is used for numbers or labels.
- **AI as auxiliary component:** The LLM is an optional explanation layer that verbalizes pre-computed outcomes. It does not perform calculations or make decisions.
- **Measurable evaluation:** Baseline (rule-based report) and LLM report can be compared on clarity, correctness, usefulness, and hallucination rate. Evaluation is reproducible.
- **Reproducibility:** Each run produces saved KPI tables and decision logs. The system can be re-run from the same data with the same results.
- **No black-box decision making:** Every decision is traceable to a rule and a threshold. There is no opaque model deciding “good” or “bad.”

## Architecture: Deterministic Core

**The system's core logic is deterministic.** All decisions (positive / negative / attention) are made by the rule-based engine before any LLM involvement.

```
Step 1: Deterministic KPI Computation (No AI)
        ↓
Step 2: Rule-Based Decision Engine (No AI)
        ↓
Step 3: Rule-Based Baseline Report (No AI)
        ↓
Step 4: LLM-Based Explanation Layer (Optional)
```

- **Step 1** — `kpi.py`: Computes KPIs from dataset columns (time, sales, product, region).
- **Step 2** — `decision_engine.py`: Applies rules, outputs decision labels + explanations.
- **Step 3** — `rule_based_report.py`: Renders baseline report from decision output only.
- **Step 4** — `llm_client.py` + `prompts.py`: LLM explains rule outcomes in natural language. **Optional**; the system functions fully without it.

## KPI Justification

All KPIs are derived **only** from available dataset fields:

| Field   | Dataset Column(s) | KPI Use                                    |
|---------|-------------------|--------------------------------------------|
| Time    | Order_Date        | Date range, monthly trend, anomalies       |
| Sales   | Sales             | Total sales, trends, top lists, shares     |
| Product | Product_Name, Category | Top products, top categories      |
| Region  | Region            | Regional distribution, concentration risk  |

No industry benchmarks or external business standards are used. Use `kpi.get_kpi_definitions()` for full traceability.

## Rule-Based Decision Logic

Rules are **data-driven and dataset-relative**:

| Rule            | Threshold      | Label      | Why (heuristic, not industry benchmark)      |
|-----------------|----------------|------------|----------------------------------------------|
| MoM trend       | ≥ +10%         | positive   | Meaningful upward change                     |
| MoM trend       | ≤ -10%         | negative   | Meaningful downward change                   |
| MoM trend       | otherwise      | stable     | Within normal fluctuation                    |
| Anomaly         | \|z\| > 1.5    | attention  | Dataset-relative outlier (z-score)           |
| Concentration   | Top category share ≥ 30% | attention | Over-reliance on single category   |
| Concentration   | Top region share ≥ 35%  | attention | Over-reliance on single region    |

Thresholds are heuristics for thesis clarity. They are not industry benchmarks.

## LLM Role Restriction

The LLM receives **only** the output of the rule-based decision engine (labels + rule explanations). It does **not**:

- Perform calculations
- Make decisions (positive / negative / attention)
- Interpret raw KPIs

The LLM is an **explanation layer only**.

## Setup

```powershell
cd <project_dir>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ollama

1. Install: https://ollama.ai
2. `ollama pull llama3`
3. Ensure Ollama is running

## Run

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Or double-click `run.bat`

## Dataset

Superstore Sales CSV. Default: `data/train.csv`

Required columns: Order_Date, Sales, Product_Name, Category, Region.

## Outputs

Per run, the following files are always written (when a report is generated):

- `outputs/kpi_tables/kpi_table_<run_id>.csv`
- `outputs/decisions/decision_<run_id>.json`
- `outputs/template_reports/rule_based_report_<run_id>.md`
- `outputs/llm_reports/llm_report_<run_id>.md` (or placeholder if LLM is disabled)

Additional outputs: `outputs/definitions/` (kpi_definitions.json, rule_definitions.json), `outputs/prompts/`, `outputs/evaluation_results.csv`, `outputs/hallucination_flags.json`.

**Each run is fully reproducible from saved KPI tables and decision logs.**

---

## KPI Selection Criteria

All KPIs are derived **only** from dataset columns (`Order_Date`, `Sales`, `Product_Name`, `Category`, `Region`). See `kpi_definitions.py` for full traceability:

| KPI | Source Columns | Formula | Business Question |
|-----|----------------|---------|-------------------|
| total_sales | Sales | sum(Sales) | What is overall sales performance? |
| date_range | Order_Date | min, max | Over what time span does the data cover? |
| monthly_sales_trend | Order_Date, Sales | groupby(month).sum | How do sales evolve over time? |
| top_categories_by_sales | Category, Sales | groupby(Category).sum.head(5) | Which categories drive revenue? |
| top_products_by_sales | Product_Name, Sales | groupby(Product_Name).sum.head(5) | Which products drive revenue? |
| sales_by_region + share | Region, Sales | groupby(Region).sum | How is sales distributed geographically? |
| month_over_month_change | Order_Date, Sales | (latest - prev) / prev * 100 | Is sales trending up or down? |
| anomaly_detection | Order_Date, Sales | z-score \|z\| > 1.5 | Which months deviate significantly? |
| peak_month / lowest_month | Order_Date, Sales | argmax, argmin | When did sales peak/bottom? |

---

## Rule Thresholds (Heuristics)

**Thresholds are heuristics for consistent baseline comparison, NOT industry benchmarks.** All values are dataset-relative. See `decision_engine.RULE_DEFINITIONS` for full traceability:

| Rule | Condition | Threshold | Why (heuristic) |
|------|-----------|-----------|-----------------|
| mom_positive | MoM change >= +10% | +10% | Meaningful upward change; avoids tiny fluctuations |
| mom_negative | MoM change <= -10% | -10% | Meaningful downward change |
| mom_stable | Otherwise | ±10% band | Within normal fluctuation |
| anomaly_z | \|z\| > 1.5 | 1.5 | Moderate statistical outlier; ~87% within ±1.5 |
| region_concentration | Top region share >= 30% | 30% | Over-reliance on single region |
| category_key_driver | Top category share >= 35% | 35% | Dominant category; key driver note |
