# LLM-based Decision Support System (DSS) for Business Reporting

Academic prototype for comparing LLM-generated business reports with template-based reports. Bachelor thesis level.

## Dataset

Superstore Sales (CSV) with columns: **Order Date**, **Category**, **Sub-Category**, **Product Name**, **Region**, **Sales**.

Example sources:
- [Tableau Superstore Sample Data](https://www.tableau.com/sample-data)
- [Kaggle Superstore datasets](https://www.kaggle.com/datasets)

## Setup

**PowerShell veya CMD’de (Cursor dışında):**

```powershell
cd "c:\Users\gasim\OneDrive\Desktop\LLM\LLM-thesis"

# Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\Activate.ps1

# Paketleri kur
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

## Run

Sanal ortam aktifken:

```powershell
streamlit run app.py
```

`streamlit` bulunamazsa:

```powershell
python -m streamlit run app.py
```

## Usage

1. Upload a Superstore Sales CSV file. A minimal `sample_superstore.csv` is included for testing.
2. Review the dataset preview.
3. Click **Generate Business Report**.
4. Compare the LLM report and template report side by side.
5. Complete the evaluation rubric (Clarity, Correctness, Usefulness, Consistency).
6. Optionally save reports to the `outputs/` directory.

## Architecture

- `app.py` — Streamlit UI
- `kpi.py` — KPI computation (no raw data sent to LLM)
- `prompts.py` — KPI → structured text for LLM
- `llm_client.py` — OpenAI API client (gpt-4o-mini, max 400 tokens, temp 0.2)
- `template_report.py` — Rule-based report generator
- `evaluation.py` — Rubric scoring and hallucination detection

## Outputs

Reports and evaluation scores are saved under `outputs/` with timestamped filenames for reproducibility.
