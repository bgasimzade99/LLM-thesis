# LLM-based DSS for Business Reporting

Bachelor thesis prototype. Compares LLM-generated reports with template-based reports.

## Dataset

Superstore Sales CSV: Order Date, Category, Sub-Category, Product Name, Region, Sales.
Default: `data/train.csv`.

## Setup

```powershell
cd <project_dir>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Run

```powershell
python run_app.py
```

Or: `streamlit run app.py`

## Outputs

- `outputs/llm_reports/`
- `outputs/template_reports/`
- `outputs/prompts/`
- `outputs/evaluation_results.csv`
- `outputs/hallucination_flags.json`
