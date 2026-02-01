# LLM-based DSS for Business Reporting

Bachelor thesis prototype. Compares LLM-generated reports with template-based reports. Uses Ollama for local LLM (no API keys).

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

## Outputs

- `outputs/llm_reports/`
- `outputs/template_reports/`
- `outputs/prompts/`
- `outputs/evaluation_results.csv`
- `outputs/hallucination_flags.json`
