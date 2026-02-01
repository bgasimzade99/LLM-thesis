"""Launcher: loads .env then runs Streamlit."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

sys.argv = ["streamlit", "run", "app.py"]
from streamlit.web import cli as stcli
sys.exit(stcli.main())
