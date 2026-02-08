# KCC 2025 Data Analysis

This project fetches 2025 Kisan Call Centre (KCC) transcripts from data.gov.in for local analysis.

## What's Included
- Config: `config/kcc.env`
- Downloader: `scripts/download_kcc_2025.py`
- Data output: `data/kcc_2025.csv`
- Notes: `docs/DATA.md`

## Quick Start
```bash
cd /home/sneharup/KCC/apt
python3 scripts/download_kcc_data.py
```

The downloader pulls records in pages and appends them into a single CSV file. It is safe to rerun; it will resume from the last completed offset if the output file already exists.

## Parameters
Edit `config/kcc.env` to change API key, year, or page size.

## Streamlit App
```bash
cd /home/sneharup/KCC/apt
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
# kcc-transcript
