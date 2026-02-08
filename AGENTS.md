# Codex Agent Instructions

## Goal
Work on KCC 2025 data analysis. Keep all assets under `/home/sneharup/KCC/apt`.

## Data Download
- Use `config/kcc.env` for API settings.
- Use `scripts/download_kcc_2025.py` to (re)download.
- Output file must be `data/kcc_2025.csv`.
- If rate limited, back off and resume later; do not delete partial data.

## Conventions
- Keep CSV intact; do not modify raw data.
- Write derived analysis outputs to `data/derived/`.
- Document new analysis steps in `docs/`.
