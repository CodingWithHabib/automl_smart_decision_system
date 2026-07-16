# AutoML Based Smart Decision Support System

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python -m streamlit run automl_dss\app.py

The app opens at `http://localhost:8501`

## Sample Datasets

Located in `data/sample/`:
- `iris_sample.csv` — Classification (target: `species`)
- `salary_sample.csv` — Regression (target: `salary`)
- For clustering: upload any dataset and select "None (Clustering)" for target

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

See `DESIGN.md` for full architecture, workflow, and design decisions.

## Constraints

- CSV only, structured tabular data
- Max 50,000 rows
- Local execution only (no cloud)
- Python 3.10+
