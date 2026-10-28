# City Hotel (H2) Booking — ML Project

CSE382 Introduction to Machine Learning — Fall 2026

## Setup

1. Clone the repo and enter the folder.
2. Download `rawdata.csv` and place it in `data/raw/`.
3. Create a virtual environment and install requirements:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Usage

```
python main.py check-data
python main.py analyze
python main.py preprocess
python main.py train --task regression
python main.py train --task classification
python main.py cluster
python main.py evaluate
python main.py predict --task regression --input booking.json
streamlit run app/app.py
```

## Project Structure

- `src/data/` — loading, cleaning, EDA
- `src/preprocessing/` — pipelines and train/test split
- `src/models/` — regression, classification, clustering models
- `src/evaluation/` — metrics, tuning, comparison tables
- `src/dimred/` — PCA and LDA
- `src/inference/` — prediction functions used by CLI and app
- `app/` — Streamlit application
- `reports/` — milestone report, leakage audit, final report

## Submission

- GitHub: <link>
- Final commit SHA: <sha>
