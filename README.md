# Social Media Sentiment Analysis (Yhills Internship)

End‑to‑end project to classify social media posts (negative / neutral / positive) and
monitor live sentiment for a keyword/brand via a Streamlit dashboard.

## 1) Environment (VS Code)
```bash
# 1) Create & activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Upgrade pip
python -m pip install --upgrade pip
```

> **If you plan to train BERT** you need `torch`. Installing CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 3) Train a baseline (TF‑IDF + Logistic Regression)
This uses the 3‑class **TweetEval** sentiment dataset via 🤗 Datasets.
```bash
python src/train_baseline.py
```
Artifacts are saved to `artifacts/`:
- `vectorizer.joblib`, `model.joblib`
- `label_encoder.joblib`
- `confusion_matrix.png`
- `metrics.json`

## 4) Evaluate (prints metrics & shows confusion matrix file path)
```bash
python src/evaluate_baseline.py
```

## 5) Predict on custom text (CLI)
```bash
python src/predict_cli.py --text "I love this phone" --text "Worst service ever!"
```

Or on a CSV with a column `text`:
```bash
python src/predict_cli.py --csv data/sample_posts.csv --text-column text
```

## 6) Live dashboard (Streamlit)
Monitors recent posts from X (formerly Twitter) using **snscrape** (no API key).
```bash
streamlit run app/streamlit_app.py
```
- Choose **Live Mode**, enter a keyword (e.g., `deloitte`), set a limit (e.g., 200),
  then click **Fetch & Analyze**.
- Or use **Manual Text** mode for one‑off texts.

> **Note:** Social platform policies change. `snscrape` usually works without login,
but if it breaks, switch to dataset mode or provide your own API.

## 7) Project structure
```
sentiment_analysis_yhills/
├─ app/
│  └─ streamlit_app.py
├─ artifacts/                  # Saved models/plots/metrics
├─ data/
│  ├─ sample_posts.csv
│  └─ .gitkeep
├─ src/
│  ├─ __init__.py
│  ├─ preprocess.py
│  ├─ train_baseline.py
│  ├─ evaluate_baseline.py
│  ├─ predict_cli.py
│  └─ utils.py
├─ requirements.txt
├─ README.md
└─ .env.example
```

## 8) Optional: Fine‑tune BERT
If you want a transformer model (better accuracy, needs GPU/longer time),
extend from this baseline or create a new script using `transformers` Trainer
on `tweet_eval`.

## 9) Tips
- Class imbalance: check label counts and use `class_weight='balanced'` (already set).
- Error analysis: inspect wrong predictions saved by `evaluate_baseline.py`.
- Reproducibility: seeds are fixed in `utils.py`.
- Ethics: Don’t collect or store personal/sensitive data. Respect site T&Cs.

Good luck! ✨
