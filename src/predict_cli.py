import argparse, os, sys, joblib, pandas as pd
from preprocess import clean_text

ARTIFACT_DIR = "artifacts"

def load_artifacts():
    vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "vectorizer.joblib"))
    model = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    return vectorizer, model, label_encoder

def predict_texts(texts):
    vectorizer, model, le = load_artifacts()
    clean = [clean_text(t) for t in texts]
    X = vectorizer.transform(clean)
    preds = model.predict(X)
    labels = le.inverse_transform(preds)
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="*", help="One or more texts to classify.")
    parser.add_argument("--csv", type=str, help="CSV file with a column of texts.")
    parser.add_argument("--text-column", type=str, default="text")

    args = parser.parse_args()

    inputs = []
    if args.text:
        inputs.extend(args.text)
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.text_column not in df.columns:
            print(f"Column '{args.text_column}' not in CSV.")
            sys.exit(1)
        inputs.extend(df[args.text_column].astype(str).tolist())

    if not inputs:
        print("No inputs provided. Use --text or --csv.")
        sys.exit(1)

    preds = predict_texts(inputs)
    for t, p in zip(inputs, preds):
        print(f"[{p}] {t}")

if __name__ == "__main__":
    main()
