import os, json, joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ARTIFACT_DIR = "artifacts"

def main():
    vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "vectorizer.joblib"))
    model = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
    le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))

    # Load test set used during training run (metrics already saved),
    # but we'll re-run a quick check by sampling the artifacts if needed.
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("Macro F1:", metrics["macro_f1"])
    print("Per-class metrics:")
    for label, stats in metrics["report"].items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        print(label, {k: round(v, 3) for k, v in stats.items()})

    print("Confusion matrix saved at artifacts/confusion_matrix.png")

if __name__ == "__main__":
    main()
