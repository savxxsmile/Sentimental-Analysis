import os
import re
import string
import json
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from datasets import load_dataset

# -----------------------------
# Preprocessing function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# -----------------------------
# Main training function
# -----------------------------
def main():
    print("Loading dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")

    # Convert to DataFrame
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    print("Cleaning text...")
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)

    # -----------------------------
    # Feature extraction
    # -----------------------------
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    y_train = train_df['label']
    y_test = test_df['label']

    # -----------------------------
    # Model training
    # -----------------------------
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=500, multi_class='ovr')
    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("Evaluating...")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Macro F1: {f1}")

    # Convert report keys to strings
    report_str_keys = {str(k): v for k, v in report.items()}

    # -----------------------------
    # Create models folder if not exists
    # -----------------------------
    os.makedirs("../models", exist_ok=True)

    # Save model
    with open("../models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save vectorizer
    with open("../models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save evaluation report
    with open("../models/eval_report.json", "w") as f:
        json.dump({"macro_f1": f1, "report": report_str_keys}, f, indent=2)

    print("Training and saving completed successfully!")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    main()
