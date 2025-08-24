import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pickle
from tqdm import tqdm

# ------------------------
# 1. Load dataset
# ------------------------
from datasets import load_dataset
dataset = load_dataset("tweet_eval", "sentiment")

# Merge train and test for simplicity (optional)
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

# ------------------------
# 2. Clean text
# ------------------------
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
    text = text.strip()
    return text

tqdm.pandas()
df_train["text"] = df_train["text"].progress_apply(clean_text)
df_test["text"] = df_test["text"].progress_apply(clean_text)

# ------------------------
# 3. Vectorize text
# ------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(df_train["text"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["text"])
y_test = df_test["label"]

# ------------------------
# 4. Train Logistic Regression
# ------------------------
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)

# ------------------------
# 5. Evaluate
# ------------------------
y_pred = logistic_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
print("Macro F1:", f1)
print(classification_report(y_test, y_pred))

# ------------------------
# 6. Save model and vectorizer
# ------------------------
import os
os.makedirs("../models", exist_ok=True)

with open("../models/model.pkl", "wb") as f:
    pickle.dump(logistic_model, f)

with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved in '../models/' folder.")
