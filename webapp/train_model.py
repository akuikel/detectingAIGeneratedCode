"""
Train and save the AI code detection model from existing embeddings.
Run this ONCE before starting the web app:
    cd ICSE2025-AI-Detector
    python webapp/train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR   = os.path.join(REPO_ROOT, "data", "embeddings")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EMB_DIM = 768

print("Loading embedding data from all models...")
dfs = []
for fname in os.listdir(EMB_DIR):
    if fname.endswith(".csv"):
        path = os.path.join(EMB_DIR, fname)
        df   = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {fname}: {len(df)} rows")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal samples: {len(combined_df)}")
print(f"Label distribution:\n{combined_df['actual label'].value_counts()}\n")

# Use combined_ columns (code + AST, 1536-dim)
combined_cols = [c for c in combined_df.columns if c.startswith("combined_")]
X = combined_df[combined_cols].values
y = combined_df["actual label"].values

# Train/test split for reporting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("Training LogisticRegression classifier...")
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average="macro")
print(f"Validation — Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

# Save model and scaler
joblib.dump(clf,    os.path.join(MODEL_DIR, "classifier.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"\nSaved to {MODEL_DIR}/classifier.pkl + scaler.pkl")
print("You can now start the web app:  python webapp/app.py")
