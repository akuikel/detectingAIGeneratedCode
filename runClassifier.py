"""
STEP 5: Run Classifier & Report Results
=========================================
Extended version of the original test_embedding.py that:
  1. Handles the 3 new models (claude, gpt4o, gemini15)
  2. Trains its own LogisticRegression (no dependency on tuned_models.pkl)
  3. Produces a clean comparison table vs. paper's results
  4. Saves a results CSV for your paper

Usage:
  python step5_run_classifier.py

Output:
  - Printed results table per model per embedding type
  - results/final_results.csv  (for your paper's table)
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAPER BASELINE RESULTS (from Table in paper)
# Fill these in from the paper's Results.xlsx once you read it
# ══════════════════════════════════════════════════════════════════════════════
PAPER_BASELINES = {
    "chatgpt": {
        "code":     {"F1": 0.8033, "Human_F1": 0.8192, "AI_F1": 0.7874},
        "ast":      {"F1": 0.7521, "Human_F1": 0.7698, "AI_F1": 0.7344},
        "combined": {"F1": 0.8255, "Human_F1": 0.8369, "AI_F1": 0.8141},
    },
    "chatgpt4": {
        "code":     {"F1": 0.0,    "Human_F1": 0.0,    "AI_F1": 0.0},
        "ast":      {"F1": 0.0,    "Human_F1": 0.0,    "AI_F1": 0.0},
        "combined": {"F1": 0.0,    "Human_F1": 0.0,    "AI_F1": 0.0},
    },
    "gemini": {
        "code":     {"F1": 0.0,    "Human_F1": 0.0,    "AI_F1": 0.0},
        "ast":      {"F1": 0.0,    "Human_F1": 0.0,    "AI_F1": 0.0},
        "combined": {"F1": 0.0,    "Human_F1": 0.0,    "AI_F1": 0.0},
    },
}
# NOTE: Update the 0.0 values from Results.xlsx before writing your paper


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred):
    acc      = accuracy_score(y_true, y_pred)
    human_f1 = f1_score(y_true, y_pred, pos_label=1)
    ai_f1    = f1_score(y_true, y_pred, pos_label=0)
    avg_f1   = (human_f1 + ai_f1) / 2

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy":  round(acc,      4),
        "TPR":       round(tpr,      4),
        "TNR":       round(tnr,      4),
        "Human_F1":  round(human_f1, 4),
        "AI_F1":     round(ai_f1,    4),
        "F1":        round(avg_f1,   4),
    }


def run_kfold_classifier(df: pd.DataFrame, emb_type: str, n_splits: int = 5):
    """
    Run stratified K-fold cross-validation for one embedding type.
    emb_type: 'code_', 'ast_', or 'combined_'
    """
    X = df.loc[:, df.columns.str.startswith(emb_type)].values
    y = df["actual label"].values

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_metrics.append(calculate_metrics(y_test, y_pred))

    # Average across folds
    avg = {}
    for key in fold_metrics[0]:
        avg[key] = round(np.mean([m[key] for m in fold_metrics]), 4)

    return avg


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    EMB_DIR    = "data/embeddings"
    EMB_TYPES  = ["code_", "ast_", "combined_"]
    NEW_MODELS = ["claude", "gpt4o", "gemini15"]

    all_results = []

    print("=" * 65)
    print("  AI-GENERATED CODE DETECTION — CLASSIFIER RESULTS")
    print("=" * 65)

    for model_name in NEW_MODELS:
        emb_file = os.path.join(EMB_DIR, f"humaneval_{model_name}_python_merged.csv")

        if not os.path.exists(emb_file):
            print(f"\n[SKIP] {emb_file} not found")
            continue

        df = pd.read_csv(emb_file)
        print(f"\n{'─'*65}")
        print(f"  MODEL: {model_name.upper()}  |  {len(df)} samples")
        print(f"{'─'*65}")

        for emb_type in EMB_TYPES:
            emb_label = emb_type.rstrip("_")  # 'code', 'ast', 'combined'

            metrics = run_kfold_classifier(df, emb_type)

            print(f"  [{emb_label:8s}] "
                  f"F1={metrics['F1']:.4f}  "
                  f"Human_F1={metrics['Human_F1']:.4f}  "
                  f"AI_F1={metrics['AI_F1']:.4f}  "
                  f"Acc={metrics['Accuracy']:.4f}")

            all_results.append({
                "Model":     model_name,
                "Emb_Type":  emb_label,
                **metrics
            })

    # ── Save results CSV ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/final_results.csv", index=False)
    print(f"\n[SAVED] results/final_results.csv")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n")
    print("=" * 65)
    print("  COMPARISON: NEW MODELS vs. PAPER BASELINES (combined emb)")
    print("=" * 65)
    print(f"  {'Model':<15} {'F1':>8} {'Human_F1':>10} {'AI_F1':>8}")
    print(f"  {'─'*45}")

    # Paper baselines
    for model, metrics in PAPER_BASELINES.items():
        m = metrics["combined"]
        print(f"  {model:<15} {m['F1']:>8.4f} {m['Human_F1']:>10.4f} {m['AI_F1']:>8.4f}  (paper)")

    print(f"  {'─'*45}")

    # New results
    new_combined = results_df[results_df["Emb_Type"] == "combined"]
    for _, row in new_combined.iterrows():
        print(f"  {row['Model']:<15} {row['F1']:>8.4f} {row['Human_F1']:>10.4f} {row['AI_F1']:>8.4f}  ← NEW")

    print()
    print("Step 5 complete. Results saved to results/final_results.csv")
    print("Use this table directly in your paper.")