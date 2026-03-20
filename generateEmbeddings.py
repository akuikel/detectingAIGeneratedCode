"""
STEP 4: Generate Code Embeddings
==================================
Uses microsoft/CodeBERT-base to generate embeddings.
CodeBERT works with current transformers/tokenizers versions
and is actually the model used in GPTSniffer (paper baseline).

Input:  data/ast_processed/*.csv
Output: data/embeddings/*.csv  (with code_, ast_, combined_ columns)
        data/splits/           (K-fold train/test splits)

Usage:
  python generateEmbeddings.py
"""

import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

os.makedirs("data/embeddings", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)

# ─── Load CodeBERT ────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/codebert-base"
print(f"Loading {MODEL_NAME}...")
print("(~500MB download on first run)\n")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

EMBEDDING_DIM = 768  # CodeBERT output dimension


# ─── Embedding function ───────────────────────────────────────────────────────
def get_embedding(text: str) -> np.ndarray:
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(EMBEDDING_DIM)
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        return np.zeros(EMBEDDING_DIM)


# ─── Process each AST-processed CSV ──────────────────────────────────────────
AST_DIR   = "data/ast_processed"
SPLIT_DIR = "data/splits"

csv_files = [f for f in os.listdir(AST_DIR) if f.endswith(".csv")]
print(f"Found {len(csv_files)} files to embed: {csv_files}\n")

for csv_file in csv_files:
    # e.g. "humaneval_claude_python_merged.csv" → model_name = "claude"
    parts      = csv_file.split("_")
    model_name = parts[1]   # claude, gpt4o, gemini15

    input_path = os.path.join(AST_DIR, csv_file)
    emb_path   = os.path.join("data/embeddings", csv_file)

    if os.path.exists(emb_path):
        print(f"[SKIP] {emb_path} already exists")
        df_embedded = pd.read_csv(emb_path)
    else:
        df = pd.read_csv(input_path)
        print(f"Generating embeddings for {model_name} ({len(df)} rows)...")

        code_embeddings = []
        ast_embeddings  = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
            code_emb = get_embedding(str(row["code"]))
            ast_emb  = get_embedding(str(row["ast"]))
            code_embeddings.append(code_emb)
            ast_embeddings.append(ast_emb)

        code_arr = np.array(code_embeddings)
        ast_arr  = np.array(ast_embeddings)
        combined = np.concatenate([code_arr, ast_arr], axis=1)

        code_cols     = [f"code_{i}"     for i in range(EMBEDDING_DIM)]
        ast_cols      = [f"ast_{i}"      for i in range(EMBEDDING_DIM)]
        combined_cols = [f"combined_{i}" for i in range(EMBEDDING_DIM * 2)]

        code_df     = pd.DataFrame(code_arr,  columns=code_cols)
        ast_df      = pd.DataFrame(ast_arr,   columns=ast_cols)
        combined_df = pd.DataFrame(combined,  columns=combined_cols)

        df_embedded = pd.concat([
            df[["idx", "code", "ast", "actual label"]].reset_index(drop=True),
            code_df, ast_df, combined_df
        ], axis=1)

        df_embedded.to_csv(emb_path, index=False)
        print(f"  [SAVED] {emb_path}\n")

    # ── Create K-fold splits ──────────────────────────────────────────────────
    print(f"Creating 5-fold splits for {model_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X   = df_embedded.drop(columns=["actual label"])
    y   = df_embedded["actual label"]

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_dir = os.path.join(SPLIT_DIR, f"{model_name}_fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df_embedded.iloc[train_idx].to_csv(
            os.path.join(fold_dir, f"{model_name}_train.csv"), index=False)
        df_embedded.iloc[test_idx].to_csv(
            os.path.join(fold_dir, f"{model_name}_test.csv"),  index=False)

    print(f"  Saved 5-fold splits to {SPLIT_DIR}/{model_name}_fold*/\n")

print("Step 4 complete!")
print(f"Embeddings: data/embeddings/")
print(f"Splits:     data/splits/")
print("\nNext: run python runClassifier.py")