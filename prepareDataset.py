"""
STEP 2: Prepare Merged Datasets
================================
Combines AI-generated code (from Step 1) with human-written
HumanEval-X solutions into the exact CSV format the pipeline expects.

Label convention (matches original repo):
  0 = AI-generated
  1 = Human-written

Output: 3 CSV files in data/merged/
  - humaneval_claude_python_merged.csv
  - humaneval_gpt4o_python_merged.csv
  - humaneval_gemini15_python_merged.csv

Each CSV has columns: [code, actual label]
"""

import os
import gzip
import json
import requests
import pandas as pd

os.makedirs("data/merged", exist_ok=True)

# ─── Load human solutions from HumanEval-X directly from GitHub ──────────────
print("Loading HumanEval-X human solutions...")
url = "https://github.com/THUDM/CodeGeeX/raw/main/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz"
response = requests.get(url, timeout=60)
response.raise_for_status()

problems = []
with gzip.open(__import__('io').BytesIO(response.content)) as f:
    for line in f:
        problems.append(json.loads(line.decode('utf-8')))

human_df = pd.DataFrame({
    "task_id": [p["task_id"] for p in problems],
    "code":    [p["canonical_solution"] for p in problems],
})
print(f"Loaded {len(human_df)} human solutions\n")


# ─── Process each model ───────────────────────────────────────────────────────
models = ["claude", "gpt4o", "gemini15"]

for model_name in models:
    raw_path    = f"data/raw/humaneval_{model_name}_python_raw.csv"
    output_path = f"data/merged/humaneval_{model_name}_python_merged.csv"

    if not os.path.exists(raw_path):
        print(f"[SKIP] {raw_path} not found — run generateCode.py first")
        continue

    ai_df = pd.read_csv(raw_path)
    print(f"Processing {model_name}: {len(ai_df)} AI solutions")

    # ── Build AI half (label = 0) ────────────────────────────────────────────
    ai_half = pd.DataFrame({
        "code":         ai_df["ai_code"].tolist(),
        "actual label": [0] * len(ai_df)
    })

    # ── Build human half (label = 1) ─────────────────────────────────────────
    matched_human = human_df[
        human_df["task_id"].isin(ai_df["task_id"])
    ].copy()

    human_half = pd.DataFrame({
        "code":         matched_human["code"].tolist(),
        "actual label": [1] * len(matched_human)
    })

    # ── Merge and shuffle ────────────────────────────────────────────────────
    merged = pd.concat([ai_half, human_half], ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Sanity check ─────────────────────────────────────────────────────────
    ai_count    = (merged["actual label"] == 0).sum()
    human_count = (merged["actual label"] == 1).sum()
    print(f"  AI samples:    {ai_count}")
    print(f"  Human samples: {human_count}")
    print(f"  Total:         {len(merged)}")

    before = len(merged)
    merged.dropna(subset=["code"], inplace=True)
    if len(merged) < before:
        print(f"  [WARN] Dropped {before - len(merged)} null rows")

    merged.to_csv(output_path, index=False)
    print(f"  [SAVED] {output_path}\n")

print("Step 2 complete. Check data/merged/ for output files.")