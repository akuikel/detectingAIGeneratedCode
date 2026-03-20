"""
STEP 3: Generate AST Sequences
================================
Processes merged CSVs to add AST columns using the repo's
existing tree-sitter parser files.

Output: data/ast_processed/*.csv
"""

import os
import shutil
import sys
import pandas as pd
from glob import glob

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.abspath(__file__))
ANALYZER_DIR   = os.path.join(REPO_ROOT, "src", "code-analyzer-tree-sitter")
MERGED_DIR     = os.path.join(REPO_ROOT, "data", "merged")
AST_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "ast_processed")

os.makedirs(AST_OUTPUT_DIR, exist_ok=True)

# ─── Add the analyzer directory to Python path ───────────────────────────────
sys.path.insert(0, ANALYZER_DIR)

# ─── Change working directory so build/my-languages.so is found ──────────────
os.chdir(REPO_ROOT)

# ─── Now import tree-sitter modules from the repo ────────────────────────────
print("Loading tree-sitter parsers...")
try:
    from tree_sitter import Language, Parser
    from tree_sitter_ast_python import F as F_python, remove_comments as R_python, replace_function_names as RF_python, rename_variables as RV_python

    PY_LANGUAGE = Language('build/my-languages.so', 'python')
    python_parser = Parser()
    python_parser.set_language(PY_LANGUAGE)
    print("Parsers loaded successfully\n")
except Exception as e:
    print(f"[FATAL] Could not load parsers: {e}")
    print("Make sure you ran buildGrammar.py first")
    sys.exit(1)


# ─── AST generation functions ─────────────────────────────────────────────────
def rename_variables(code):
    try:
        tree = python_parser.parse(bytes(code, "utf8"))
        return RV_python(tree, code)
    except:
        return code

def generate_ast(code):
    try:
        tree = python_parser.parse(bytes(str(code), "utf8"))
        ast_tokens = F_python(tree.root_node, bytes(str(code), 'utf8'))
        return ' '.join(ast_tokens)
    except:
        return None


# ─── Process each merged CSV ──────────────────────────────────────────────────
csv_files = glob(os.path.join(MERGED_DIR, "*.csv"))
print(f"Found {len(csv_files)} files to process\n")

for csv_path in csv_files:
    fname = os.path.basename(csv_path)
    output_path = os.path.join(AST_OUTPUT_DIR, fname)

    if os.path.exists(output_path):
        print(f"[SKIP] {fname} already processed")
        continue

    print(f"Processing {fname}...")
    df = pd.read_csv(csv_path)
    df['idx'] = df.index

    original_size = len(df)

    df['new_code'] = df['code'].apply(lambda c: rename_variables(str(c)))
    df['ast']      = df['new_code'].apply(lambda c: generate_ast(str(c)) if c else None)

    df.dropna(subset=['ast'], inplace=True)
    removed = original_size - len(df)
    if removed > 0:
        print(f"  [WARN] {removed} rows dropped (could not parse)")

    output_df = df[['idx', 'code', 'new_code', 'ast', 'actual label']]
    output_df.to_csv(output_path, index=False)
    print(f"  [SAVED] {output_path} — {len(output_df)} rows\n")

print("Step 3 complete. Check data/ast_processed/ for output files.")