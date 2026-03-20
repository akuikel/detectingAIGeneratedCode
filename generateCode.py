"""
STEP 1: Generate AI Code from New LLMs
=======================================
Queries Claude Haiku, GPT-4o, and Gemini 2.0 Flash
for all 164 HumanEval-X Python problems.

Output: 3 CSV files in data/raw/
  - humaneval_claude_python_raw.csv
  - humaneval_gpt4o_python_raw.csv
  - humaneval_gemini15_python_raw.csv
"""

import os
import time
import gzip
import json
import requests
import pandas as pd
from tqdm import tqdm

# ─── API Keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
OPENAI_API_KEY    = "YOUR_OPENAI_API_KEY"
GOOGLE_API_KEY    = "YOUR_GOOGLE_API_KEY"

# ─── Output folder ────────────────────────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)

# ─── Load HumanEval-X Python problems directly from GitHub ───────────────────
def load_humaneval_x_python():
    url = "https://github.com/THUDM/CodeGeeX/raw/main/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz"
    print("Downloading HumanEval-X Python dataset from GitHub...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    problems = []
    with gzip.open(__import__('io').BytesIO(response.content)) as f:
        for line in f:
            problems.append(json.loads(line.decode('utf-8')))
    print(f"Loaded {len(problems)} problems\n")
    return problems


problems = load_humaneval_x_python()


# ─── Prompt builder ───────────────────────────────────────────────────────────
def build_prompt(problem):
    return (
        "Complete the following Python function. "
        "Return ONLY the complete function code with no explanation, "
        "no markdown, no extra text.\n\n"
        f"{problem['prompt']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_claude(problems):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    print("=== Generating with Claude Haiku ===")
    for problem in tqdm(problems):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": build_prompt(problem)}]
            )
            ai_code = message.content[0].text.strip()
        except Exception as e:
            print(f"\n[ERROR] Task {problem['task_id']}: {e}")
            ai_code = None
        results.append({"task_id": problem["task_id"], "prompt": problem["prompt"], "ai_code": ai_code})
        time.sleep(0.3)
    return pd.DataFrame(results)


def generate_gpt4o(problems):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []
    print("=== Generating with GPT-4o ===")
    for problem in tqdm(problems):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a Python expert. Return ONLY the complete function code. No explanation, no markdown, no extra text."},
                    {"role": "user", "content": build_prompt(problem)}
                ]
            )
            ai_code = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n[ERROR] Task {problem['task_id']}: {e}")
            ai_code = None
        results.append({"task_id": problem["task_id"], "prompt": problem["prompt"], "ai_code": ai_code})
        time.sleep(0.3)
    return pd.DataFrame(results)


def generate_gemini(problems):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    results = []
    print("=== Generating with Gemini 2.0 Flash ===")
    for problem in tqdm(problems):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=build_prompt(problem),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                )
            )
            ai_code = response.text.strip()
        except Exception as e:
            print(f"\n[ERROR] Task {problem['task_id']}: {e}")
            ai_code = None
        results.append({"task_id": problem["task_id"], "prompt": problem["prompt"], "ai_code": ai_code})
        time.sleep(1.0)
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    generators = {
        "claude":   generate_claude,
        "gpt4o":    generate_gpt4o,
        "gemini15": generate_gemini,
    }

    for model_name, generator_fn in generators.items():
        output_path = f"data/raw/humaneval_{model_name}_python_raw.csv"

        if os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            print(f"[SKIP] {output_path} already exists ({len(existing)} rows)")
            continue

        df = generator_fn(problems)

        before = len(df)
        df.dropna(subset=["ai_code"], inplace=True)
        after = len(df)
        if before != after:
            print(f"[WARN] {before - after} failed generations dropped for {model_name}")

        df.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path} — {len(df)} solutions\n")

    print("Step 1 complete. Check data/raw/ for output files.")