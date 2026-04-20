"""
AI Code Detector — FastAPI Backend
Run from ICSE2025-AI-Detector root:
    python webapp/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "code-analyzer-tree-sitter"))
os.chdir(REPO_ROOT)

import numpy as np
import joblib
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

EMB_DIM = 768
state = {}  # shared app state loaded at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Load saved model ───────────────────────────────────────
    MODEL_DIR   = os.path.join(WEBAPP_DIR, "models")
    clf_path    = os.path.join(MODEL_DIR, "classifier.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(
            "Model not found. Run:  python webapp/train_model.py  first."
        )
    print("Loading classifier and scaler...")
    state["clf"]    = joblib.load(clf_path)
    state["scaler"] = joblib.load(scaler_path)

    # ── Load CodeBERT ──────────────────────────────────────────
    CODEBERT = "microsoft/codebert-base"
    print(f"Loading {CODEBERT}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained(CODEBERT)
    bert_model = AutoModel.from_pretrained(CODEBERT).to(device)
    bert_model.eval()
    state["tokenizer"]  = tokenizer
    state["bert_model"] = bert_model
    state["device"]     = device
    print(f"CodeBERT ready on {device}")

    # ── Load tree-sitter ───────────────────────────────────────
    state["ast_available"] = False
    try:
        from tree_sitter import Language, Parser
        from tree_sitter_ast_python import F as ast_tokens_fn, rename_variables as rv_fn
        PY_LANGUAGE   = Language("build/my-languages.so", "python")
        parser        = Parser()
        parser.set_language(PY_LANGUAGE)
        state["parser"]       = parser
        state["ast_tokens_fn"] = ast_tokens_fn
        state["rv_fn"]        = rv_fn
        state["ast_available"] = True
        print("tree-sitter AST parser ready")
    except Exception as e:
        print(f"[WARN] tree-sitter unavailable ({e}). Will use code-only embeddings.")

    print("\nCodeSense is ready -> http://localhost:8000\n")
    yield
    state.clear()


app = FastAPI(title="AI Code Detector", lifespan=lifespan)

HTML_FILE = os.path.join(WEBAPP_DIR, "templates", "index.html")
with open(HTML_FILE, encoding="utf-8") as f:
    INDEX_HTML = f.read()


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(EMB_DIM)
    try:
        tokenizer  = state["tokenizer"]
        bert_model = state["bert_model"]
        device     = state["device"]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(device)
        with torch.no_grad():
            out = bert_model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except Exception:
        return np.zeros(EMB_DIM)


def generate_ast_sequence(code: str) -> str:
    if not state.get("ast_available"):
        return ""
    try:
        tree   = state["parser"].parse(bytes(code, "utf8"))
        tokens = state["ast_tokens_fn"](tree.root_node, bytes(code, "utf8"))
        return " ".join(tokens)
    except Exception:
        return ""


def rename_vars(code: str) -> str:
    if not state.get("ast_available"):
        return code
    try:
        tree = state["parser"].parse(bytes(code, "utf8"))
        return state["rv_fn"](tree, code)
    except Exception:
        return code


def predict_code(code: str) -> dict:
    processed = rename_vars(code)
    ast_seq   = generate_ast_sequence(processed)

    code_emb = get_embedding(code)
    ast_emb  = get_embedding(ast_seq) if ast_seq else np.zeros(EMB_DIM)
    combined = np.concatenate([code_emb, ast_emb]).reshape(1, -1)

    combined_scaled = state["scaler"].transform(combined)
    proba   = state["clf"].predict_proba(combined_scaled)[0]
    classes = list(state["clf"].classes_)

    ai_idx    = classes.index(0)
    human_idx = classes.index(1)
    ai_prob   = float(proba[ai_idx])
    human_prob = float(proba[human_idx])
    verdict   = "AI-Generated" if ai_prob > human_prob else "Human-Written"

    return {
        "verdict":       verdict,
        "ai_percent":    round(ai_prob * 100, 1),
        "human_percent": round(human_prob * 100, 1),
        "ast_used":      state.get("ast_available", False) and bool(ast_seq),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=INDEX_HTML)


@app.post("/detect")
async def detect(req: CodeRequest):
    code = req.code.strip()
    if not code:
        return JSONResponse({"error": "No code provided"}, status_code=400)
    if len(code) > 20000:
        return JSONResponse({"error": "Code too long (max 20,000 chars)"}, status_code=400)
    result = predict_code(code)
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
