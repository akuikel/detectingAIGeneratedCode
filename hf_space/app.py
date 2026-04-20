"""
AI Code Detector — FastAPI app for HuggingFace Spaces (port 7860)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

import numpy as np
import joblib
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

EMB_DIM = 768
state = {}

HTML_FILE = os.path.join(APP_DIR, "templates", "index.html")
with open(HTML_FILE, encoding="utf-8") as f:
    INDEX_HTML = f.read()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load classifier + scaler
    model_dir = os.path.join(APP_DIR, "models")
    print("Loading classifier and scaler...")
    state["clf"]    = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    state["scaler"] = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    # Load CodeBERT
    codebert = "microsoft/codebert-base"
    print(f"Loading {codebert}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained(codebert)
    bert_model = AutoModel.from_pretrained(codebert).to(device)
    bert_model.eval()
    state["tokenizer"]  = tokenizer
    state["bert_model"] = bert_model
    state["device"]     = device
    print(f"CodeBERT ready on {device}")

    # Load tree-sitter
    state["ast_available"] = False
    try:
        from tree_sitter import Language, Parser
        from tree_sitter_ast_python import F as ast_tokens_fn, rename_variables as rv_fn
        PY_LANGUAGE = Language(os.path.join(APP_DIR, "build", "my-languages.so"), "python")
        parser = Parser()
        parser.set_language(PY_LANGUAGE)
        state["parser"]        = parser
        state["ast_tokens_fn"] = ast_tokens_fn
        state["rv_fn"]         = rv_fn
        state["ast_available"] = True
        print("tree-sitter AST parser ready")
    except Exception as e:
        print(f"[WARN] tree-sitter unavailable: {e}")

    print("\nCodeSense ready on port 7860\n")
    yield
    state.clear()


app = FastAPI(title="AI Code Detector", lifespan=lifespan)


def get_embedding(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(EMB_DIM)
    try:
        inputs = state["tokenizer"](
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(state["device"])
        with torch.no_grad():
            out = state["bert_model"](**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except Exception:
        return np.zeros(EMB_DIM)


def generate_ast(code: str) -> str:
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
    ast_seq   = generate_ast(processed)

    code_emb = get_embedding(code)
    ast_emb  = get_embedding(ast_seq) if ast_seq else np.zeros(EMB_DIM)
    combined = np.concatenate([code_emb, ast_emb]).reshape(1, -1)

    scaled  = state["scaler"].transform(combined)
    proba   = state["clf"].predict_proba(scaled)[0]
    classes = list(state["clf"].classes_)

    ai_prob    = float(proba[classes.index(0)])
    human_prob = float(proba[classes.index(1)])
    verdict    = "AI-Generated" if ai_prob > human_prob else "Human-Written"

    return {
        "verdict":       verdict,
        "ai_percent":    round(ai_prob * 100, 1),
        "human_percent": round(human_prob * 100, 1),
        "ast_used":      state.get("ast_available", False) and bool(ast_seq),
    }


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
    return JSONResponse(predict_code(code))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
