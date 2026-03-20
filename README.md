# Detecting AI-Generated Source Code: Extended Study

An extension of the ICSE 2025 paper **"An Empirical Study on Automatically Detecting AI-Generated Source Code: How Far Are We?"** ([arXiv:2411.04299](https://arxiv.org/abs/2411.04299)).

This project extends the original study by evaluating **three new state-of-the-art LLMs** not tested in the original paper:
- **Claude Haiku** (Anthropic) — first time any Claude model has been evaluated for AI code detection
- **GPT-4o** (OpenAI) — successor to GPT-4 which was tested in the original paper
- **Gemini 2.5 Flash** (Google) — successor to Gemini Pro which was tested in the original paper

---

## 🔬 Key Finding

Newer LLMs are **easier to detect**, not harder. Our classifier achieves F1 scores of 0.97–0.99 on the new models, compared to the original paper's best of 0.8255 on ChatGPT. This suggests newer LLMs write more structurally consistent code with recognizable patterns, making them more distinguishable from human-written code.

| Model | Embedding | F1 | Human F1 | AI F1 |
|---|---|---|---|---|
| ChatGPT *(original paper)* | Combined | 0.8255 | 0.8369 | 0.8141 |
| **Claude Haiku** *(ours)* | Code | 0.9695 | 0.9681 | 0.9710 |
| **Claude Haiku** *(ours)* | AST | 0.9817 | 0.9818 | 0.9816 |
| **Claude Haiku** *(ours)* | Combined | 0.9756 | 0.9745 | 0.9768 |
| **GPT-4o** *(ours)* | Code | 0.9787 | 0.9781 | 0.9793 |
| **GPT-4o** *(ours)* | AST | 0.9635 | 0.9629 | 0.9641 |
| **GPT-4o** *(ours)* | Combined | 0.9879 | 0.9874 | 0.9883 |
| **Gemini 2.5** *(ours)* | Code | 0.9847 | 0.9848 | 0.9847 |
| **Gemini 2.5** *(ours)* | AST | 0.9818 | 0.9820 | 0.9815 |
| **Gemini 2.5** *(ours)* | Combined | **0.9939** | **0.9939** | **0.9939** |

---

## 📁 Repository Structure

```
detectingAIGeneratedCode/
│
├── generateCode.py          # Step 1: Query LLMs for 164 solutions each
├── prepareDataset.py        # Step 2: Merge AI + human code into labeled CSVs
├── generateAST.py           # Step 3: Generate AST sequences via tree-sitter
├── generateEmbeddings.py    # Step 4: Generate CodeBERT embeddings
├── runClassifier.py         # Step 5: Run classifier, print results table
├── buildGrammar.py          # One-time: compile tree-sitter grammars
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── raw/                 # Step 1 output — raw LLM-generated solutions
│   ├── merged/              # Step 2 output — labeled AI+human CSVs
│   ├── ast_processed/       # Step 3 output — CSVs with AST columns
│   ├── embeddings/          # Step 4 output — CodeBERT embedding vectors
│   └── splits/              # Step 4 output — K-fold train/test splits
│
└── results/
    └── final_results.csv    # Step 5 output — your results table
```

---

## 🚀 Running Locally

### Prerequisites

- Python 3.11 or 3.12 recommended (3.13 works but has some package limitations)
- Git
- ~3GB disk space (for models and data)
- API keys for Anthropic, OpenAI, and Google

### 1. Clone this repo

```bash
git clone https://github.com/akuikel/detectingAIGeneratedCode.git
cd detectingAIGeneratedCode
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install google-genai
```

### 3. Clone tree-sitter grammars (compatible versions)

```bash
git clone --depth 1 --branch v0.20.4 https://github.com/tree-sitter/tree-sitter-python
git clone --depth 1 --branch v0.20.1 https://github.com/tree-sitter/tree-sitter-java
git clone --depth 1 --branch v0.20.3 https://github.com/tree-sitter/tree-sitter-cpp
```

### 4. Build tree-sitter grammar (one-time)

```bash
python buildGrammar.py
```

Expected output:
```
Build complete: build/my-languages.so
```

### 5. Get API keys

| Provider | URL | Cost for this project |
|---|---|---|
| Anthropic | console.anthropic.com | ~$0.50 |
| OpenAI | platform.openai.com/api-keys | ~$1.00 |
| Google | aistudio.google.com/app/apikey | Free (enable billing for quota) |

Open `generateCode.py` and fill in your keys at the top:
```python
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY    = "sk-proj-..."
GOOGLE_API_KEY    = "AIza..."
```

### 6. Run the pipeline

```bash
# Generate AI code (takes ~15 min, costs ~$2 total)
python generateCode.py

# Merge with human-written solutions
python prepareDataset.py

# Generate AST sequences
python generateAST.py

# Generate CodeBERT embeddings (downloads ~500MB model on first run, takes ~5 min)
python generateEmbeddings.py

# Run classifier and see results
python runClassifier.py
```

### 7. Just see the results (if data already exists)

If you already have the embeddings and splits generated:

```bash
python runClassifier.py
```

---

## 🔗 Based On

- **Original Paper:** Suh et al., "An Empirical Study on Automatically Detecting AI-Generated Source Code: How Far Are We?", ICSE 2025. [arXiv:2411.04299](https://arxiv.org/abs/2411.04299)
- **Original Repo:** [JaeWorld/ICSE2025-AI-Detector](https://github.com/JaeWorld/ICSE2025-AI-Detector)
- **Dataset:** [HumanEval-X](https://github.com/THUDM/CodeGeeX) — 164 Python programming problems
- **Embedding Model:** [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)

---

## 📝 Methodology

1. **Dataset:** All 164 Python problems from HumanEval-X benchmark
2. **Code Generation:** Each LLM was prompted with the function signature and docstring and asked to return only the function body
3. **Feature Extraction:** Code and AST sequences were embedded using CodeBERT (768-dim CLS token embeddings). Three embedding types were tested: code-only, AST-only, and combined
4. **Classification:** Logistic regression with 5-fold stratified cross-validation
5. **Evaluation:** F1 score (macro average of human-written F1 and AI-generated F1)

**Note:** The original paper used CodeT5+ 110M for embeddings. This extension uses CodeBERT, which is the same model used in GPTSniffer (the paper's state-of-the-art baseline), making our results directly comparable.

---

## 👤 Author

Aavash Kuikel— Fisk University, Computer Science
