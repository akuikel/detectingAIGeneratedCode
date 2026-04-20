#!/usr/bin/env bash
# Run from the ICSE2025-AI-Detector root directory:
#   bash webapp/start.sh

set -e
cd "$(dirname "$0")/.."

echo "=== CodeSense — AI Code Detector ==="
echo ""

# Step 1: Install dependencies
echo "[1/3] Installing webapp dependencies..."
pip install -r webapp/requirements.txt -q

# Step 2: Train and save model (skips if already done)
if [ ! -f "webapp/models/classifier.pkl" ]; then
  echo "[2/3] Training model from existing embeddings..."
  python webapp/train_model.py
else
  echo "[2/3] Model already trained — skipping."
fi

# Step 3: Start server
echo "[3/3] Starting server at http://localhost:8000"
echo "      Press Ctrl+C to stop."
echo ""
cd webapp
python app.py
