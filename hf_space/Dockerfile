FROM python:3.11

RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# System deps for compiling tree-sitter grammar
RUN apt-get update && apt-get install -y git gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all app files
COPY --chown=user . /app

# Build tree-sitter Python grammar
RUN git clone --depth 1 https://github.com/tree-sitter/tree-sitter-python \
    && python buildGrammar.py \
    && rm -rf tree-sitter-python

# Pre-download CodeBERT so first request is fast
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('microsoft/codebert-base'); \
AutoModel.from_pretrained('microsoft/codebert-base'); \
print('CodeBERT cached')"

USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
