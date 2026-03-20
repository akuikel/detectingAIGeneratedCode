"""
build_grammar.py — One-time setup
===================================
Compiles the tree-sitter grammars for Python, Java, and C++
into build/my-languages.so which ast-generator.py requires.

Run this ONCE from the repo root before running step3.

Usage:
  python build_grammar.py

Prerequisites:
  pip install tree-sitter==0.20.4

  git clone https://github.com/tree-sitter/tree-sitter-python
  git clone https://github.com/tree-sitter/tree-sitter-java
  git clone https://github.com/tree-sitter/tree-sitter-cpp
"""

from tree_sitter import Language
import os

os.makedirs("build", exist_ok=True)

print("Building tree-sitter grammars...")
print("This requires tree-sitter-python, tree-sitter-java, tree-sitter-cpp")
print("cloned in the current directory.\n")

Language.build_library(
    "build/my-languages.so",
    [
        "tree-sitter-python",
        "tree-sitter-java",
        "tree-sitter-cpp",
    ]
)

print("Build complete: build/my-languages.so")
print("You can now run step3_generate_ast.py")