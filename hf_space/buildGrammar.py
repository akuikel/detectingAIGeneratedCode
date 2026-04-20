"""
Compiles tree-sitter Python grammar into build/my-languages.so.
Run after cloning tree-sitter-python in the same directory.
"""
from tree_sitter import Language
import os

os.makedirs("build", exist_ok=True)
print("Building tree-sitter Python grammar...")
Language.build_library("build/my-languages.so", ["tree-sitter-python"])
print("Done: build/my-languages.so")
