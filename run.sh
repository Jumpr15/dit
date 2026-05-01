#!/bin/bash
set -e
uv sync
uv run python src/train.py
# pip install transformers??
