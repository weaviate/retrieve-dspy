#!/bin/bash
echo "Switching to local dspy development..."
uv remove dspy-ai 2>/dev/null || true
uv add --editable ../dspy
echo "Using local dspy from ../dspy"