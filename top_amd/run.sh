#!/bin/bash
# Qwen3-4B AMD 一键训练
cd "$(dirname "$0")/.."
python top_amd/run.py "$@"
