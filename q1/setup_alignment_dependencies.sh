#!/usr/bin/env bash
# Recorded Linux audit overlays. Keep the existing project CUDA/PyTorch intact.
# Run from the Q1 root after the base environment has been prepared.
set -euo pipefail
PY=${Q1_PYTHON:-env/bin/python}
"$PY" -c 'import torch, av, soundfile, librosa, einops, pandas, numpy, tokenizers, huggingface_hub; print("Base PyTorch:", torch.__version__)'
mkdir -p qwen_deps audit_deps
"$PY" -m pip install --disable-pip-version-check --no-deps --upgrade --target qwen_deps \
  'qwen-asr==0.0.6' 'transformers==4.57.6' 'accelerate==1.12.0' \
  'nagisa==0.2.11' 'soynlp==0.0.493' 'qwen-omni-utils==0.0.9' \
  'sox==1.5.0' 'Dynet38==2.2' 'six==1.17.0'
"$PY" -m pip install --disable-pip-version-check --no-deps --upgrade --target audit_deps \
  'faster-whisper==1.2.1' 'ctranslate2==4.6.0' 'onnxruntime==1.23.2'
PYTHONPATH=qwen_deps "$PY" -c 'import qwen_asr, transformers; from importlib.metadata import version; assert version("qwen-asr")=="0.0.6" and version("transformers")=="4.57.6"; print("Qwen audit overlay imports OK")'
PYTHONPATH=audit_deps "$PY" -c 'import faster_whisper, ctranslate2, onnxruntime; from importlib.metadata import version; assert version("faster-whisper")=="1.2.1" and version("ctranslate2")=="4.6.0"; print("Language audit overlay imports OK")'
