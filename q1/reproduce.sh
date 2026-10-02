#!/usr/bin/env bash
# Run from an isolated Q1 root on Linux, after the documented environment setup.
set -euo pipefail
PY=${Q1_PYTHON:-env/bin/python}
export HF_HOME="$PWD/models"
export HF_HUB_DISABLE_XET=1
mkdir -p logs outputs reports features/text_audio features/acoustic_visual
test -f data/manifest.jsonl
test -f review/manual_review.json
# Run setup_alignment_dependencies.sh and the documented OpenFace setup first.
# These checks only import; they do not install or replace GPU/PyTorch packages.
# Fail before downloading/processing if setup is absent.
PYTHONPATH=audit_deps "$PY" -c 'import faster_whisper, ctranslate2; from importlib.metadata import version; assert version("faster-whisper")=="1.2.1" and version("ctranslate2")=="4.6.0", "Use the recorded audit dependency versions"'
PYTHONPATH=qwen_deps "$PY" -c 'import qwen_asr, transformers; from importlib.metadata import version; assert version("qwen-asr")=="0.0.6" and version("transformers")=="4.57.6", "Use the recorded Qwen dependency versions"'
if [[ -f tools/openface_env.sh ]]; then source tools/openface_env.sh; fi
if [[ -z "${OPENFACE_BIN:-}" || ! -x "$OPENFACE_BIN" ]]; then
  echo 'OpenFace is not ready: run bash setup_openface.sh --build-dlib, then source tools/openface_env.sh (or provide OPENFACE_BIN).' >&2
  exit 2
fi
"$PY" -m feature_extraction.download_models --branches text,audio,video --cache models/hub
"$PY" -m feature_extraction.download_audit_models
"$PY" -m feature_extraction.media --manifest data/manifest.jsonl --data-root data --output outputs
CTC_SNAPSHOT="$PWD/models/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
"$PY" -m feature_extraction.align --manifest data/manifest.jsonl --data-root data --output outputs --device cuda:0 --model-name "$CTC_SNAPSHOT" --model-cache-dir models/hub
"$PY" -m feature_extraction.alignment_refine --manifest data/manifest.jsonl --output outputs --model-cache-dir models/hub --device cuda:0 --model-revision 22aad52d435eb6dbaf354bdad9b0da84ce7d6156
PYTHONPATH=audit_deps "$PY" tools/audit_q1_language.py --output outputs --model models/whisper_tiny --device cpu > logs/language_audit.log
PYTHONPATH=qwen_deps "$PY" -m feature_extraction.align_qwen_check --manifest data/manifest.jsonl --output outputs --model models/qwen_aligner --device cuda:0 --text-mode refined > logs/qwen_audit.log
"$PY" -m feature_extraction.finalize_alignment --manifest data/manifest.jsonl --output outputs --language-audit outputs/language_audit_summary.json --manual-review review/manual_review.json --report-dir reports/finalization
"$PY" -m feature_extraction.promote_alignment --manifest data/manifest.jsonl --output outputs --summary reports/finalization/finalization_summary.json
"$PY" -m feature_extraction.encode --manifest data/manifest.jsonl --data-root data --output outputs --branch text --device cuda:0 --revision 722cf37b1afa9454edce342e7895e588b6ff1d59
"$PY" -m feature_extraction.encode --manifest data/manifest.jsonl --data-root data --output outputs --branch audio --device cuda:0 --revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
"$PY" -m feature_extraction.encode --manifest data/manifest.jsonl --data-root data --output outputs --branch acoustic
"$PY" -m feature_extraction.vision --manifest data/manifest.jsonl --data-root data --output outputs --branch face
"$PY" -m feature_extraction.vision --manifest data/manifest.jsonl --data-root data --output outputs --branch video --device cuda:0 --batch-size 2 --revision dc740ceda42fce44faed2ea03c6d447db72f6af9
"$PY" -m feature_extraction.validate --manifest data/manifest.jsonl --output outputs --report-dir reports/full100
for ROLE in text_audio acoustic_visual; do
  "$PY" tools/export_q1_outputs.py --manifest data/manifest.jsonl --output outputs --role "$ROLE" --validation reports/full100/summary.json --destination "reports/$ROLE.tar.gz"
  tar -xzf "reports/$ROLE.tar.gz" -C "features/$ROLE"
done
"$PY" tools/index_q1_features.py --features features --manifest data/manifest.jsonl
"$PY" tools/read_q1_features.py --features features
# Exact recorded diagnostic used Python 3.11.14 / NumPy 1.26.4 / sklearn 1.6.1;
# See README.md section 3 and environment/diagnostic_requirements.txt in the package root.
# python -m feature_extraction.experiments --index features/samples_index.csv --labels data/labels.csv --manifest data/manifest.jsonl --output experiments --tag reproduced
