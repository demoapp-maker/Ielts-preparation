#!/usr/bin/env bash
# Mengunduh model Vosk (~40 MB zip) ke ./models dan model spaCy en_core_web_sm.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=vosk-model-small-en-us-0.15
mkdir -p models
if [ ! -d "models/$MODEL" ]; then
  echo ">> Mengunduh $MODEL ..."
  curl -L -o "models/$MODEL.zip" "https://alphacephei.com/vosk/models/$MODEL.zip"
  unzip -q "models/$MODEL.zip" -d models && rm "models/$MODEL.zip"
fi
echo ">> Mengunduh model spaCy en_core_web_sm ..."
python -m spacy download en_core_web_sm
echo ">> Selesai. Jalankan: uvicorn app:app --host 0.0.0.0 --port 8000"
