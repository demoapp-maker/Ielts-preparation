# 🎙️ IELTS Speaking Practice Analyzer

Aplikasi web full-stack untuk latihan IELTS Speaking. Audio direkam di browser,
lalu **semua analisis AI berjalan di server (CPU-only, offline)**:

- **Vosk** (`vosk-model-small-en-us`) — speech-to-text + word timestamps
- **spaCy** (`en_core_web_sm`) — Type-Token Ratio, POS, struktur kalimat
- **LanguageTool** (offline, Java) — deteksi kesalahan grammar

Frontend: HTML5 + Tailwind CSS + Vanilla JS (MediaRecorder API → WAV 16 kHz → `fetch` FormData).

## Fitur
- Rekam langsung dari mikrofon (`Mulai Rekam` → `Berhenti & Kirim`) atau unggah `.wav` / `.mp3`
- **Fluency & Coherence**: WPM, filler words (`uh, um, ah, err, like, you know`), jeda > 1.5 detik
- **Lexical Resource**: TTR, MATTR, transisi akademis (`however, therefore, furthermore`, …)
- **Grammatical Accuracy**: error LanguageTool per 100 kata + kompleksitas struktur
- **Overall Band** 1–9 dibulatkan ke kelipatan 0.5 (aturan IELTS)
- Transkrip dengan highlight filler & jeda, daftar error grammar + saran

## Instalasi

Prasyarat: Python 3.9+, **Java** (17+ → LanguageTool 6.8; 9–16 → otomatis 6.5), `ffmpeg` opsional (hanya untuk `.mp3`).

```bash
git clone <repo-url> && cd ielts-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_models.sh          # model Vosk -> models/  + spaCy en_core_web_sm
uvicorn app:app --host 0.0.0.0 --port 8000
```

Buka **http://localhost:8000**. Saat start pertama, LanguageTool (~250 MB) diunduh sekali ke `~/.cache/language_tool_python/`.

Struktur folder model:
```
models/vosk-model-small-en-us-0.15/{am,conf,graph,ivector,...}
```
Lokasi lain: `VOSK_MODEL_PATH=/path/ke/model`.

## API
| Method | Path | Keterangan |
|---|---|---|
| `POST` | `/analyze` | form-data field `audio` (.wav/.mp3) → JSON transkrip + skor |
| `GET` | `/health` | status model (vosk, spacy, languagetool, ffmpeg) |
| `GET` | `/` | frontend `index.html` |

Contoh: `curl -F audio=@sample.wav http://localhost:8000/analyze`

## Deploy
Backend butuh Java + proses LanguageTool yang persisten → **tidak cocok untuk Vercel serverless**.
Gunakan VPS / container (Hugging Face Spaces Docker, Render, Railway, Fly.io, Cloud Run).
Frontend bisa di-host terpisah (mis. Vercel) dengan mengubah `API_BASE` di `index.html`.

> ⚠️ Skor adalah estimasi heuristik otomatis, bukan skor resmi IELTS.
