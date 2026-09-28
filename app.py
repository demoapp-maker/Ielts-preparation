"""
===============================================================================
 IELTS Speaking Practice Analyzer — BACKEND (FastAPI, CPU-only)
===============================================================================

Semua model AI berjalan di SERVER (bukan di browser):
  • Vosk (vosk-model-small-en-us)  -> Speech-to-Text offline + word timestamps
  • LanguageTool (offline, Java)   -> cek grammar
  • spaCy (en_core_web_sm)         -> analisis kosakata (TTR, POS, dependensi)

-------------------------------------------------------------------------------
 CARA INSTALASI (ringkas)
-------------------------------------------------------------------------------
 1) Python 3.9+  (disarankan virtualenv)
        python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate

 2) Install dependensi pip:
        pip install fastapi "uvicorn[standard]" python-multipart vosk \\
                    language-tool-python spacy numpy
    (atau: pip install -r requirements.txt)

 3) Download model spaCy:
        python -m spacy download en_core_web_sm

 4) Java untuk LanguageTool (offline):
        - Java 17+  -> LanguageTool 6.8 (default terbaru)
        - Java 9-16 -> otomatis memakai LanguageTool 6.5
        Ubuntu: sudo apt install openjdk-17-jre-headless
        Saat server pertama kali start, language-tool-python akan mengunduh
        LanguageTool (~250 MB) SATU KALI ke ~/.cache/language_tool_python/.
        Setelah itu semuanya berjalan 100% offline.

 5) Model Vosk (~40 MB) — taruh di folder "models/" di samping app.py:
        mkdir -p models && cd models
        wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
        unzip vosk-model-small-en-us-0.15.zip
        cd ..
    Struktur akhirnya:
        ielts-analyzer/
        ├── app.py
        ├── index.html
        └── models/
            └── vosk-model-small-en-us-0.15/
                ├── am/  ├── conf/  ├── graph/  └── ivector/ ...
    Lokasi lain bisa diatur lewat env var:  VOSK_MODEL_PATH=/path/ke/model

 6) (Opsional) ffmpeg — HANYA diperlukan untuk upload .mp3 / format non-WAV.
    File WAV PCM (yang dikirim oleh index.html) diproses tanpa ffmpeg.
        Ubuntu: sudo apt install ffmpeg

 7) Jalankan server:
        uvicorn app:app --host 0.0.0.0 --port 8000
    Lalu buka http://localhost:8000  (index.html disajikan langsung oleh server)

 Catatan: akses mikrofon di browser memerlukan "secure context"
 (https:// atau http://localhost).
===============================================================================
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------------------------------------------------------
# Konfigurasi
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VOSK_MODEL_PATH = Path(os.getenv("VOSK_MODEL_PATH", BASE_DIR / "models" / "vosk-model-small-en-us-0.15"))
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "360"))  # 6 menit
TARGET_SR = 16000  # Vosk small model dilatih pada 16 kHz mono

LONG_PAUSE_SEC = 1.5      # jeda tidak alami (sesuai spesifikasi)
SENTENCE_GAP_SEC = 0.8    # jeda yang dianggap batas "kalimat" (Vosk tidak memberi tanda baca)

# Filler words sesuai spesifikasi (+ beberapa varian ejaan yang sering keluar dari ASR)
FILLER_SINGLE = {"uh", "um", "ah", "err", "er", "erm", "uhm", "umm", "hmm", "like"}
FILLER_BIGRAMS = {("you", "know")}

# Transisi akademis / discourse markers (frasa multi-kata didukung)
ACADEMIC_TRANSITIONS = [
    "however", "therefore", "furthermore", "moreover", "nevertheless", "nonetheless",
    "consequently", "thus", "hence", "in addition", "additionally", "on the other hand",
    "in contrast", "as a result", "for instance", "for example", "in conclusion",
    "to sum up", "overall", "whereas", "although", "even though", "similarly", "likewise",
    "meanwhile", "firstly", "secondly", "finally", "in fact", "indeed", "as well as",
    "not only", "in other words", "that is to say", "on the contrary", "subsequently",
    "in particular", "particularly", "admittedly", "undoubtedly", "generally speaking",
]
BASIC_CONNECTORS = ["because", "so", "but", "and then", "also", "first", "then", "if", "when", "while"]

# Kategori / rule LanguageTool yang DIABAIKAN karena transkrip ASR tidak punya
# huruf kapital & tanda baca yang andal (bukan kesalahan si pembicara).
LT_IGNORED_CATEGORIES = {"TYPOS", "CASING", "PUNCTUATION", "TYPOGRAPHY", "STYLE", "REDUNDANCY"}
LT_IGNORED_RULES = {
    "UPPERCASE_SENTENCE_START", "I_LOWERCASE", "PUNCTUATION_PARAGRAPH_END",
    "COMMA_PARENTHESIS_WHITESPACE", "WHITESPACE_RULE", "EN_QUOTES", "MORFOLOGIK_RULE_EN_US",
    "SENTENCE_FRAGMENT", "EN_UNPAIRED_BRACKETS", "ENGLISH_WORD_REPEAT_BEGINNING_RULE",
    "SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA", "EN_COMPOUNDS", "WORD_REPEAT_RULE",
}

log = logging.getLogger("ielts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Model global (dimuat sekali saat startup)
MODELS: dict[str, Any] = {"vosk": None, "nlp": None, "lt": None, "lt_error": None}
LT_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Loader model
# -----------------------------------------------------------------------------
def _pick_languagetool_version() -> str:
    """LanguageTool 6.6+ butuh Java 17. Pilih versi otomatis sesuai Java yang terpasang."""
    if os.getenv("LT_VERSION"):
        return os.environ["LT_VERSION"]
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=15)
        m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
        if m:
            major = int(m.group(1))
            if major == 1 and m.group(2):  # format lama "1.8.0_xxx"
                major = int(m.group(2))
            return "6.8" if major >= 17 else "6.5"
    except Exception:  # noqa: BLE001
        pass
    return "6.8"


def load_models() -> None:
    from vosk import Model, SetLogLevel
    import spacy

    SetLogLevel(-1)
    if not VOSK_MODEL_PATH.exists():
        raise RuntimeError(
            f"Model Vosk tidak ditemukan di '{VOSK_MODEL_PATH}'. "
            "Unduh vosk-model-small-en-us-0.15 lalu ekstrak ke folder models/ (lihat docstring app.py)."
        )
    t = time.time()
    MODELS["vosk"] = Model(str(VOSK_MODEL_PATH))
    log.info("Vosk model loaded (%.1fs)", time.time() - t)

    t = time.time()
    MODELS["nlp"] = spacy.load(SPACY_MODEL)
    log.info("spaCy '%s' loaded (%.1fs)", SPACY_MODEL, time.time() - t)

    # LanguageTool: jika gagal (mis. Java tidak ada), server tetap jalan,
    # tetapi skor grammar ditandai "tidak tersedia".
    try:
        import language_tool_python

        t = time.time()
        version = _pick_languagetool_version()
        MODELS["lt"] = language_tool_python.LanguageTool("en-US", language_tool_download_version=version)
        log.info("LanguageTool %s loaded (%.1fs)", version, time.time() - t)
    except Exception as exc:  # noqa: BLE001
        MODELS["lt_error"] = str(exc)
        log.error("LanguageTool gagal dimuat: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield
    if MODELS["lt"] is not None:
        try:
            MODELS["lt"].close()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="IELTS Speaking Practice Analyzer", version="1.0.0", lifespan=lifespan)
# CORS dibuka agar index.html juga bisa di-host terpisah (mis. file:// atau server statis lain)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# -----------------------------------------------------------------------------
# Utilitas audio
# -----------------------------------------------------------------------------
def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    # Anti-aliasing sederhana (moving average) saat downsampling rasio besar
    ratio = sr_in / sr_out
    if ratio >= 2:
        k = int(ratio)
        x = np.convolve(x, np.ones(k) / k, mode="same")
    n_out = int(round(x.size * sr_out / sr_in))
    t_in = np.arange(x.size) / sr_in
    t_out = np.arange(n_out) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def _decode_wav_pcm(raw: bytes) -> np.ndarray | None:
    """Decode WAV PCM murni-Python (tanpa ffmpeg). Return float32 mono 16 kHz atau None."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            ch, sw, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            frames = wf.readframes(n)
    except (wave.Error, EOFError):
        return None  # bukan WAV PCM (mis. WAV float / mp3) -> coba ffmpeg

    if sw == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608.0
    else:
        return None
    if ch > 1:
        data = data[: data.size - data.size % ch].reshape(-1, ch).mean(axis=1)
    return _resample_linear(data, sr, TARGET_SR)


def _decode_with_ffmpeg(raw: bytes) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(
            status_code=415,
            detail="Format audio ini (mis. .mp3/.webm) butuh ffmpeg di server. "
                   "Install ffmpeg atau kirim file WAV PCM.",
        )
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(TARGET_SR), "-f", "s16le", "pipe:1"],
        input=raw, capture_output=True, timeout=180,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(status_code=400, detail=f"Gagal decode audio: {proc.stderr.decode()[:300]}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def load_audio_16k_mono(raw: bytes) -> np.ndarray:
    audio = _decode_wav_pcm(raw)
    if audio is None:
        audio = _decode_with_ffmpeg(raw)
    return audio


# -----------------------------------------------------------------------------
# Speech-to-Text (Vosk)
# -----------------------------------------------------------------------------
def transcribe(audio: np.ndarray) -> list[dict]:
    """Return list kata: [{'word','start','end','conf'}, ...] dengan timestamp (detik)."""
    from vosk import KaldiRecognizer

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    rec = KaldiRecognizer(MODELS["vosk"], TARGET_SR)
    rec.SetWords(True)

    words: list[dict] = []
    chunk = 8000  # 4000 sampel = 0.25 detik
    for i in range(0, len(pcm), chunk):
        if rec.AcceptWaveform(pcm[i:i + chunk]):
            words.extend(json.loads(rec.Result()).get("result", []))
    words.extend(json.loads(rec.FinalResult()).get("result", []))
    return [w for w in words if w.get("word") and w["word"] != "[unk]"]


# -----------------------------------------------------------------------------
# Helper skor
# -----------------------------------------------------------------------------
def round_band(x: float) -> float:
    """Pembulatan IELTS: ke kelipatan 0.5 terdekat (.25 -> naik ke .5, .75 -> naik ke bulat)."""
    return float(max(1.0, min(9.0, math.floor(x * 2 + 0.5 + 1e-9) / 2)))


def interp_score(value: float, points: list[tuple[float, float]]) -> float:
    xs, ys = zip(*points)
    return float(np.interp(value, xs, ys))


def clamp(x: float, lo: float = 1.0, hi: float = 9.0) -> float:
    return max(lo, min(hi, x))


def find_phrases(text_lower: str, phrases: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for p in phrases:
        n = len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
        if n:
            found[p] = n
    return found


# -----------------------------------------------------------------------------
# Deteksi filler & jeda
# -----------------------------------------------------------------------------
def mark_fillers(words: list[dict]) -> None:
    """Menandai w['is_filler'] = True untuk filler words.
    'like' hanya dianggap filler jika BUKAN kata kerja ("I like pizza") dan
    muncul di posisi khas filler (diapit jeda, diulang, atau ditag INTJ oleh spaCy).
    'you know' dianggap filler kecuali didahului 'do/did/don't' (pertanyaan)."""
    from spacy.tokens import Doc

    nlp = MODELS["nlp"]
    tokens = [w["word"].lower() for w in words]
    for w in words:
        w["is_filler"] = False
    if not tokens:
        return

    doc = Doc(nlp.vocab, words=tokens)  # 1 token spaCy = 1 kata Vosk (alignment terjaga)
    for _, proc in nlp.pipeline:
        doc = proc(doc)

    def gap_before(i: int) -> float:
        return words[i]["start"] - words[i - 1]["end"] if i > 0 else 0.0

    def gap_after(i: int) -> float:
        return words[i + 1]["start"] - words[i]["end"] if i < len(words) - 1 else 0.0

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # bigram "you know"
        if i + 1 < len(tokens) and (tok, tokens[i + 1]) in FILLER_BIGRAMS:
            prev = tokens[i - 1] if i > 0 else ""
            if prev not in {"do", "did", "don't", "didn't", "if"}:
                words[i]["is_filler"] = words[i + 1]["is_filler"] = True
                i += 2
                continue
        if tok == "like":
            pos = doc[i].pos_
            prev_like = i > 0 and tokens[i - 1] == "like"
            paused = gap_before(i) >= 0.4 or gap_after(i) >= 0.4
            if pos == "INTJ" or (pos != "VERB" and (paused or prev_like)):
                words[i]["is_filler"] = True
        elif tok in FILLER_SINGLE:
            words[i]["is_filler"] = True
        i += 1


def find_long_pauses(words: list[dict]) -> list[dict]:
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > LONG_PAUSE_SEC:
            pauses.append({
                "after_index": i - 1,
                "after_word": words[i - 1]["word"],
                "before_word": words[i]["word"],
                "at_sec": round(words[i - 1]["end"], 2),
                "duration_sec": round(gap, 2),
            })
    return pauses


def build_clean_text(words: list[dict]) -> str:
    """Susun teks tanpa filler + tanda titik pada jeda panjang (pseudo-kalimat),
    agar LanguageTool & spaCy bekerja lebih akurat pada output ASR."""
    sentences: list[list[str]] = [[]]
    last_end = None
    for w in words:
        if last_end is not None and w["start"] - last_end >= SENTENCE_GAP_SEC and sentences[-1]:
            sentences.append([])
        last_end = w["end"]
        if w.get("is_filler"):
            continue
        sentences[-1].append("I" if w["word"] == "i" else re.sub(r"^i'", "I'", w["word"]))
    out = []
    for s in sentences:
        if not s:
            continue
        text = " ".join(s)
        out.append(text[0].upper() + text[1:] + ".")
    return " ".join(out)


# -----------------------------------------------------------------------------
# Komponen skor IELTS
# -----------------------------------------------------------------------------
def score_fluency(words: list[dict], long_pauses: list[dict], transcript_lower: str) -> dict:
    total_words = len(words)
    speaking_time = max(words[-1]["end"] - words[0]["start"], 1.0)
    minutes = speaking_time / 60.0
    wpm = total_words / minutes

    filler_counts: dict[str, int] = {}
    i = 0
    while i < len(words):
        w = words[i]
        if w["is_filler"]:
            if i + 1 < len(words) and words[i + 1]["is_filler"] and \
                    (w["word"].lower(), words[i + 1]["word"].lower()) in FILLER_BIGRAMS:
                key = "you know"
                i += 1
            else:
                key = w["word"].lower()
            filler_counts[key] = filler_counts.get(key, 0) + 1
        i += 1
    filler_total = sum(filler_counts.values())
    filler_pct = filler_total / max(total_words, 1) * 100
    # lantai 0.75 menit agar jawaban sangat pendek tidak dihukum berlebihan
    pauses_per_min = len(long_pauses) / max(minutes, 0.75)

    # 1) Kecepatan bicara -> skor dasar
    if wpm <= 170:
        base = interp_score(wpm, [(30, 2.5), (50, 3.5), (70, 4.5), (90, 5.5), (110, 6.5), (130, 7.5), (150, 8.5), (170, 9.0)])
    else:  # terlalu cepat -> sedikit penalti (terburu-buru)
        base = max(6.5, 9.0 - (wpm - 170) * 0.05)
    # 2) Penalti filler (toleransi 1% kata)
    filler_pen = min(3.0, max(0.0, (filler_pct - 1.0) * 0.35))
    # 3) Penalti jeda tidak alami (> 1.5 detik)
    pause_pen = min(2.5, pauses_per_min * 0.5)
    # 4) Coherence: variasi penghubung wacana
    markers = find_phrases(transcript_lower, ACADEMIC_TRANSITIONS + BASIC_CONNECTORS)
    coherence_bonus = 0.5 if len(markers) >= 4 else (0.25 if len(markers) >= 2 else 0.0)

    raw = clamp(base - filler_pen - pause_pen + coherence_bonus)

    feedback = []
    if wpm < 90:
        feedback.append("Tempo bicara lambat. Latih berbicara lebih lancar tanpa terlalu banyak berhenti.")
    elif wpm > 185:
        feedback.append("Tempo sangat cepat — perlambat sedikit agar ide lebih jelas.")
    else:
        feedback.append("Tempo bicara sudah berada di rentang yang natural.")
    if filler_pct > 3:
        feedback.append(f"Filler words cukup sering ({filler_pct:.1f}% dari kata). Ganti dengan jeda singkat atau frasa seperti 'Well, let me think...'.")
    if long_pauses:
        feedback.append(f"Terdeteksi {len(long_pauses)} jeda panjang (>{LONG_PAUSE_SEC:g} detik). Siapkan ide dengan pola PREP/story agar alur tidak terputus.")
    if len(markers) < 2:
        feedback.append("Gunakan lebih banyak penghubung (because, however, for example) agar jawaban koheren.")

    return {
        "score": round_band(raw),
        "raw": round(raw, 2),
        "metrics": {
            "wpm": round(wpm, 1),
            "speaking_time_sec": round(speaking_time, 2),
            "total_words": total_words,
            "filler_total": filler_total,
            "filler_percent": round(filler_pct, 2),
            "long_pause_count": len(long_pauses),
            "long_pauses_per_min": round(pauses_per_min, 2),
            "discourse_markers": markers,
        },
        "fillers": filler_counts,
        "long_pauses": long_pauses,
        "breakdown": {
            "wpm_base": round(base, 2), "filler_penalty": round(filler_pen, 2),
            "pause_penalty": round(pause_pen, 2), "coherence_bonus": coherence_bonus,
        },
        "feedback": feedback,
    }


def score_lexical(doc, transcript_lower: str) -> dict:
    tokens = [t for t in doc if t.is_alpha]
    forms = [t.text.lower() for t in tokens]
    n = len(forms)
    ttr = len(set(forms)) / n if n else 0.0

    # MATTR (Moving-Average TTR, window 50): TTR yang lebih stabil terhadap panjang jawaban
    window = 50
    if n > window:
        ttrs = [len(set(forms[i:i + window])) / window for i in range(n - window + 1)]
        mattr = float(np.mean(ttrs))
    else:
        mattr = ttr

    content = [t for t in tokens if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and not t.is_stop]
    lexical_density = len(content) / n if n else 0.0
    advanced = sorted({t.lemma_.lower() for t in content if len(t.text) >= 8})
    advanced_ratio = len([t for t in content if len(t.text) >= 8]) / max(len(content), 1)
    transitions = find_phrases(transcript_lower, ACADEMIC_TRANSITIONS)

    base = interp_score(mattr, [(0.35, 3.5), (0.45, 4.5), (0.55, 5.5), (0.62, 6.5), (0.70, 7.5), (0.78, 8.5), (0.85, 9.0)])
    transition_bonus = min(1.0, 0.25 * len(transitions))
    adv_adj = 0.5 if advanced_ratio >= 0.20 else (-0.5 if advanced_ratio < 0.07 else 0.0)
    raw = clamp(base + transition_bonus + adv_adj)

    warnings = []
    if n < 50:  # TTR sampel pendek cenderung tinggi secara semu
        raw = min(raw, 6.0)
        warnings.append("Sampel < 50 kata: skor Lexical dibatasi maks. 6.0 karena TTR belum representatif.")

    feedback = []
    if mattr < 0.55:
        feedback.append("Banyak pengulangan kata. Gunakan sinonim & parafrase (mis. 'important' -> 'crucial', 'significant').")
    else:
        feedback.append("Variasi kosakata cukup baik.")
    if not transitions:
        feedback.append("Belum ada transisi akademis. Coba sisipkan 'however', 'therefore', 'furthermore', 'for instance'.")
    if advanced_ratio < 0.1:
        feedback.append("Tambahkan kosakata yang lebih spesifik/less common dan collocation topik.")

    return {
        "score": round_band(raw),
        "raw": round(raw, 2),
        "metrics": {
            "tokens": n,
            "unique_tokens": len(set(forms)),
            "ttr": round(ttr, 3),
            "mattr_50": round(mattr, 3),
            "lexical_density": round(lexical_density, 3),
            "advanced_word_ratio": round(advanced_ratio, 3),
            "academic_transitions": transitions,
            "advanced_words": advanced[:25],
        },
        "breakdown": {"mattr_base": round(base, 2), "transition_bonus": transition_bonus, "advanced_vocab_adj": adv_adj},
        "feedback": feedback,
        "warnings": warnings,
    }


def _m(match, *names, default=None):
    """Kompatibel dengan versi language-tool-python lama (camelCase) & baru (snake_case)."""
    for n in names:
        if hasattr(match, n):
            return getattr(match, n)
    return default


def score_grammar(clean_text: str, doc, word_count: int) -> dict:
    lt = MODELS["lt"]
    if lt is None:
        return {"available": False, "score": None, "raw": None,
                "error": MODELS["lt_error"] or "LanguageTool tidak tersedia",
                "metrics": {}, "errors": [], "feedback": ["LanguageTool tidak aktif di server (cek instalasi Java)."]}

    with LT_LOCK:
        matches = lt.check(clean_text)

    errors = []
    for m in matches:
        rule_id = _m(m, "rule_id", "ruleId", default="")
        category = _m(m, "category", default="")
        if category in LT_IGNORED_CATEGORIES or rule_id in LT_IGNORED_RULES:
            continue
        offset = _m(m, "offset", default=0)
        length = _m(m, "error_length", "errorLength", default=0)
        errors.append({
            "rule_id": rule_id,
            "category": category,
            "message": _m(m, "message", default=""),
            "text": clean_text[offset:offset + length],
            "context": clean_text[max(0, offset - 30):offset + length + 30],
            "replacements": list(_m(m, "replacements", default=[]) or [])[:3],
        })

    errors_per_100 = len(errors) / max(word_count, 1) * 100

    # Kompleksitas kalimat (range of structures) dari parse dependensi spaCy
    complex_deps = {"advcl", "ccomp", "relcl", "acl", "csubj", "xcomp"}
    complex_count = sum(1 for t in doc if t.dep_ in complex_deps)
    complex_per_100 = complex_count / max(word_count, 1) * 100
    sentences = [s for s in doc.sents if any(t.is_alpha for t in s)]
    avg_sentence_len = word_count / max(len(sentences), 1)

    accuracy = clamp(9.0 - 0.45 * errors_per_100)
    if complex_per_100 >= 6:
        range_cap = 9.0
    elif complex_per_100 >= 3:
        range_cap = 8.0
    elif complex_per_100 >= 1.5:
        range_cap = 7.0
    elif complex_count > 0:
        range_cap = 6.0
    else:
        range_cap = 5.0
    raw = min(accuracy, range_cap)

    warnings = []
    if word_count < 20:
        raw = min(raw, 6.0)
        warnings.append("Sampel < 20 kata: skor Grammar dibatasi maks. 6.0.")

    feedback = []
    if errors:
        feedback.append(f"Ditemukan {len(errors)} potensi kesalahan grammar ({errors_per_100:.1f} per 100 kata).")
    else:
        feedback.append("Tidak ditemukan kesalahan grammar yang jelas pada transkrip.")
    if range_cap <= 6.0:
        feedback.append("Variasikan struktur: gunakan klausa relatif (which/who), kondisional (if), dan klausa keterangan (although/because).")

    return {
        "available": True,
        "score": round_band(raw),
        "raw": round(raw, 2),
        "metrics": {
            "error_count": len(errors),
            "errors_per_100_words": round(errors_per_100, 2),
            "complex_structures": complex_count,
            "complex_per_100_words": round(complex_per_100, 2),
            "avg_sentence_length": round(avg_sentence_len, 1),
        },
        "breakdown": {"accuracy_score": round(accuracy, 2), "range_cap": range_cap},
        "errors": errors,
        "feedback": feedback,
        "warnings": warnings,
    }


def pronunciation_proxy(words: list[dict]) -> dict:
    """EKSPERIMENTAL (tidak dihitung ke overall): rata-rata confidence Vosk sebagai proxy kejelasan."""
    confs = [w.get("conf", 1.0) for w in words]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    low = [w["word"] for w in words if w.get("conf", 1.0) < 0.6]
    est = interp_score(mean_conf, [(0.5, 4.0), (0.7, 5.0), (0.8, 6.0), (0.88, 7.0), (0.94, 8.0), (0.98, 9.0)])
    return {"estimate": round_band(est), "mean_confidence": round(mean_conf, 3),
            "low_confidence_words": low[:20], "counted_in_overall": False}


# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vosk": MODELS["vosk"] is not None,
        "spacy": MODELS["nlp"] is not None,
        "languagetool": MODELS["lt"] is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


# Catatan: sengaja memakai "def" (bukan async def) agar FastAPI menjalankan
# pekerjaan CPU-berat ini di threadpool dan tidak memblokir event loop.
@app.post("/analyze")
def analyze(audio: UploadFile = File(..., description="File audio .wav atau .mp3")):
    t0 = time.time()
    raw = audio.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File audio kosong.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File terlalu besar (maks {MAX_UPLOAD_MB:g} MB).")

    samples = load_audio_16k_mono(raw)
    duration = samples.size / TARGET_SR
    if duration < 1.0:
        raise HTTPException(status_code=422, detail="Audio terlalu pendek (< 1 detik).")
    if duration > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=413, detail=f"Audio terlalu panjang (maks {MAX_AUDIO_SECONDS:g} detik).")

    # 1) Speech-to-Text
    words = transcribe(samples)
    if len(words) < 5:
        raise HTTPException(status_code=422, detail="Ucapan tidak terdeteksi / terlalu sedikit. Bicaralah lebih jelas & dekat dengan mikrofon.")
    t_stt = time.time()

    # 2) Pra-proses
    mark_fillers(words)
    long_pauses = find_long_pauses(words)
    transcript = " ".join(w["word"] for w in words)
    transcript_lower = transcript.lower()
    clean_text = build_clean_text(words)
    doc = MODELS["nlp"](clean_text)
    content_word_count = sum(1 for t in doc if t.is_alpha)

    # 3) Skor komponen
    fluency = score_fluency(words, long_pauses, transcript_lower)
    lexical = score_lexical(doc, transcript_lower)
    grammar = score_grammar(clean_text, doc, content_word_count)
    pron = pronunciation_proxy(words)

    # 4) Overall band = rata-rata komponen, dibulatkan ke 0.5 (standar IELTS)
    comp = [fluency["score"], lexical["score"]] + ([grammar["score"]] if grammar["available"] else [])
    overall_raw = sum(comp) / len(comp)
    overall = round_band(overall_raw)

    warnings = lexical.pop("warnings", []) + grammar.pop("warnings", [])
    if duration < 30:
        warnings.insert(0, "Rekaman < 30 detik. Untuk estimasi lebih akurat, bicaralah 1–2 menit (seperti Part 2).")
    if not grammar["available"]:
        warnings.append("Grammar tidak dinilai (LanguageTool tidak aktif); overall dihitung dari 2 komponen.")

    return JSONResponse({
        "transcript": transcript,
        "transcript_clean": clean_text,
        "words": [
            {"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2),
             "conf": round(w.get("conf", 1.0), 3), "is_filler": w["is_filler"]}
            for w in words
        ],
        "audio_duration_sec": round(duration, 2),
        "scores": {
            "fluency_coherence": fluency["score"],
            "lexical_resource": lexical["score"],
            "grammatical_accuracy": grammar["score"],
            "overall_band": overall,
            "overall_raw_average": round(overall_raw, 3),
        },
        "fluency": fluency,
        "lexical": lexical,
        "grammar": grammar,
        "pronunciation_proxy": pron,
        "overall_band": overall,
        "warnings": warnings,
        "timing_ms": {"stt": int((t_stt - t0) * 1000), "total": int((time.time() - t0) * 1000)},
        "disclaimer": "Estimasi otomatis berbasis heuristik, bukan skor resmi IELTS.",
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
