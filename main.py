import uuid
import asyncio
import threading
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

VOICES_DIR = Path("voices")
OUTPUTS_DIR = Path("outputs")
for d in (VOICES_DIR, OUTPUTS_DIR):
    d.mkdir(exist_ok=True)

# ──────────────────────────── XTTS v2 모델 ────────────────────────────

_xtts = None
_model_status = {"status": "loading", "message": "XTTS v2 로딩 중..."}


def _load_model():
    global _xtts, _model_status
    try:
        _model_status = {"status": "loading", "message": "XTTS v2 로딩 중... (첫 실행 시 ~1.8GB 다운로드)"}
        import os
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        _model_status = {"status": "ready", "message": "준비 완료"}
    except Exception as e:
        _model_status = {"status": "error", "message": str(e)}


app = FastAPI()


@app.on_event("startup")
async def startup():
    threading.Thread(target=_load_model, daemon=True).start()


# ──────────────────────────── 음성 생성 ────────────────────────────

def _generate_sync(text: str, ref_paths: list[str], speed: float) -> tuple[np.ndarray, int]:
    if _xtts is None:
        raise RuntimeError(_model_status.get("message", "모델이 아직 로딩 중이에요."))

    tmp_out = OUTPUTS_DIR / f"_tmp_{uuid.uuid4().hex}.wav"
    try:
        _xtts.tts_to_file(
            text=text,
            speaker_wav=ref_paths,
            language="ko",
            file_path=str(tmp_out),
        )
        audio, sr = sf.read(str(tmp_out), dtype="float32")
    finally:
        tmp_out.unlink(missing_ok=True)

    # 속도 조절 (재샘플링 방식)
    if abs(speed - 1.0) > 0.01:
        target_len = int(len(audio) / speed)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio, sr


# ──────────────────────────── API ────────────────────────────

@app.get("/voices")
def list_voices():
    exts = {".wav", ".mp3", ".flac", ".m4a"}
    files = [f.name for f in VOICES_DIR.iterdir() if f.suffix in exts]
    return {"voices": sorted(files)}


@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    if Path(file.filename).suffix not in {".wav", ".mp3", ".flac", ".m4a"}:
        raise HTTPException(400, "wav, mp3, flac, m4a 파일만 업로드 가능해요.")
    save_path = VOICES_DIR / file.filename
    save_path.write_bytes(await file.read())
    return {"message": "업로드 완료", "filename": file.filename}


@app.delete("/voices/{filename}")
def delete_voice(filename: str):
    p = VOICES_DIR / filename
    if not p.exists():
        raise HTTPException(404, "파일을 찾을 수 없어요.")
    p.unlink()
    return {"message": "삭제 완료"}


@app.get("/model-status")
def model_status():
    return _model_status


@app.post("/generate")
async def generate(
    text: str = Form(...),
    ref_audios: str = Form(...),  # JSON array of filenames
    speed: float = Form(default=1.0),
):
    if not text.strip():
        raise HTTPException(400, "텍스트를 입력해주세요.")

    if _model_status["status"] == "loading":
        raise HTTPException(503, "모델 로딩 중이에요. 잠시 후 다시 시도해주세요.")
    if _model_status["status"] == "error":
        raise HTTPException(500, f"모델 오류: {_model_status['message']}")

    try:
        refs = json.loads(ref_audios)
    except Exception:
        raise HTTPException(400, "잘못된 요청 형식이에요.")

    if not refs:
        raise HTTPException(400, "참조 오디오를 하나 이상 선택해주세요.")

    ref_paths = [str(VOICES_DIR / r) for r in refs if (VOICES_DIR / r).exists()]
    if not ref_paths:
        raise HTTPException(400, "선택한 참조 오디오 파일을 찾을 수 없어요.")

    loop = asyncio.get_event_loop()
    try:
        audio, sr = await loop.run_in_executor(
            None,
            lambda: _generate_sync(text.strip(), ref_paths, speed),
        )
    except Exception as e:
        raise HTTPException(500, f"음성 생성 실패: {e}")

    out_name = f"{uuid.uuid4().hex}.wav"
    sf.write(str(OUTPUTS_DIR / out_name), audio, sr)
    return {"output": out_name}


@app.get("/download/{filename}")
def download(filename: str):
    p = OUTPUTS_DIR / filename
    if not p.exists():
        raise HTTPException(404, "파일을 찾을 수 없어요.")
    return FileResponse(p, media_type="audio/wav", filename=filename)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
