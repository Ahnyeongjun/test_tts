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
        _model_status = {"status": "loading", "message": "XTTS v2 로딩 중..."}
        import os
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        tts_home = os.environ.get("TTS_HOME", os.path.expanduser("~/.local/share/tts"))
        model_dir = os.path.join(tts_home, "tts", "tts_models--multilingual--multi-dataset--xtts_v2")

        # 모델 없으면 다운로드
        if not os.path.exists(os.path.join(model_dir, "model.pth")):
            _model_status = {"status": "loading", "message": "XTTS v2 다운로드 중... (~1.8GB)"}
            from TTS.api import TTS
            TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        _xtts = model
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

    import librosa

    # 참조 오디오에서 화자 임베딩 추출
    gpt_cond_latent, speaker_embedding = _xtts.get_conditioning_latents(
        audio_path=ref_paths,
        gpt_cond_len=30,
        max_ref_length=60,
    )

    # 한국어로 직접 추론
    out = _xtts.inference(
        text=text,
        language="ko",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.85,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True,
    )

    audio = np.array(out["wav"], dtype=np.float32)
    sr = 24000  # XTTS v2 출력 샘플레이트

    # 끝부분 노이즈·무음만 제거 (앞은 건드리지 않음)
    _, trim_idx = librosa.effects.trim(audio, top_db=25, frame_length=2048, hop_length=512)
    audio = audio[:trim_idx[1]]
    # 끝 0.1초 fade-out
    fade_len = int(sr * 0.1)
    if len(audio) > fade_len:
        audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    # 속도 조절 (pitch 보존 time-stretch)
    if abs(speed - 1.0) > 0.01:
        audio = librosa.effects.time_stretch(audio, rate=speed)

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
