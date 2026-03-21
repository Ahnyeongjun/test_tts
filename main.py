import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, BackgroundTasks, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

VOICES_DIR = Path("voices")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")
for d in (VOICES_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

SAMPLE_RATE = 24000

tts_pipeline = None
rvc_model = None
training_status = {"status": "idle", "message": "학습 전"}


# ──────────────────────────── TTS (edge-tts) ────────────────────────────

async def edge_tts_generate(text: str, voice: str, rate: str) -> np.ndarray:
    import asyncio
    import edge_tts
    import io

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    mp3_buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf.write(chunk["data"])

    mp3_buf.seek(0)
    if mp3_buf.getbuffer().nbytes == 0:
        raise RuntimeError("edge-tts가 오디오를 생성하지 못했어요.")

    # mp3 → float32 numpy (24kHz)
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_buf.read())
        mp3_path = f.name
    wav_path = mp3_path.replace(".mp3", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path],
            check=True, capture_output=True,
        )
        audio, _ = sf.read(wav_path, dtype="float32")
    finally:
        os.unlink(mp3_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    return audio


# ──────────────────────────── RVC ────────────────────────────

def load_rvc():
    global rvc_model
    if rvc_model is not None:
        return rvc_model
    pth = MODELS_DIR / "voice.pth"
    if not pth.exists():
        return None
    from rvc_python.infer import RVCModel
    idx = str(MODELS_DIR / "voice.index")
    m = RVCModel()
    m.load_model(str(pth), idx if Path(idx).exists() else "")
    rvc_model = m
    return rvc_model


def rvc_convert(audio: np.ndarray, f0_up_key: int = 0) -> np.ndarray:
    rvc = load_rvc()
    if rvc is None:
        return audio

    tmp_in = OUTPUTS_DIR / f"_rvc_in_{uuid.uuid4().hex}.wav"
    tmp_out = OUTPUTS_DIR / f"_rvc_out_{uuid.uuid4().hex}.wav"
    sf.write(str(tmp_in), audio, SAMPLE_RATE)
    try:
        rvc.infer_file(str(tmp_in), str(tmp_out), f0up_key=f0_up_key, f0method="rmvpe")
        result, _ = sf.read(str(tmp_out), dtype="float32")
        return result
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


# ──────────────────────────── 학습 ────────────────────────────

def _run_training(voice_files: list[str]):
    global training_status, rvc_model
    rvc_model = None  # 기존 모델 초기화
    training_status = {"status": "running", "message": "오디오 전처리 중..."}
    try:
        # 선택된 voice 파일 모두 합치기
        audio_chunks = []
        for fname in voice_files:
            path = VOICES_DIR / fname
            if not path.exists():
                continue
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
            audio_chunks.append(data)

        if not audio_chunks:
            training_status = {"status": "error", "message": "학습할 오디오 파일이 없어요."}
            return

        combined = np.concatenate(audio_chunks)
        combined_path = MODELS_DIR / "training_audio.wav"
        sf.write(str(combined_path), combined, SAMPLE_RATE)

        total_sec = len(combined) / SAMPLE_RATE
        if total_sec < 30:
            training_status = {
                "status": "error",
                "message": f"학습 음성이 너무 짧아요 ({total_sec:.0f}초). 최소 30초 이상 필요합니다.",
            }
            return

        training_status = {"status": "running", "message": f"RVC 학습 중... ({total_sec:.0f}초 분량)"}

        from rvc_python.train import train_model
        train_model(
            model_name="voice",
            audio_files=[str(combined_path)],
            save_dir=str(MODELS_DIR),
            epochs=100,
            sample_rate=SAMPLE_RATE,
        )
        training_status = {"status": "done", "message": "학습 완료! 이제 음성 생성에 사용돼요."}
    except Exception as e:
        training_status = {"status": "error", "message": str(e)}


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


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """사전 학습된 RVC .pth 또는 .index 파일 업로드"""
    global rvc_model
    suffix = Path(file.filename).suffix
    if suffix not in {".pth", ".index"}:
        raise HTTPException(400, ".pth 또는 .index 파일만 업로드 가능해요.")
    dest = MODELS_DIR / ("voice" + suffix)
    dest.write_bytes(await file.read())
    rvc_model = None  # 다음 호출 때 재로드
    return {"message": f"모델 업로드 완료 ({dest.name})"}


@app.get("/model-status")
def model_status():
    pth_exists = (MODELS_DIR / "voice.pth").exists()
    idx_exists = (MODELS_DIR / "voice.index").exists()
    return {
        "model_ready": pth_exists,
        "index_ready": idx_exists,
        "training": training_status,
    }


@app.post("/train")
async def start_train(
    background_tasks: BackgroundTasks,
    voice_files: str = Form(...),  # JSON array string
):
    import json
    files = json.loads(voice_files)
    if not files:
        raise HTTPException(400, "학습할 파일을 선택해주세요.")
    if training_status["status"] == "running":
        raise HTTPException(400, "이미 학습 중이에요.")
    background_tasks.add_task(_run_training, files)
    return {"message": "학습 시작"}


@app.post("/generate")
async def generate(
    text: str = Form(...),
    voice: str = Form(default="ko-KR-SunHiNeural"),
    speed: float = Form(default=1.0),
    use_rvc: bool = Form(default=True),
    f0_key: int = Form(default=0),
):
    if not text.strip():
        raise HTTPException(400, "텍스트를 입력해주세요.")

    try:
        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        audio = await edge_tts_generate(text.strip(), voice=voice, rate=rate_str)
    except Exception as e:
        raise HTTPException(500, f"TTS 생성 실패: {e}")

    try:
        if use_rvc and (MODELS_DIR / "voice.pth").exists():
            audio = rvc_convert(audio, f0_up_key=f0_key)
    except Exception as e:
        raise HTTPException(500, f"RVC 변환 실패: {e}")

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUTS_DIR / out_name
    sf.write(str(out_path), audio, SAMPLE_RATE)
    return {"output": out_name}


@app.get("/download/{filename}")
def download(filename: str):
    p = OUTPUTS_DIR / filename
    if not p.exists():
        raise HTTPException(404, "파일을 찾을 수 없어요.")
    return FileResponse(p, media_type="audio/wav", filename=filename)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
