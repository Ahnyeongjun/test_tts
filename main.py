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
RVC_DIR = Path("/rvc")
RVC_EXP_NAME = "myvoice"

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

def _set_status(msg: str, status: str = "running"):
    training_status["status"] = status
    training_status["message"] = msg


def _ensure_pretrained_models():
    """학습에 필요한 사전학습 모델 다운로드 (없을 때만)."""
    import subprocess
    assets = RVC_DIR / "assets"
    downloads = {
        assets / "hubert" / "hubert_base.pt":
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        assets / "pretrained_v2" / "f0G40k.pth":
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
        assets / "pretrained_v2" / "f0D40k.pth":
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
        assets / "rmvpe" / "rmvpe.pt":
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    }
    for dest, url in downloads.items():
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _set_status(f"다운로드 중: {dest.name}")
            subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)


def _run_training(voice_files: list[str]):
    global training_status, rvc_model
    import subprocess, os, shutil
    rvc_model = None
    training_status = {"status": "running", "message": "시작 중..."}
    try:
        # ── 1. 오디오를 40kHz wav로 변환해서 trainset 디렉토리에 저장 ──
        exp_dir = RVC_DIR / "logs" / RVC_EXP_NAME
        trainset_dir = exp_dir / "trainset"
        trainset_dir.mkdir(parents=True, exist_ok=True)

        total_sec = 0.0
        for fname in voice_files:
            src = VOICES_DIR / fname
            if not src.exists():
                continue
            dst = trainset_dir / (src.stem + ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-ar", "40000", "-ac", "1", str(dst)],
                check=True, capture_output=True,
            )
            data, _ = sf.read(str(dst), dtype="float32")
            total_sec += len(data) / 40000

        if total_sec == 0:
            return _set_status("학습할 오디오 파일이 없어요.", "error")
        if total_sec < 30:
            return _set_status(f"음성이 너무 짧아요 ({total_sec:.0f}초). 30초 이상 필요합니다.", "error")

        # ── 2. 사전학습 모델 확보 ──
        _set_status("사전학습 모델 확인 중...")
        _ensure_pretrained_models()

        # ── 3. 오디오 전처리 ──
        _set_status(f"오디오 전처리 중... ({total_sec:.0f}초 분량)")
        subprocess.run(
            ["python", "infer/modules/train/preprocess.py",
             str(trainset_dir), "40000", "2", str(exp_dir), "False", "3.0"],
            check=True, cwd=str(RVC_DIR), capture_output=True,
        )

        # ── 4. F0 추출 ──
        _set_status("F0(피치) 추출 중...")
        subprocess.run(
            ["python", "infer/modules/train/extract/extract_f0_print.py",
             str(exp_dir), "2", "rmvpe"],
            check=True, cwd=str(RVC_DIR), capture_output=True,
        )

        # ── 5. HuBERT 특징 추출 ──
        _set_status("HuBERT 특징 추출 중...")
        subprocess.run(
            ["python", "infer/modules/train/extract_feature_print.py",
             "cuda:0", "1", "0", "0", str(exp_dir), "v2", "true"],
            check=True, cwd=str(RVC_DIR), capture_output=True,
        )

        # ── 6. filelist.txt 생성 ──
        _set_status("파일 목록 생성 중...")
        import json, random as _random
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        feature_dir = exp_dir / "3_feature768"
        f0_dir      = exp_dir / "2a_f0"
        f0nsf_dir   = exp_dir / "2b-f0nsf"
        names = (
            {n.split(".")[0] for n in os.listdir(gt_wavs_dir)}
            & {n.split(".")[0] for n in os.listdir(feature_dir)}
            & {n.split(".")[0] for n in os.listdir(f0_dir)}
            & {n.split(".")[0] for n in os.listdir(f0nsf_dir)}
        )
        opt = [
            f"{gt_wavs_dir}/{n}.wav|{feature_dir}/{n}.npy|{f0_dir}/{n}.wav.npy|{f0nsf_dir}/{n}.wav.npy|0"
            for n in names
        ]
        mute_dir = RVC_DIR / "logs" / "mute"
        for _ in range(2):
            opt.append(
                f"{mute_dir}/0_gt_wavs/mute40k.wav|{mute_dir}/3_feature768/mute.npy"
                f"|{mute_dir}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy|0"
            )
        _random.shuffle(opt)
        (exp_dir / "filelist.txt").write_text("\n".join(opt))

        # config.json 복사
        config_src = RVC_DIR / "configs" / "v2" / "40k.json"
        if config_src.exists() and not (exp_dir / "config.json").exists():
            shutil.copy(str(config_src), str(exp_dir / "config.json"))

        # ── 7. 학습 ──
        _set_status("학습 중... (GPU에 따라 10~60분 소요)")
        assets = RVC_DIR / "assets" / "pretrained_v2"
        subprocess.run(
            ["python", "infer/modules/train/train.py",
             "-e", RVC_EXP_NAME, "-sr", "40k", "-f0", "1",
             "-bs", "4", "-g", "0", "-te", "200", "-se", "50",
             "-pg", str(assets / "f0G40k.pth"),
             "-pd", str(assets / "f0D40k.pth"),
             "-l", "1", "-c", "0", "-sw", "1", "-v", "v2"],
            check=True, cwd=str(RVC_DIR), capture_output=True,
        )

        # ── 8. 결과 모델 복사 ──
        weights = sorted((RVC_DIR / "weights").glob(f"{RVC_EXP_NAME}*.pth"))
        if not weights:
            return _set_status("학습 완료됐지만 모델 파일을 찾지 못했어요.", "error")
        shutil.copy(str(weights[-1]), str(MODELS_DIR / "voice.pth"))

        index_files = sorted(exp_dir.glob("added_*.index"))
        if index_files:
            shutil.copy(str(index_files[-1]), str(MODELS_DIR / "voice.index"))

        _set_status("학습 완료! 이제 음성 생성에 사용돼요.", "done")
    except subprocess.CalledProcessError as e:
        _set_status(f"스크립트 오류: {e.stderr.decode()[-300:] if e.stderr else str(e)}", "error")
    except Exception as e:
        _set_status(str(e), "error")


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
