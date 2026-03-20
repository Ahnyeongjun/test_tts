import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

VOICES_DIR = Path("voices")
OUTPUTS_DIR = Path("outputs")
VOICES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

tts_model = None


def get_tts():
    global tts_model
    if tts_model is None:
        from f5_tts.api import F5TTS
        tts_model = F5TTS()
    return tts_model


@app.get("/voices")
def list_voices():
    files = [f.name for f in VOICES_DIR.iterdir() if f.suffix in (".wav", ".mp3", ".flac", ".m4a")]
    return {"voices": files}


@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    if suffix not in (".wav", ".mp3", ".flac", ".m4a"):
        raise HTTPException(status_code=400, detail="wav, mp3, flac 파일만 업로드 가능해요.")
    save_path = VOICES_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"message": "업로드 완료", "filename": file.filename}


@app.post("/generate")
async def generate(
    text: str = Form(...),
    voice_file: str = Form(...),
    ref_text: str = Form(default=""),
):
    voice_path = VOICES_DIR / voice_file
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail="목소리 파일을 찾을 수 없어요.")

    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path = OUTPUTS_DIR / output_filename

    tts = get_tts()
    tts.infer(
        ref_file=str(voice_path),
        ref_text=ref_text,   # 비워두면 자동 전사
        gen_text=text,
        output_file=str(output_path),
    )

    return {"output": output_filename}


@app.get("/download/{filename}")
def download(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없어요.")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
