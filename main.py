import re
import uuid
import wave
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
        import os
        from huggingface_hub import hf_hub_download
        from f5_tts.api import F5TTS

        # Korean fine-tuned model (team-lucid/F5-TTS-ko)
        ckpt_raw = hf_hub_download(repo_id="team-lucid/F5-TTS-ko", filename="pytorch_model.bin")
        vocab_json = hf_hub_download(repo_id="team-lucid/F5-TTS-ko", filename="vocab.json")

        # vocab.json → vocab.txt 변환 (F5TTS 포맷)
        import json, torch
        vocab_txt = vocab_json.replace("vocab.json", "vocab.txt")
        if not os.path.exists(vocab_txt):
            with open(vocab_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            tokens = sorted(data.items(), key=lambda x: x[1])
            with open(vocab_txt, "w", encoding="utf-8") as f:
                for token, _ in tokens:
                    f.write(token + "\n")

        # checkpoint 포맷 변환 (raw state_dict → model_state_dict 래핑)
        ckpt_converted = ckpt_raw.replace("pytorch_model.bin", "model_converted.pt")
        if not os.path.exists(ckpt_converted):
            state_dict = torch.load(ckpt_raw, map_location="cpu")
            torch.save({"model_state_dict": state_dict}, ckpt_converted)

        tts_model = F5TTS(ckpt_file=ckpt_converted, vocab_file=vocab_txt, use_ema=False)
    return tts_model


def split_sentences(text: str) -> list[str]:
    """마침표/느낌표/물음표 기준으로 문장 분리. 너무 짧은 조각은 합침."""
    parts = re.split(r'(?<=[.!?。！？])\s*', text.strip())
    sentences, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= 15:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences or [text]


def concat_wavs(paths: list[Path], output: Path):
    """여러 wav 파일을 하나로 합침."""
    with wave.open(str(output), "wb") as out_wav:
        for i, p in enumerate(paths):
            with wave.open(str(p), "rb") as w:
                if i == 0:
                    out_wav.setparams(w.getparams())
                out_wav.writeframes(w.readframes(w.getnframes()))


@app.get("/voices")
def list_voices():
    files = [f.name for f in VOICES_DIR.iterdir() if f.suffix in (".wav", ".mp3", ".flac", ".m4a")]
    return {"voices": files}


@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    if suffix not in (".wav", ".mp3", ".flac", ".m4a"):
        raise HTTPException(status_code=400, detail="wav, mp3, flac, m4a 파일만 업로드 가능해요.")
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

    tts = get_tts()
    sentences = split_sentences(text)
    tmp_files = []

    for i, sentence in enumerate(sentences):
        tmp_path = OUTPUTS_DIR / f"_tmp_{uuid.uuid4().hex}.wav"
        tts.infer(
            ref_file=str(voice_path),
            ref_text=ref_text,
            gen_text=sentence,
            file_wave=str(tmp_path),
        )
        tmp_files.append(tmp_path)

    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path = OUTPUTS_DIR / output_filename

    if len(tmp_files) == 1:
        tmp_files[0].rename(output_path)
    else:
        concat_wavs(tmp_files, output_path)
        for f in tmp_files:
            f.unlink(missing_ok=True)

    return {"output": output_filename}


@app.get("/download/{filename}")
def download(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없어요.")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
