# PyTorch 공식 이미지 — Python 3.10 + CUDA 12.1 + torch 2.3.1 포함
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1
ENV TTS_HOME=/app/models/tts_cache

# ── 시스템 패키지 ──
RUN sed -i 's|archive.ubuntu.com|mirror.kakao.com|g; s|security.ubuntu.com|mirror.kakao.com|g' /etc/apt/sources.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential g++ git wget ffmpeg espeak-ng espeak-ng-data libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python 패키지만 이미지에 설치 (코드·모델은 볼륨 마운트) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# --reload: 마운트된 코드 변경 시 자동 재시작
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
