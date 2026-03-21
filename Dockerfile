FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ── 시스템 패키지 ──
RUN apt-get update && apt-get install -y \
    software-properties-common curl git build-essential \
    ffmpeg espeak-ng espeak-ng-data libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── pip 설치 ──
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

WORKDIR /app

# ── Python 패키지 (레이어 캐시 활용) ──
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir librosa

# ── 앱 코드 ──
COPY main.py .
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
