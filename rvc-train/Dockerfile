FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 시스템 패키지
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 build-essential cmake tzdata curl \
    && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사 후 pip 설치 (캐시 활용)
COPY requirements-train.txt /app/requirements-train.txt
RUN pip install --upgrade pip==23.3.2 && \
    pip install -r /app/requirements-train.txt --use-deprecated=legacy-resolver && \
    pip install --upgrade pip

# RVC 클론
RUN git clone --depth=1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI /app/rvc

WORKDIR /app/rvc

COPY download_models.sh /app/download_models.sh
COPY train.sh /app/train.sh

RUN chmod +x /app/download_models.sh /app/train.sh
