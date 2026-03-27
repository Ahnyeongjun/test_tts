# 내 목소리 TTS — 기술 문서

내 목소리 샘플로 RVC 모델을 학습시키고, edge-tts(Microsoft Neural TTS)로 생성된 음성을 내 목소리 톤으로 변환해주는 시스템입니다.

---

## 왜 이 조합을 쓰는가 (edge-tts + RVC)

### 선택지 비교

| 방식 | 장점 | 단점 |
|------|------|------|
| **직접 TTS 학습** (e.g. VITS, YourTTS) | 처음부터 내 목소리로 생성 | 고품질 학습에 수십 시간·수천 문장 필요 |
| **edge-tts만 사용** | 즉시 사용, 고품질 발음 | 내 목소리가 아님 |
| **edge-tts + RVC** ← 현재 방식 | 30초~수 분 샘플로 내 목소리 근사 가능, 발음 품질은 MS TTS가 보장 | 목소리가 완전히 같지는 않음 |

### 핵심 이유

> **"발음 품질은 MS TTS에 맡기고, 목소리 색깔(음색)만 RVC로 덮어씌운다"**

- **edge-tts**: Microsoft Azure Neural TTS를 무료로 사용. 자연스러운 한국어 발음, 억양, 속도 조절 지원.
- **RVC**: 짧은 샘플(30초~몇 분)로도 음색 변환 모델 학습 가능. 발음 정보는 건드리지 않고 "목소리 특성"만 바꿈.

---

## RVC 원리

**RVC (Retrieval-based Voice Conversion)**는 *발음/내용은 유지하면서 목소리 톤만 교체*하는 음성 변환 기술입니다.

### 핵심 아이디어: 내용 ↔ 음색 분리

음성 신호는 두 가지 정보로 분해할 수 있습니다:

```
음성 = 언어 내용(무슨 말인지) + 목소리 특성(누구 목소리인지)
       └─ HuBERT로 추출 ──────┘   └─ 피치(F0) + 음색으로 표현 ─┘
```

RVC는 이 두 가지를 분리해서, 언어 내용은 유지하고 목소리 특성만 교체합니다.

### 학습 파이프라인 (5단계)

```
원본 음성 샘플
      │
      ▼
1. 전처리 (preprocess.py)
   - 무음 제거, 세그먼트 분할
   - 40kHz 모노 WAV로 표준화
      │
      ▼
2. F0 추출 (extract_f0_*.py)
   - RMVPE 알고리즘으로 각 프레임의 피치(기본 주파수) 추출
   - "이 사람이 얼마나 높은/낮은 음으로 말하는가"를 숫자화
      │
      ▼
3. HuBERT 특징 추출 (extract_feature_print.py)
   - Meta의 HuBERT 모델로 768차원 음성 특징 벡터 추출
   - 언어적 내용(음소, 발화 패턴)을 인코딩 — 화자 정보는 제거됨
      │
      ▼
4. 학습 (train.py)
   - Generator + Discriminator 구조 (GAN 기반)
   - 입력: HuBERT 특징(내용) + F0(피치) → 출력: 학습한 목소리로 재합성
   - Pretrained 모델(f0G40k.pth, f0D40k.pth)에서 파인튜닝
      │
      ▼
5. 결과물
   - voice.pth : 음성 변환 Generator 모델
   - voice.index: HuBERT 특징 검색 인덱스 (FAISS)
                  → 추론 시 학습 데이터의 특징과 유사한 것을 찾아 품질 향상
```

### 추론(Inference) 원리

```
edge-tts 생성 음성
      │
      ├─ RMVPE → 피치(F0) 추출 → (f0_key 값으로 조절 가능)
      │
      └─ HuBERT → 언어 특징 벡터 추출
                    │
                    └─ voice.index에서 내 목소리 특징과 가장 유사한 벡터 검색(Retrieval)
                          │
                          ▼
                    Generator(voice.pth)
                    → 내 목소리 특성으로 음성 재합성
                          │
                          ▼
                    내 목소리 톤으로 변환된 WAV
```

### RMVPE란?

F0(피치) 추출 알고리즘. 기존 parselmouth/dio보다 노이즈에 강하고 정확도가 높아 RVC에서 기본값으로 사용.

### FAISS Index (.index 파일)이란?

학습 데이터의 HuBERT 특징 벡터를 저장한 검색 인덱스. 추론 시 입력 음성의 특징 벡터와 가장 유사한 학습 샘플을 찾아(Retrieval) 변환에 활용. 없어도 동작하지만 있으면 목소리 유사도가 향상됨.

---

## 현재 시스템 구조

### 전체 흐름

```
[브라우저 UI]
     │
     │  POST /generate (text, voice, speed, use_rvc, f0_key)
     ▼
[FastAPI 서버 — main.py]
     │
     ├─ 1. edge_tts_generate()
     │      edge-tts → MP3 스트림
     │      → ffmpeg으로 WAV 24kHz 변환
     │      → numpy float32 배열 반환
     │
     ├─ 2. rvc_convert()  ← use_rvc=True이고 voice.pth 존재할 때만
     │      임시 WAV 파일 저장
     │      → rvc_python.RVCInference.infer_file()
     │      → 변환된 WAV 읽어서 numpy 반환
     │
     └─ 3. soundfile.write() → outputs/{uuid}.wav
            → { "output": "uuid.wav" } 응답
```

### 학습 흐름

```
[브라우저 UI — RVC 학습 탭]
     │
     │  POST /train (voice_files: JSON 배열)
     ▼
[FastAPI BackgroundTasks → _run_training()]
     │
     ├─ 1. ffmpeg으로 40kHz WAV 변환 → logs/myvoice/trainset/
     ├─ 2. _ensure_pretrained_models() — HuBERT, f0G40k, f0D40k, rmvpe 다운로드
     ├─ 3. preprocess.py — 무음 제거·세그먼트 분할
     ├─ 4. extract_f0_print.py — RMVPE 피치 추출
     ├─ 5. extract_feature_print.py — HuBERT 768차원 특징 추출
     ├─ 6. filelist.txt 생성 (gt_wav|feature|f0|f0nsf|speaker_id)
     ├─ 7. train.py — 200 에폭 학습 (returncode 77 = 정상 완료)
     └─ 8. 결과 복사: assets/weights/myvoice*.pth → models/voice.pth
                      logs/myvoice/added_*.index → models/voice.index
```

### API 엔드포인트 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/generate` | TTS → (RVC) → WAV 생성 |
| `POST` | `/train` | 백그라운드 RVC 학습 시작 |
| `GET`  | `/model-status` | 모델 존재 여부 + 학습 상태 폴링 |
| `POST` | `/upload-voice` | 학습용 음성 파일 업로드 |
| `POST` | `/upload-model` | 외부 학습된 .pth/.index 업로드 |
| `GET`  | `/voices` | 업로드된 음성 파일 목록 |
| `GET`  | `/download/{filename}` | 생성된 WAV 다운로드 |

### 디렉터리 구조

```
test_tts/
├── main.py              # FastAPI 서버 + 학습 로직
├── Dockerfile           # PyTorch 2.3.1 + CUDA 12.1 + RVC WebUI 클론
├── docker-compose.yml   # GPU 1개, voices/outputs/models 볼륨 마운트
├── requirements.txt     # fastapi, edge-tts, rvc-python, soundfile, numpy
├── static/index.html    # 단일 페이지 UI (생성/학습/모델업로드 탭)
├── voices/              # 업로드된 학습용 음성 샘플
├── models/              # voice.pth + voice.index (학습 결과물)
├── outputs/             # 생성된 TTS 결과 WAV
└── rvc-train/           # 멀티-GPU 학습용 별도 Docker 환경
    ├── train.sh         # 4-GPU 병렬 학습 스크립트
    └── Dockerfile
```

### 주요 설계 결정

**`edge-tts` 선택 이유**: Microsoft Azure Neural TTS를 API 키 없이 사용 가능. 8가지 한국어 목소리, 속도 조절 지원. MP3로 오고 ffmpeg으로 WAV 변환.

**`rvc-python` 사용**: RVC WebUI의 inference 부분을 패키지화한 라이브러리. `RVCInference.infer_file()`로 간단하게 음성 변환 호출 가능.

**학습은 RVC WebUI 소스 직접 호출**: `rvc-python`은 inference만 지원하므로, 학습은 `/rvc`에 클론된 RVC WebUI의 Python 스크립트를 직접 `subprocess`로 실행.

**`train.py`의 returncode=77**: RVC WebUI의 `train.py`는 학습 완료 후 `os._exit(2333333)`을 호출하는데 이게 OS 레벨에서 77로 잘림. 오류가 아닌 정상 완료 신호로 처리.

**샘플레이트**: 학습은 40kHz (RVC 기본), 추론 출력은 24kHz (edge-tts와 통일).

---

## 실행 방법

```bash
# GPU 환경에서 실행
docker compose up --build

# 브라우저에서 접속
open http://localhost:8000
```

**학습 최소 요건**: 목소리 샘플 총 30초 이상, NVIDIA GPU 권장 (CPU도 동작하나 매우 느림).
