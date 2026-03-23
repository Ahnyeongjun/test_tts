#!/bin/bash
set -e

MODEL_NAME="${MODEL_NAME:-my_voice}"
AUDIO_FILE="${AUDIO_FILE:-/data/input.m4a}"
EPOCHS="${EPOCHS:-100}"
SAMPLE_RATE=40000
SR_STR=40k

cd /app/rvc

# matplotlib tostring_rgb -> buffer_rgba 패치
python - <<'PATCH'
import re, pathlib
f = pathlib.Path("infer/lib/train/utils.py")
txt = f.read_text()
old = '    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")'
new = ('    if hasattr(fig.canvas, "tostring_rgb"):\n'
       '        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")\n'
       '    else:\n'
       '        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)'
       '.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].flatten()')
if old in txt:
    f.write_text(txt.replace(old, new))
    print("패치 적용")
else:
    print("이미 패치됨")
PATCH

mkdir -p /app/rvc/logs/$MODEL_NAME /data/dataset/$MODEL_NAME

# config.json 생성 (없으면)
if [ ! -f /app/rvc/logs/$MODEL_NAME/config.json ]; then
  cp configs/v1/40k.json /app/rvc/logs/$MODEL_NAME/config.json
fi

echo "=== 1. 음성 파일 변환 ==="
ffmpeg -i "$AUDIO_FILE" -ar $SAMPLE_RATE -ac 1 /data/dataset/$MODEL_NAME/audio.wav -y

echo "=== 2. 전처리 ==="
python infer/modules/train/preprocess.py \
  /data/dataset/$MODEL_NAME $SAMPLE_RATE 2 \
  /app/rvc/logs/$MODEL_NAME False 3.0

echo "=== 3. F0 추출 (GPU 4개 병렬) ==="
python infer/modules/train/extract/extract_f0_rmvpe.py \
  4 0 0 /app/rvc/logs/$MODEL_NAME True $SAMPLE_RATE &
python infer/modules/train/extract/extract_f0_rmvpe.py \
  4 1 1 /app/rvc/logs/$MODEL_NAME True $SAMPLE_RATE &
python infer/modules/train/extract/extract_f0_rmvpe.py \
  4 2 2 /app/rvc/logs/$MODEL_NAME True $SAMPLE_RATE &
python infer/modules/train/extract/extract_f0_rmvpe.py \
  4 3 3 /app/rvc/logs/$MODEL_NAME True $SAMPLE_RATE &
wait

echo "=== 4. 특징 추출 (GPU 4개 병렬) ==="
python infer/modules/train/extract_feature_print.py \
  cuda:0 4 0 0 /app/rvc/logs/$MODEL_NAME v2 True &
python infer/modules/train/extract_feature_print.py \
  cuda:1 4 1 1 /app/rvc/logs/$MODEL_NAME v2 True &
python infer/modules/train/extract_feature_print.py \
  cuda:2 4 2 2 /app/rvc/logs/$MODEL_NAME v2 True &
python infer/modules/train/extract_feature_print.py \
  cuda:3 4 3 3 /app/rvc/logs/$MODEL_NAME v2 True &
wait

echo "=== 4.5. filelist.txt 생성 ==="
python - <<'PYEOF'
import os, random
exp_dir = "/app/rvc/logs/" + os.environ["MODEL_NAME"]
gt_dir  = exp_dir + "/0_gt_wavs"
feat_dir= exp_dir + "/3_feature768"
f0_dir  = exp_dir + "/2a_f0"
f0nsf   = exp_dir + "/2b-f0nsf"
names = (set(n.split(".")[0] for n in os.listdir(gt_dir))
       & set(n.split(".")[0] for n in os.listdir(feat_dir))
       & set(n.split(".")[0] for n in os.listdir(f0_dir))
       & set(n.split(".")[0] for n in os.listdir(f0nsf)))
opt = [f"{gt_dir}/{n}.wav|{feat_dir}/{n}.npy|{f0_dir}/{n}.wav.npy|{f0nsf}/{n}.wav.npy|0"
       for n in names]
# mute 샘플 추가
mute = "/app/rvc/logs/mute"
for _ in range(2):
    opt.append(f"{mute}/0_gt_wavs/mute40k.wav|{mute}/3_feature768/mute.npy|{mute}/2a_f0/mute.wav.npy|{mute}/2b-f0nsf/mute.wav.npy|0")
random.shuffle(opt)
with open(exp_dir + "/filelist.txt", "w") as f:
    f.write("\n".join(opt))
print(f"filelist.txt 생성 완료: {len(opt)}개")
PYEOF

echo "=== 5. 학습 (4 GPU) ==="
python infer/modules/train/train.py \
  -e $MODEL_NAME \
  -sr $SR_STR \
  -f0 1 \
  -bs 16 \
  -g 0 \
  -te $EPOCHS \
  -se 10 \
  -pg assets/pretrained_v2/f0G40k.pth \
  -pd assets/pretrained_v2/f0D40k.pth \
  -l 0 \
  -c 0 \
  -sw 0 \
  -v v2

echo "=== 학습 완료! ==="
echo "모델 위치: /app/rvc/logs/$MODEL_NAME/"
