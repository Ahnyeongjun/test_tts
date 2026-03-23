#!/bin/bash
# RVC pretrained 모델 다운로드
cd /app/rvc

mkdir -p assets/pretrained_v2 assets/hubert assets/rmvpe

# hubert base
curl -L -o assets/hubert/hubert_base.pt \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

# pretrained v2 모델
for f in f0D40k.pth f0G40k.pth; do
  curl -L -o assets/pretrained_v2/$f \
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/$f"
done

# rmvpe 모델
curl -L -o assets/rmvpe/rmvpe.pt \
  "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

echo "모델 다운로드 완료"
