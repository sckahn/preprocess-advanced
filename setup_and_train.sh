#!/bin/bash
# TAIR 한글 진료비영수증 학습 셋업 + 실행 스크립트
# GPU 서버에서 실행: bash setup_and_train.sh
#
# 사전 준비:
#   1. 이 repo를 GPU 서버에 clone
#   2. CUDA + PyTorch 설치 확인
#
# 디렉토리 구조 (clone 후):
#   preprocess-advanced/
#   ├── tair_repo/              # TAIR 코드 (한글 패치 적용됨)
#   │   └── weights/            # 사전학습 가중치
#   ├── dataset/                # 학습 데이터 (5000장)
#   │   ├── images/
#   │   └── restoration_dataset.json
#   ├── dataset_val/            # 검증 데이터 (500장)
#   │   ├── HQ/
#   │   ├── LQ/
#   │   └── real_benchmark_dataset.json
#   ├── configs/
#   │   └── train_korean_stage1.yaml
#   └── setup_and_train.sh      # 이 파일

set -e

echo "============================================"
echo "  TAIR 한글 진료비영수증 학습 셋업"
echo "============================================"

# 1. 의존성 설치
echo ""
echo "[1/4] 의존성 설치..."
pip install -q accelerate omegaconf einops wandb pyiqa tqdm opencv-python-headless pillow

# TAIR 추가 의존성
cd tair_repo
pip install -q -r requirements.txt 2>/dev/null || true
cd ..

# 2. 가중치 확인
echo ""
echo "[2/4] 사전학습 가중치 확인..."
WEIGHTS_DIR="tair_repo/weights"
MISSING=0

for f in "sd2.1-base-zsnr-laionaes5.ckpt" "realesrgan_s4_swinir_100k.pth" "DiffBIR_v2.1.pt"; do
    if [ -f "$WEIGHTS_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f (없음!)"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "가중치 다운로드:"
    echo "  cd tair_repo && bash download_weights.sh"
    echo ""
    echo "또는 수동 다운로드 후 tair_repo/weights/에 배치"
    exit 1
fi

# 3. 데이터셋 확인
echo ""
echo "[3/4] 데이터셋 확인..."
if [ -d "dataset/images" ] && [ -f "dataset/restoration_dataset.json" ]; then
    N_IMGS=$(ls dataset/images/*.jpg 2>/dev/null | wc -l)
    echo "  ✓ 학습 데이터: ${N_IMGS}장"
else
    echo "  ✗ dataset/ 없음. 먼저 생성:"
    echo "    python3 generate_dataset.py --count 5000 --output dataset --mode train"
    exit 1
fi

# 4. GPU 확인
echo ""
echo "[4/4] GPU 확인..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  ✓ {name} ({mem:.0f}GB)')
else:
    print('  ✗ CUDA 사용 불가!')
    exit(1)
"

# 학습 시작
echo ""
echo "============================================"
echo "  Stage 1 학습 시작 (Image Restoration)"
echo "  batch_size=2, lr=5e-5, steps=50000"
echo "  체크포인트: checkpoints/korean_receipt/stage1/"
echo "============================================"
echo ""

cd tair_repo
accelerate launch --num_processes=1 --mixed_precision=fp16 \
    train.py --config ../configs/train_korean_stage1.yaml
