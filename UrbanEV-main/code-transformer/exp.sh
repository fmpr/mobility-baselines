#!/bin/bash
#BSUB -J ml_transport_forecast_transformers   # Job name
#BSUB -o ml_transport_forecast_transformers.out # Standard output
#BSUB -e ml_transport_forecast_transformers.err # Standard error
#BSUB -q gpuv100                 # Queue (GPU V100 nodes)
#BSUB -n 4                       # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # 1 GPU, exclusive
#BSUB -W 8:00                    # Wall‐clock time (hh:mm)
#BSUB -R "rusage[mem=20GB]"       # Memory requirement per host

# Define the absolute path to your environment's python
PY_EXEC="/zhome/44/2/213836/ml-transport-forecasting/UrbanEV_v1/bin/python"

# Sanity Check: Print which python is actually running
echo "Using Python at: $PY_EXEC"
$PY_EXEC -c "import torch_geometric_temporal; print('✅ Library found!')"

# 设置变量
pre_lens=(9 12)
folds=(1 2 3 4 5 6)
EPOCH=1
features=(volume)

# 运行 TimeXer 模型
for fe in "${features[@]}"; do
    for l in "${pre_lens[@]}"; do
        for f in "${folds[@]}"; do
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            python run.py \
                --seq_len 12 \
                --label_len 12 \
                --epoch $EPOCH \
                --model TimeXer \
                --pred_len $l \
                --fold $f \
                --batch_size 16 \
                --use_amp \
                --feat "$fe"
        done
    done
done

# 运行 TimesNet 模型
for fe in "${features[@]}"; do
    for l in "${pre_lens[@]}"; do
        for f in "${folds[@]}"; do
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            python run.py \
                --seq_len 12 \
                --label_len 12 \
                --epoch $EPOCH \
                --model TimesNet \
                --pred_len $l \
                --fold $f \
                --batch_size 16 \
                --use_amp \
                --feat "$fe"
        done
    done
done

echo "✅ All tasks completed!"
