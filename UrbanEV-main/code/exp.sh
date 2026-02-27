#!/bin/bash

# Define the absolute path to your environment's python
PY_EXEC="./UrbanEV_v1/bin/python"

echo "Using Python at: $PY_EXEC"
$PY_EXEC -c "import torch_geometric_temporal; print('Library found!')"

models=(lo ar arima fcnn lstm gcn gcnlstm astgcn)
pred_lens=(3 6 9 12)
folds=(1 2 3 4 5 6)
features=(duration volume)
EPOCH=20

# 嵌套循环执行实验
for fe in "${features[@]}"; do
    for m in "${models[@]}"; do
        for l in "${pred_lens[@]}"; do
            for f in "${folds[@]}"; do
                echo "Running: Model $m | Horizon $l | Fold $f"
                
                # Use $PY_EXEC instead of just 'python'
                $PY_EXEC main.py --model "$m" --pred_len "$l" --fold "$f" --epoch "$EPOCH" --feat "$fe"
            done
        done
    done
done

echo "✅ All experiments completed."