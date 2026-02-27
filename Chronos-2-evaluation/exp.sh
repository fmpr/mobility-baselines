#!/bin/bash

# Define the Python executable (change to your virtual environment's python)
PYTHON_EXEC=".venv/bin/python"

# Sanity Check: Print which python is actually running
echo "Using Python at: $PYTHON_EXEC"

# Define the datasets to run
datasets=('PeMSD7\(M\)' 'Urban1' 'NYC_Citi_Bike_pick-drop' 'PeMSD4' 'SZ-taxi' 'METR-LA' 'PEMS-BAY' 'NYC_Citi_Bike_in-out' 'Seattle_loop_data')

for d in "${datasets[@]}"; do
    echo "----------------------------------------"
    echo "Running: Dataset $d"
    echo "----------------------------------------"
    
    $PYTHON_EXEC main.py dataset_cfg="dataset/$d"
done

# --- UrbanEV experiments ---
cd UrbanEV || exit
pred_lens=(3 6 9 12)
folds=(1 2 3 4 5 6)
features=(duration volume occ)

echo "Running UrbanEV experiments..."

for fe in "${features[@]}"; do
    for l in "${pred_lens[@]}"; do
        for f in "${folds[@]}"; do
            for w in 168; do
                echo "------------------------------------------------------"
                echo "Running: Feature $fe | Horizon $l | Fold $f"
                echo "------------------------------------------------------"
                
                # Note: Chronos-2 will skip training and go straight to inference/testing
                $PY_EXEC main.py --model "chronos-2" --feat "$fe" --pred_len "$l" --fold "$f" --pred_type "region" --context_window $w
            done
        done
    done
done

echo "✅ All experiments completed."