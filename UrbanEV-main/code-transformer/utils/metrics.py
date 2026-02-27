import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
import os
import matplotlib.pyplot as plt

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def plot_and_save_node_predictions(preds, trues, save_dir,args):
    os.makedirs(save_dir, exist_ok=True)
    num_nodes = preds.shape[1]
    time_steps = preds.shape[0]

    for node in range(num_nodes):
        plt.figure(figsize=(10, 4))
        plt.plot(range(time_steps), trues[:, node], label='True Values', color='blue')
        plt.plot(range(time_steps), preds[:, node], label='Predicted Values', color='red')
        plt.title(f'Node {node} Predictions vs True Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()

        # Save figure
        output_dir = save_dir + '/' + 'fig' + '/' + f'{args.model}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir + '/' + f"node-{args.pred_type}_fold-{args.fold}_pred_len{args.pred_len}_predictions.png"  # 保存路径
        plt.savefig(output_path)
        plt.close()

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true, args):
    """
    Calculates metrics. Automatically handles 3-channel Quantile predictions.
    pred shape: (Batch, Seq_Len, Nodes) OR (Batch, Seq_Len, Nodes, 3)
    true shape: (Batch, Seq_Len, Nodes)
    """
    # 1. Slice to get the specific prediction step (last step defined by horizon)
    # Shapes become: (Batch, Nodes) OR (Batch, Nodes, 3)
    pred_step = pred[:, -1, :]
    true_step = true[:, -1, :]
    
    qloss = np.nan # Default if no quantiles exist

    # 2. Check for Quantiles (3 Channels: P10, P50, P90)
    if pred_step.ndim == 3 and pred_step.shape[-1] == 3:
        # Extract channels
        p10 = pred_step[..., 0]
        p50 = pred_step[..., 1] # Median
        p90 = pred_step[..., 2]
        
        target = true_step
        
        # --- QUANTILE LOSS CALCULATION ---
        # Formula: sum( max(q * err, (q-1) * err) )
        err10 = target - p10
        loss10 = np.maximum(0.1 * err10, (0.1 - 1) * err10).mean()
        
        err50 = target - p50
        loss50 = np.maximum(0.5 * err50, (0.5 - 1) * err50).mean()
        
        err90 = target - p90
        loss90 = np.maximum(0.9 * err90, (0.9 - 1) * err90).mean()
        
        qloss = loss10 + loss50 + loss90
        # ---------------------------------
        
        # Set pred_step to Median (P50) for the standard metrics below
        pred_step = p50

    # 3. Standard Metrics Calculation (using Median or Point Estimate)
    eps = 2e-2
    MAPE_true = true_step.copy()
    MAPE_pred = pred_step.copy()
    
    # Masking for stability
    mask = MAPE_true <= eps
    MAPE_true[mask] = np.abs(MAPE_true[mask]) + eps
    MAPE_pred[mask] = np.abs(MAPE_pred[mask]) + eps

    mape = mean_absolute_percentage_error(MAPE_true, MAPE_pred)
    mae = mean_absolute_error(true_step, pred_step)
    mse = mean_squared_error(true_step, pred_step)
    rmse = np.sqrt(mse)
    
    # RAE Calculation
    numerator = np.sum(np.abs(MAPE_pred - MAPE_true))
    denominator = np.sum(np.abs(MAPE_true - np.mean(MAPE_true)))
    rae = numerator / denominator

    # 4. Print & Return
    print(f'MAPE: {mape:.4f}')
    print(f'MAE:  {mae:.4f}')
    print(f'MSE:  {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'RAE:  {rae:.4f}')
    if not np.isnan(qloss):
        print(f'QLoss: {qloss:.4f}')

    # Returns list: [MSE, RMSE, MAPE, RAE, MAE, QLoss]
    output_list = [mse, rmse, mape, rae, mae, qloss]
    return output_list
