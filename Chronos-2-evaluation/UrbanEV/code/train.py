# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import os
import torch
import numpy as np
import utils
import pandas as pd

def test(args, test_loader, occ, net, scaler='None'):
    result_list = []
    
    if not args.stat_model:
        predict_list = np.zeros([1, occ.shape[1], 3])
    else:
        predict_list = np.zeros([1, occ.shape[1]])
        
    label_list = np.zeros([1, occ.shape[1]])
    
    if args.pred_type != 'region':
        predict_list = np.zeros([1,1])
        label_list = np.zeros([1,1])
    
    # Variable to hold raw samples (rollouts) if provided (e.g. by original Chronos)
    samples_data = None 

    if not args.stat_model and 'chronos-2' not in args.model:
        output_dir = '../checkpoints/'
        os.makedirs(output_dir, exist_ok=True)
        path = (output_dir + args.model + '_' +
                'feat-' + args.feat + '_' +
                'pred_len-' + str(args.pred_len) + '_' +
                'fold-' + str(args.fold) + '_' +
                'node-' + str(args.pred_type) + '_' +
                'add_feat-' + str(args.add_feat) + '_' +
                'epoch-' + str(args.epoch) + '.pth')
        state_dict = torch.load(path, weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        for j, data in enumerate(test_loader):
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data
            with torch.no_grad():
                predict = net(occupancy, extra_feat)
                
                predict = predict.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            predict_list = np.concatenate((predict_list, predict), axis=0)
            label_list = np.concatenate((label_list, label), axis=0)

    else:
        train_valid_occ, test_occ = test_loader
        
        output = net.predict(train_valid_occ, test_occ)
        if isinstance(output, tuple):
            predict, samples_data = output
        else:
            predict = output

        label = test_occ
        
        predict_list = np.zeros((1, *predict.shape[1:]))
        
        predict_list = np.concatenate((predict_list, predict), axis=0)
        label_list = np.concatenate((label_list, label), axis=0)

    # Save predictions
    save_dir = f'../predictions_{args.model}/'
    os.makedirs(save_dir, exist_ok=True)
    
    file_id = f"{args.model}_fold{args.fold}_len{args.pred_len}_{args.pred_type}"
    
    np.save(f"{save_dir}/pred_{file_id}.npy", predict_list[1:]) 
    np.save(f"{save_dir}/true_{file_id}.npy", label_list[1:])
    print(f"Predictions saved to: {save_dir}/pred_{file_id}.npy")

    # Save Samples (Rollouts) if they exist (for original Chronos)
    if samples_data is not None:
        np.save(f"{save_dir}/samples_{file_id}.npy", samples_data)
        print(f"Samples saved to: {save_dir}/samples_{file_id}.npy")

    if scaler != 'None':
        if predict_list.ndim == 3:
            for k in range(3):
                predict_list[:, :, k] = scaler.inverse_transform(predict_list[:, :, k])
        else:
            predict_list = scaler.inverse_transform(predict_list)
            
        label_list = scaler.inverse_transform(label_list)

    # Prepare Point Estimate (Median P50)
    if predict_list.ndim == 3:
        point_pred = predict_list[1:, :, 1] # Index 1 is Median/P50
    else:
        point_pred = predict_list[1:]

    # Calculate Point Metrics
    output_no_noise = utils.metrics(test_pre=point_pred, test_real=label_list[1:], args=args)
    
    # Calculate Quantile Loss if applicable
    if predict_list.ndim == 3:
        target = label_list[1:]
        
        err10 = target - predict_list[1:, :, 0]
        loss10 = np.maximum((0.1-1)*err10, 0.1*err10).mean()
        
        err50 = target - predict_list[1:, :, 1]
        loss50 = np.maximum((0.5-1)*err50, 0.5*err50).mean()
        
        err90 = target - predict_list[1:, :, 2]
        loss90 = np.maximum((0.9-1)*err90, 0.9*err90).mean()
        
        avg_q_loss = loss10 + loss50 + loss90
        output_no_noise.append(avg_q_loss)

        # Compute empirical coverage and IQR mean from quantile forecasts
        lower = predict_list[1:, :, 0]
        upper = predict_list[1:, :, 2]
        inside = (target >= lower) & (target <= upper)
        coverage = float(np.sum(inside)) / float(inside.size) if inside.size > 0 else float('nan')
        iqr_mean = float(np.mean(upper - lower)) if inside.size > 0 else float('nan')
        output_no_noise.insert(5, coverage)
        output_no_noise.insert(6, iqr_mean)

    # If no quantiles, ensure placeholders for COVERAGE and IQR_MEAN exist
    if predict_list.ndim != 3:
        output_no_noise.append(float('nan'))
        output_no_noise.append(float('nan'))

    result_list.append(output_no_noise)

    columns = ['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE']
    if len(output_no_noise) >= 6:
        columns.append('COVERAGE')
    if len(output_no_noise) >= 7:
        columns.append('IQR_MEAN')
    if len(output_no_noise) > 7:
        columns.append('QLoss')

    result_df = pd.DataFrame(result_list, columns=columns)
    result_df['model_name'] = args.model
    result_df['pred_len'] = args.pred_len
    result_df['fold'] = args.fold 

    output_dir = '../result' + '/' + 'main_exp' + '/' + 'region'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = output_dir + '/' + f'results_{args.feat}.csv'

    if os.path.exists(csv_file):
        try:
            existing = pd.read_csv(csv_file, encoding='gbk', on_bad_lines='skip')
            combined = pd.concat([existing, result_df], ignore_index=True, sort=False)
            combined.to_csv(csv_file, index=False, encoding='gbk')
        except Exception:
            result_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='gbk')
    else:
        result_df.to_csv(csv_file, index=False, encoding='gbk')