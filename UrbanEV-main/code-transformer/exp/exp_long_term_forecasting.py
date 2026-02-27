from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd # Fixed import name

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _calculate_loss(self, outputs, batch_y, criterion):
        """
        Helper to calculate either Quantile Loss (if 3 channels) or MSE (if 1 channel).
        """
        # 1. Check for Quantiles (4 dims: [Batch, Seq, Node, 3])
        if outputs.dim() == 4 and outputs.shape[-1] == 3:
            # Reshape target to [Batch, Seq, Node, 1] for broadcasting
            target = batch_y.unsqueeze(-1)
            
            p10 = outputs[..., 0].unsqueeze(-1)
            p50 = outputs[..., 1].unsqueeze(-1)
            p90 = outputs[..., 2].unsqueeze(-1)
            
            # Pinball Loss: max( (q-1)*err, q*err )
            err10 = target - p10
            loss10 = torch.mean(torch.max((0.1 - 1) * err10, 0.1 * err10))
            
            err50 = target - p50
            loss50 = torch.mean(torch.max((0.5 - 1) * err50, 0.5 * err50))
            
            err90 = target - p90
            loss90 = torch.mean(torch.max((0.9 - 1) * err90, 0.9 * err90))
            
            return loss10 + loss50 + loss90
        else:
            # 2. Standard MSE
            return criterion(outputs, batch_y)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = 275 if self.args.features == 'M' else 1
                
                # --- FIX: Handle slicing for 3-channel outputs ---
                if outputs.dim() == 4 and outputs.shape[-1] == 3:
                    outputs = outputs[:, -self.args.pred_len:, :f_dim, :]
                else:
                    outputs = outputs[:, -self.args.pred_len:, :f_dim]
                
                batch_y = batch_y[:, -self.args.pred_len:, :f_dim].to(self.device)

                # Calculate Loss (Using Helper to support Quantiles)
                loss = self._calculate_loss(outputs, batch_y, criterion)

                total_loss.append(loss.item()) # Use .item() to save memory
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epoch):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = 275 if self.args.features == 'M' else 1
                        
                        # --- FIX: Slicing for Quantiles ---
                        if outputs.dim() == 4 and outputs.shape[-1] == 3:
                            outputs = outputs[:, -self.args.pred_len:, :f_dim, :]
                        else:
                            outputs = outputs[:, -self.args.pred_len:, :f_dim]
                            
                        batch_y = batch_y[:, -self.args.pred_len:, :f_dim].to(self.device)

                        loss = self._calculate_loss(outputs, batch_y, criterion)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = 275 if self.args.features == 'M' else 1
                    
                    # --- FIX: Slicing for Quantiles ---
                    if outputs.dim() == 4 and outputs.shape[-1] == 3:
                        outputs = outputs[:, -self.args.pred_len:, :f_dim, :]
                    else:
                        outputs = outputs[:, -self.args.pred_len:, :f_dim]
                        
                    batch_y = batch_y[:, -self.args.pred_len:, :f_dim].to(self.device)

                    loss = self._calculate_loss(outputs, batch_y, criterion)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epoch - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, args, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('../checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = 275 if self.args.features == 'M' else 1
                
                # --- FIX: Slicing ---
                if outputs.dim() == 4 and outputs.shape[-1] == 3:
                     outputs = outputs[:, -self.args.pred_len:, :f_dim, :]
                else:
                     outputs = outputs[:, -self.args.pred_len:, :f_dim]

                batch_y = batch_y[:, -self.args.pred_len:, :f_dim].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # --- FIX: Inverse Transform for Quantiles ---
                if test_data.scale and self.args.inverse:
                    if outputs.ndim == 4: # [Batch, Time, Node, 3]
                        # Inverse transform each quantile channel separately
                        for k in range(3):
                            shape = outputs[..., k].shape
                            temp = outputs[..., k].reshape(shape[0] * shape[1], -1)
                            temp = test_data.inverse_transform(temp)
                            outputs[..., k] = temp.reshape(shape)
                    else:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                    # True values (only 1 channel usually)
                    shape = batch_y.shape
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        
        # Save predictions
        pred_dir = f'../predictions_{args.model}/'
        os.makedirs(pred_dir, exist_ok=True)
        file_id = f"{args.model}_fold{args.fold}_len{args.pred_len}_{args.pred_type}"
        np.save(f"{pred_dir}/pred_{file_id}.npy", preds)
        np.save(f"{pred_dir}/true_{file_id}.npy", trues)
        print(f"Predictions saved to: {pred_dir}/pred_{file_id}.npy")

        # Save Results
        output_dir = '../result' + '/' + 'main_exp' + '/' + 'region'
        os.makedirs(output_dir, exist_ok=True)

        result_list = []
        output_no_noise = metric(preds, trues, args) # Returns list [MSE, RMSE, MAPE, RAE, MAE, (QLoss)]
        result_list.append(output_no_noise)
        
        # --- FIX: Dynamic Columns ---
        cols = ['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE']
        if len(output_no_noise) > 5:
            cols.append('QLoss')
            
        result_df = pd.DataFrame(result_list, columns=cols)
        result_df['model_name'] = args.model
        result_df['pre_len'] = args.pred_len
        result_df['fold'] = args.fold
        
        csv_file = output_dir + '/' + f'results.csv'
        if os.path.exists(csv_file):
            result_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='gbk')
        else:
            result_df.to_csv(csv_file, index=False, encoding='gbk')

        return preds