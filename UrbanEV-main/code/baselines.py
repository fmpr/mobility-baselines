# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import os

import torch
import torch.nn as nn
import numpy as np
from torch_geometric_temporal.nn.attention.astgcn import ASTGCN
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings

class Lo:
    def __init__(self, args):
        self.args = args
        self.pred_len = args.pred_len

    def predict(self, train_valid_occ, test_occ):
        """
        Use the latest observed value as the prediction for the next time step.
        """
        time_len, node = test_occ.shape
        preds = np.zeros((time_len, node))

        for j in range(node):
            for i in range(time_len):
                if i < self.pred_len:
                    preds[i, j] = train_valid_occ[-self.pred_len + i, j]
                else:
                    preds[i, j] = test_occ[i - self.pred_len, j]

        return preds


class Ar:
    def __init__(self, pred_len, args, lags=1):
        """
        Initialize the AR model parameters.

        Args:
            lags (int): The number of lagged observations to use in the model.
        """
        self.args = args
        self.pred_len = pred_len
        self.lags = lags

    def predict(self, train_valid_occ, test_occ):
        time_len, node = test_occ.shape
        train_valid_occ = train_valid_occ[:-self.pred_len, :]
        preds = np.zeros((time_len, node))

        for j in range(node):
            series = list(train_valid_occ[:, j])
            for i in range(time_len):
                try:
                    model = AutoReg(series, lags=self.lags).fit()
                    pred = model.predict(start=len(series), end=len(series))
                    preds[i, j] = pred[0]
                    series.append(pred[0])  # 滚动加入预测值
                except Exception as e:
                    print(f"[AutoReg] Failed at node {j}, step {i}: {e}")
                    preds[i, j] = np.nan

        return preds

class Arima:
    def __init__(self, pred_len, args, p=1, d=1, q=1):
        self.pred_len = pred_len
        self.args = args
        self.p = p
        self.d = d
        self.q = q

    def predict(self, train_valid_occ, test_occ):
        time_len, node = test_occ.shape
        train_valid_occ = train_valid_occ[:-self.pred_len, :]
        preds = np.zeros((time_len, node))

        warnings.filterwarnings("ignore")

        for j in range(node):
            fit_series = train_valid_occ[:, j]
            try:
                model = ARIMA(fit_series, order=(self.p, self.d, self.q))
                model_fitted = model.fit()
                # Predict future time steps (equal to time_len)
                pred = model_fitted.forecast(steps=time_len)
                preds[:, j] = pred
            except Exception as e:
                print(f"ARIMA failed for node {j}: {e}")
                preds[:, j] = np.nan  # or use zeros

        return preds


class Fcnn(nn.Module):
    def __init__(self, n_fea, node=247, seq=12):  # input_dim = seq_length
        super(Fcnn, self).__init__()
        self.num_feat = n_fea
        self.seq = seq
        self.nodes = node
        self.linear = nn.Linear(seq * n_fea, 3)

    def forward(self, occ, extra_feat='None'):
        x = occ # batch, nodes,seq for region or batch, seq for nodes
        if extra_feat != 'None':
            x = torch.cat([occ.unsqueeze(-1), extra_feat], dim=-1)
            assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."
        x = x.view(-1, self.nodes, self.seq * self.num_feat)
        x = self.linear(x)
        return x


class Lstm(nn.Module):
    def __init__(self, seq, n_fea, node=331):
        super(Lstm, self).__init__()
        self.num_feat = n_fea
        self.nodes = node
        self.seq_len = seq
        self.encoder = nn.Linear(n_fea, 1)# input.shape: [batch, channel, width, height]
        self.lstm_hidden_dim = 16
        self.lstm = nn.LSTM(input_size=n_fea, hidden_size=self.lstm_hidden_dim, num_layers=2,
                            batch_first=True)
        self.linear = nn.Linear(seq * self.lstm_hidden_dim, 3)

    def forward(self, occ, extra_feat=None):  # occ.shape = [batch, node, seq]
        x = occ.unsqueeze(-1)
        if extra_feat != 'None':
            x = torch.cat([occ.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."

        bs = x.shape[0]
        x = x.view(bs * self.nodes, self.seq_len, self.num_feat)

        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size * node, seq_len, lstm_hidden_dim]
        lstm_out = lstm_out.reshape(bs, self.nodes, self.seq_len * self.lstm_hidden_dim)
        x = self.linear(lstm_out)
        return x


class Gcn(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, gcn_hidden=32, gcn_layers=2):
        super(Gcn, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.gcn_hidden = gcn_hidden
        self.gcn_layers = gcn_layers
        self.num_feat = n_fea
        self.act = nn.ReLU()
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (1, n_fea))

        # Calculate A_delta matrix (normalized adjacency)
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

        # Define GCN layers
        self.gcn_layers_list = nn.ModuleList()
        self.gcn_layers_list.append(nn.Linear(seq, self.gcn_hidden))  # Input to first GCN layer
        for _ in range(self.gcn_layers - 1):
            self.gcn_layers_list.append(nn.Linear(self.gcn_hidden, self.gcn_hidden))

        self.decoder = nn.Linear(self.gcn_hidden + seq, 3)

    def forward(self, occ, extra_feat=None):
        x = occ.clone().unsqueeze(-1)  # Add feature dimension
        if extra_feat != 'None':
            x = torch.cat([occ.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."

        # GCN forward pass
        x = self.encoder(x)
        batch_size, node, seq_len = occ.size()
        gcn_out = x.view(batch_size, node, -1)  # Flatten sequence and features into [batch, node, seq * n_fea]

        for gcn_layer in self.gcn_layers_list:
            gcn_out = gcn_layer(gcn_out)
            gcn_out = torch.matmul(self.A, gcn_out)
            gcn_out = self.act(gcn_out)

        combined_out = torch.cat((occ, gcn_out), dim=-1)
        x = self.decoder(combined_out)
        return x


class Gcnlstm(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, node=307, gcn_out=32, gcn_layers=1, lstm_hidden_dim=256, lstm_layers=2,
                 hidden_dim=32):
        super(Gcnlstm, self).__init__()

        self.nodes = node
        self.seq_len = seq
        self.num_feat = n_fea
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.gcn_out = gcn_out
        self.gcn_layers = gcn_layers
        self.hidden_dim = hidden_dim

        # Initialize GCN layers
        self.gcn_layers_list = nn.ModuleList()
        for i in range(gcn_layers):
            in_dim = seq * n_fea if i == 0 else gcn_out
            self.gcn_layers_list.append(nn.Linear(in_dim, gcn_out))
        self.act = nn.ReLU()

        self.encoder = nn.Conv2d(self.nodes, self.nodes, (1, n_fea))
        # Initialize LSTM layer
        self.lstm = nn.LSTM(input_size=n_fea, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layers,
                            batch_first=True)
        self.decoder = nn.Linear(seq + self.gcn_out + self.lstm_hidden_dim, 3)
        # Calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, extra_feat=None):
        x = occ.clone().unsqueeze(-1)  # Add feature dimension
        if extra_feat != 'None':
            x = torch.cat([occ.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."
        # Create a copy of occ to avoid modifying the original data
        x = self.encoder(x)
        x_copy = x.clone().unsqueeze(-1)
        batch_size = x_copy.size(0)

        # Process all timesteps with LSTM
        x_lstm = x_copy.view(batch_size * self.nodes, self.seq_len, self.num_feat)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_out = lstm_out.view(batch_size, self.nodes, self.seq_len,
                                 self.lstm_hidden_dim)  # Shape: (batch, node, seq, lstm_hidden_dim)

        # Process with GCN layers
        gcn_out = x.view(batch_size,self.nodes,self.seq_len * self.num_feat)
        for gcn_layer in self.gcn_layers_list:
            gcn_out = gcn_layer(gcn_out)
            gcn_out = torch.matmul(self.A, gcn_out)
            gcn_out = self.act(gcn_out)

        # Concatenate LSTM and GCN outputs
        combined_out = torch.cat((occ,lstm_out[:,:,-1,:], gcn_out), dim=-1)  # Shape: (batch, node, seq, lstm_hidden_dim + gcn_out)

        x = self.decoder(combined_out)
        return x


class Astgcn(ASTGCN):
    def __init__(self, adj_dense,nb_block,in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict,len_input,num_of_vertices,node_idx=None):
        super(Astgcn, self).__init__(nb_block,in_channels, K, nb_chev_filter, nb_time_filter, time_strides, 3, len_input,num_of_vertices,normalization=None)
        
        # self.fc = nn.Linear(self.num_nodes, 1)
        # Preprocess the adjacency matrix to create edge_index and edge_weight
        self.node_idx = node_idx
        self.adj_dense = adj_dense
        self.num_feat = in_channels
        if self.node_idx is not None:
            filtered_adj = torch.zeros_like(adj_dense)
            filtered_adj[node_idx, :] = adj_dense[node_idx, :]
            filtered_adj[:, node_idx] = adj_dense[:, node_idx]
            self.adj_dense = filtered_adj
        self.edge_index = self.create_edge_index(adj_dense)
        self.edge_weight = adj_dense[adj_dense > 0]

    def create_edge_index(self, adj_dense):
        # Convert dense adjacency matrix to sparse edge index format
        edge_index = torch.nonzero(adj_dense, as_tuple=False).t().contiguous()
        return edge_index

    def forward(self, occ,extra_feat=None):
        x = occ.unsqueeze(-1)  # Add feature dimension
        if extra_feat != 'None':
            x = torch.cat([occ.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."

        x = super().forward(x.transpose(2,3),self.edge_index)
        return x

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        """
        preds: [Batch, Time, Nodes, 3] (one output per quantile)
        target: [Batch, Time, Nodes]
        """
        loss = 0
        # Expand target to match preds shape if needed, or broadcast
        # target shape becomes [Batch, Time, Nodes, 1]
        target = target.unsqueeze(-1)
        
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i:i+1] # Slice to keep dimension
            
            # Pinball Loss Formula: max((q-1)*error, q*error)
            loss += torch.max((q - 1) * errors, q * errors).mean()
            
        return loss