# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import utils
import torch
from parse import parse_args

import train
import numpy as np
from utils import split_cv, create_loaders
from baselines import QuantileLoss 

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

if __name__ == "__main__":

    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    utils.set_seed(seed=args.seed, flag=True)
    feat, adj, extra_feat, time = utils.read_data(args)
    print(
        f"Running {args.model} with feat={args.feat}, pre_l={args.pred_len}, fold={args.fold}, add_feat={args.add_feat}, pred_type(node)={args.pred_type}")

    # Initialize and train model
    net = utils.load_net(args, np.array(adj), device, feat)

    train_feat, valid_feat, test_feat, train_extra_feat, valid_extra_feat, test_extra_feat, scaler = split_cv(args,
                                                                                                              time,
                                                                                                              feat,
                                                                                                              TRAIN_RATIO,
                                                                                                              VAL_RATIO,
                                                                                                              TEST_RATIO,
                                                                                                              extra_feat)
    train_loader, valid_loader, test_loader = create_loaders(train_feat, valid_feat, test_feat,
                                                             train_extra_feat, valid_extra_feat,
                                                             test_extra_feat,
                                                             args, device)
    
    
    simple_models = ['lo', 'ar', 'arima']
    if args.model in simple_models:
        args.stat_model = True
    else:
        args.stat_model = False

    # --- Statistical & Foundation Models (No Training Loop) ---
    if args.stat_model:
        optim = None
        loss_func = None
        args.is_train = False
        args.stat_model = True
        # Concatenate history for statistical lookback
        train_valid_feat = np.vstack((train_feat, valid_feat, test_feat[:args.seq_len+args.pred_len, :]))
        test_loader = [train_valid_feat, test_feat[args.pred_len+args.seq_len:, :]]
    
    # --- Deep Learning Models (Training Loop) ---
    else:
        optim = torch.optim.Adam(net.parameters(), weight_decay=0.00001)
        args.stat_model = False
        
        loss_func = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        if args.is_train:
            train.training(args, net, optim, loss_func, train_loader, valid_loader, args.fold)

    train.test(args, test_loader, feat, net, scaler)