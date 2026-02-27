# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import utils
import torch
from parse import parse_args

import train
import numpy as np
from utils import split_cv, create_loaders, compute_metrics

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    utils.set_seed(seed=args.seed, flag=True)
    feat, adj, extra_feat, time = utils.read_data(args)
    print(
        f"Running {args.model} with feat={args.feat}, pre_l={args.pred_len}, fold={args.fold}, add_feat={args.add_feat}, pred_type(node)={args.pred_type}"
    )

    # Initialize and train model
    net = utils.load_net(args, np.array(adj), device, feat)

    (
        train_feat,
        valid_feat,
        test_feat,
        train_extra_feat,
        valid_extra_feat,
        test_extra_feat,
        scaler,
    ) = split_cv(args, time, feat, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, extra_feat)
    train_loader, valid_loader, test_loader = create_loaders(
        train_feat,
        valid_feat,
        test_feat,
        train_extra_feat,
        valid_extra_feat,
        test_extra_feat,
        args,
        device,
    )

    optim = None
    loss_func = None
    args.is_train = False
    args.stat_model = True
    train_valid_feat = np.vstack(
        (train_feat, valid_feat, test_feat[: args.seq_len + args.pred_len, :])
    )
    test_loader = [train_valid_feat, test_feat[args.pred_len + args.seq_len :, :]]

    # Using the zero-shot version of Chronos-2, so directly evaluate
    train.test(args, test_loader, feat, net, scaler)

    # Display results once we're done
    if args.pred_len == 12 and args.fold == 6:
        compute_metrics(f"../result/main_exp/region/results_{args.feat}.csv")