# Adapted file from https://github.com/IntelligentSystemsLab/UrbanEV

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go Spatio-temporal EV Charging Demand Prediction!")

    parser.add_argument('--device', type=int, default=0, help="CUDA.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--seq_len', type=int, default=12, help="The sequence length of input data.")
    parser.add_argument('--bs', type=int, default=32, help="The batch size of fine-tuning.")
    parser.add_argument('--epoch', type=int, default=20, help="The max epoch of the training process.")
    parser.add_argument('--total_fold', type=int, default=6, help="The fold used for spliting data in cross-validation")

    parser.add_argument('--model', type=str, default='gcn', 
                    help="The used model: ar, lo, arima, fcnn, lstm, gcn, gcnlstm, astgcn")
    parser.add_argument('--pred_len', type=int, default=1, help="The length of prediction interval.")
    parser.add_argument('--add_feat', type=str, default='None', help="Whether to use additional features for prediction")
    parser.add_argument('--fold', type=int, default=0, help="The current fold number for training data")
    parser.add_argument('--pred_type', type=str, default='region', help="Prediction at node or regional level")
    parser.add_argument('--feat', type=str, default='occ', help="Which feature to use for prediction")

    parser.add_argument('--is_train', action='store_true', default=True)
    # parser.add_argument('--is_train', action='store_true', default=False)

    return parser.parse_args()
