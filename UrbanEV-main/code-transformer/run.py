# Adapted file and directory from https://github.com/IntelligentSystemsLab/UrbanEV

import argparse
import os

import pandas as pd
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

NODE_NUM = 275
if __name__ == '__main__':
    fix_seed = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimeXer')

    # basic config list
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='TimeXer',
                        help='model name, options: [TimeXer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/UrbanEV/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='occ-None.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='../checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # optimization
    parser.add_argument('--epoch', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # Augmentation
    parser.add_argument('--seed', type=int, default=42, help="Randomization seed")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')

    #exp_setting
    parser.add_argument('--pred_len', type=int, default=3, help='prediction sequence length')
    parser.add_argument('--pred_type', type=str, default='region', help="Type of prediction: node-level or region-level")
    parser.add_argument('--add_feat', type=str, default=None, help="Whether to use additional features for prediction")
    parser.add_argument('--fold', type=int, default=0, help="The fold number of data to train with")
    parser.add_argument('--feat', type=str, default='occ', help="The type of data to use for prediction")



    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())
    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    print(
        f"Running {args.model} with feat={args.feat}, pre_l={args.pred_len}, fold={args.fold}, add_feat"
        f"={args.add_feat}, "
        f"pred_type(node)={args.pred_type}")
    if args.add_feat != 'None':
        args.data_path = f'{args.feat}-{args.add_feat}.csv'
        add_num = pd.read_csv(args.root_path + args.data_path).iloc[:, 1 + NODE_NUM:].shape[1]
        args.enc_in = NODE_NUM + add_num
        args.dec_in = NODE_NUM + add_num
        args.c_out = NODE_NUM + add_num
    else:
        if args.pred_type == 'region':
            args.enc_in = NODE_NUM
            args.dec_in = NODE_NUM
            args.c_out = NODE_NUM
            args.data_path = f'{args.feat}-{args.add_feat}.csv'
        else:
            args.enc_in = 1
            args.dec_in = 1
            args.c_out = 1
            args.data_path = f'{args.feat}-{args.add_feat}_node-{args.pred_type}.csv'

    if args.is_training:
        exp = Exp(args)
        setting =  (args.model +'_'+ 'feat-' + args.feat +  '_'  + 'pred_len-'+ str(args.pred_len) +  '_'  + 'fold-'+ str(
            args.fold) +
                    '_'   +  'node-'+ str(args.pred_type) + '_' + 'add_feat-' + str(args.add_feat) + '_' + 'epoch-' +
                    str(args.epoch))# set experiments

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        predictions = exp.test(args, setting)
        
        # After training, save predictions
        results_dir = 'result/main_exp/region'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'results_{args.feat}.csv')
        # if predictions is not None:
            # pd.DataFrame(predictions).to_csv(results_file, index=False)
        
        torch.cuda.empty_cache()
    else:
            setting = (args.model +'_'+ 'feat-' + args.feat +  '_'  + 'pred_len-'+ str(args.pred_len) +  '_'  + 'fold-'+ str(
            args.fold) +
                    '_'   +  'node-'+ str(args.pred_type) + '_' + 'add_feat-' + str(args.add_feat) + '_' + 'epoch-' +
                    str(args.epoch))# set experiments

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            predictions = exp.test(args, setting, test=1)
            
            # After testing, save predictions
            results_dir = 'result/main_exp/region'
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f'results_{args.feat}.csv')
            # if predictions is not None:
                # pd.DataFrame(predictions).to_csv(results_file, index=False)
            
            torch.cuda.empty_cache()
