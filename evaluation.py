# @Time     : Feb. 11, 2022 15:15
# @Author   : Filipe Rodrigues
# @FileName : evaluation.py

import numpy as np

def MAE(true, pred, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE(true, pred, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

def MAPE(true, pred, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def evaluation(true, pred, mask_value=0):
    return np.array([MAPE(true, pred, mask_value=mask_value), MAE(true, pred), RMSE(true, pred)])