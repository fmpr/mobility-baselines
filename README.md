# Mobility baselines

**Baselines and benchmarks for spatio-temporal forecasting problems in transportation**

This repository contains source code for reproducing the results in:

* [Rodrigues, F. On the importance of stationarity, strong baselines and benchmarks in transport prediction problems. In ArXiv, 2022](https://arxiv.org/abs/)

It consists of code for computing the "HA" and "HA+LR" baselines described in the paper, as well as code for preparing the experimental setups (i.e., train/val/test splits, forecasting horizons, evaluation metrics, etc.) for the following 9 publicly-available datasets:

| Name                        | Type                 | Timespan                | Time granularity | Train/val/test split | Source                                 |
|-----------------------------|----------------------|-------------------------|------------------|----------------------|----------------------------------------|
| PeMSD7(M) - California      | traffic speeds       | 01/04/2016 - 30/06/2016 | 5 minutes        | 34/5/5 days          | Yu et al., 2018 [1]    |
| Urban1 - South Korea        | traffic speeds       | 01/04/2018 - 30/04/2018 | 5 minutes        | 70/10/20 \%          | Lee and Rhee, 2022 [2]   |
| NYC Citi Bike - New York    | pickups and dropoffs | 01/04/2016 - 01/04/2016 | 30 minutes       | 63/14/14 days        | Ye et al., 2021 [3]   |
| PeMSD4 - California         | traffic volumes      | 01/01/2018 - 28/02/2018 | 5 minutes        | 60/20/20 \%          | Choi et al., 2022 [4] |
| SZ-taxi - Shenzhen          | traffic speeds       | 01/01/2015 - 31/01/2015 | 15 minutes       | 80/-/20 \%           | Zhao et al., 2021 [5] |
| METR-LA - Los Angeles       | traffic speeds       | 01/03/2012 - 30/06/2012 | 5 minutes        | 70/10/20 \%          | Li et al., 2018 [6] |
| PEMS-BAY - California       | traffic speeds       | 01/01/2017 - 31/05/2017 | 5 minutes        | 70/10/20 \%          | Li et al., 2018 [6] |
| NYC Citi Bike - New York    | in- and out-flows    | 01/07/2017 - 30/09/2017 | 1 hour           | 80/10/10 \%          | Xia et al., 2021 [7]   |
| Seattle loop data - Seattle | traffic speeds       | 01/11/2015 - 31/12/2015 | 5 minutes        | 56/-/5 days          | Yang et al., 2021 [8]  |

The goal is to facilitate the comparison between different spatio-temporal forecasting approaches by providing multiple well-defined reference benchmarks. 

The repo does not include the actual data, but the table below provides links to where the data can be downloaded: 

| Name                        | Download link                 | Jupyter notebook with baseline                 | 
|-----------------------------|----------------------|----------------------|
| PeMSD7(M) - California      | [Link](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/data_loader) ([Mirror](https://mega.nz/file/IR4SCaAY#L22swMzsea5O-EuD_KQf6kuAu5pNkit_9p07qFXQ80U))      | [STGCN - Historical average baselines-FINAL.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/STGCN_IJCAI-18-master/STGCN%20-%20Historical%20average%20baselines-FINAL.ipynb) |
| Urban1 - South Korea        | [Link](https://github.com/snu-adsl/DDP-GCN/tree/main/dataset) ([Mirror](https://mega.nz/file/gVAlxCTb#wI_29erVJlstayKcLKdAj9p0gdYTxhcbrCc509w-Qbs))       | [DDP-GCN - Historical average baselines-FINAL.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/DDP-GCN-main/DDP-GCN%20-%20Historical%20average%20baselines-FINAL.ipynb) |
| NYC Citi Bike - New York    | [Link](https://github.com/Essaim/CGCDemandPrediction/tree/main/data) ([Mirror](https://mega.nz/file/ZcZVHAKa#YxfaSlsIKsxjLWtTmbUOEaonCK7HmqKmuiCpnP06D1E)) | [CGCDemand - Historical average baselines-seq2seq.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/CGCDemandPrediction-main/CGCDemand%20-%20Historical%20average%20baselines-seq2seq.ipynb) |
| PeMSD4 - California         | [Link](https://github.com/jeongwhanchoi/STG-NCDE/tree/main/data) ([Mirror](https://mega.nz/file/oExDgC6Z#2NdQ28ogIYXc5wAeVTclG2DQmW91FrgOGhNQ39kmlPE))      | [STG-NCDE - Historical average baselines-FINAL-seq2seq.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/STG-NCDE-main/STG-NCDE%20-%20Historical%20average%20baselines-FINAL-seq2seq.ipynb) |
| SZ-taxi - Shenzhen          | [Link](https://github.com/lehaifeng/T-GCN/tree/master/data) ([Mirror](https://mega.nz/file/RF43ka4R#u2DgbzTQUTj1Ya_ON95vQn6QA9dmpNLrsDwkKLyV8RQ))       | [T-GCN - Mobility baselines.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/T-GCN-master/T-GCN%20-%20Mobility%20baselines.ipynb) |
| METR-LA - Los Angeles       | [Link](https://github.com/liyaguang/DCRNN) ([Mirror](https://mega.nz/file/lUgFmKIL#YnrZoY7xy2XoYZ_cpfe-F-1WMXUIOS0d8-nCq4KuBfY))      | [DCRNN - Mobility baselines - METR-LA.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/DCRNN-master/DCRNN%20-%20Mobility%20baselines%20-%20METR-LA.ipynb) |
| PEMS-BAY - California       | [Link](https://github.com/liyaguang/DCRNN) ([Mirror](https://mega.nz/file/dN5VQaob#m9E9WQbgtwYFIWveEmFQPI8I9Z_spBJkZW7LT2GGuGE))      | [DCRNN - Mobility baselines - PEMS-BAY.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/DCRNN-master/DCRNN%20-%20Mobility%20baselines%20-%20PEMS-BAY.ipynb) | 
| NYC Citi Bike - New York    | [Link](https://github.com/FIBLAB/3D-DGCN/tree/master/flow) ([Mirror](https://mega.nz/file/dMo1mSQA#op2C4Rjp7x5UifsEEj8_1LmlSV-6iSK8Qhv7SpLPqm0))    | [3D-DGCN - Mobility baselines-FINAL.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/3D-DGCN-master/3D-DGCN%20-%20Mobility%20baselines-FINAL.ipynb) |
| Seattle loop data - Seattle |  [Link](https://github.com/Vadermit/TransPAI/tree/master/datasets/Seattle_loop-data-set) ([Mirror](https://mega.nz/file/0NZHgILC#Y5f7XBrQkAgTguZaLNWNGjBN5Z_uMSZjPfz8TFapObw))       | [TransPAI - Mobility baselines.ipynb](https://github.com/fmpr/mobility-baselines/blob/main/TransPAI-master/Experiments/TransPAI%20-%20Mobility%20baselines.ipynb) |

The file [mobility_detrender.py](https://github.com/fmpr/mobility-baselines/blob/main/mobility_detrender.py) contains the code for computing the historical averages (HA).

References:
* [1] B. Yu, H. Yin, and Z. Zhu, ???Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,??? in 27th International Joint Conference on Artificial Intelligence (IJCAI-18), 2018.
* [2] K. Lee and W. Rhee, ???DDP-GCN: Multi-graph convolutional network for spatiotemporal traffic forecasting,??? Transportation Research Part C: Emerging Technologies, vol. 134, p. 103466, 2022.
* [3] J. Ye, L. Sun, B. Du, Y. Fu, and H. Xiong, ???Coupled layer-wise graph convolution for transportation demand prediction,??? in Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21), 2021.
* [4] J. Choi, H. Choi, J. Hwang, and N. Park, ???Graph neural controlled differential equations for traffic forecasting,??? in Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22), 2022.
* [5] L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, and H. Li, ???T-GCN: A temporal graph convolutional network for traffic prediction,??? IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 9, pp. 3848???3858, 2019.
* [6] Y. Li, R. Yu, C. Shahabi, and Y. Liu, ???Diffusion convolutional recurrent neural network: Data-driven traffic forecasting,??? in International Conference on Learning Representations (ICLR-18), 2018.
* [7] T. Xia, J. Lin, Y. Li, J. Feng, P. Hui, F. Sun, D. Guo, and D. Jin, ???3DGCN: 3-dimensional dynamic graph convolutional network for citywide crowd flow prediction,??? ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 15, no. 6, pp. 1???21, 2021.
* [8] J.-M. Yang, Z.-R. Peng, and L. Lin, ???Real-time spatiotemporal prediction and imputation of traffic status based on lstm and graph laplacian regularized matrix factorization,??? Transportation Research Part C: Emerging Technologies, vol. 129, p. 103228, 2021.
