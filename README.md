# Mobility baselines

**Baselines and benchmarks for spatio-temporal forecasting problems in transportation**

This repository contains source code for reproducing the results in:

* [Rodrigues, F. On the importance of stationarity, strong baselines and benchmarks in transport prediction problems. In ArXiv, 2022](https://arxiv.org/abs/)

It consists of code for computing the "HA" and "HA+LR" baselines described in the paper, as well as code for preparing the experimental setups (i.e., train/val/test splits, forecasting horizons, evaluation metrics, etc.) for the following 9 publicly-available datasets:

| Name                        | Type                 | Timespan                | Time granularity | Train/val/test split | Source                                 |
|-----------------------------|----------------------|-------------------------|------------------|----------------------|----------------------------------------|
| PeMSD7(M) - California      | traffic speeds       | 01/04/2016 - 30/06/2016 | 5 minutes        | 34/5/5 days          | Yu et al., 2018     |
| Urban1 - South Korea        | traffic speeds       | 01/04/2018 - 30/04/2018 | 5 minutes        | 70/10/20 \%          | Lee and Rhee, 2022    |
| NYC Citi Bike - New York    | pickups and dropoffs | 01/04/2016 - 01/04/2016 | 30 minutes       | 63/14/14 days        | Ye et al., 2021    |
| PeMSD4 - California         | traffic volumes      | 01/01/2018 - 28/02/2018 | 5 minutes        | 60/20/20 \%          | Choi et al., 2022  |
| SZ-taxi - Shenzhen          | traffic speeds       | 01/01/2015 - 31/01/2015 | 15 minutes       | 80/-/20 \%           | Zhao et al., 2021     |
| METR-LA - Los Angeles       | traffic speeds       | 01/03/2012 - 30/06/2012 | 5 minutes        | 70/10/20 \%          | Li et al., 2018  |
| PEMS-BAY - California       | traffic speeds       | 01/01/2017 - 31/05/2017 | 5 minutes        | 70/10/20 \%          | Li et al., 2018  |
| NYC Citi Bike - New York    | in- and out-flows    | 01/07/2017 - 30/09/2017 | 1 hour           | 80/10/10 \%          | Xia et al., 2021    |
| Seattle loop data - Seattle | traffic speeds       | 01/11/2015 - 31/12/2015 | 5 minutes        | 56/-/5 days          | Yang et al., 2021   |

The goal is to facilitate the comparison between different spatio-temporal forecasting approaches by providing multiple well-defined reference benchmarks. 

The repo does not include the actual data, but the table below provides links to where the data can be downloaded: 

| Name                        | Download link                 |
|-----------------------------|----------------------|
| PeMSD7(M) - California      | [Link](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/data_loader) ([Mirror](https://mega.nz/file/IR4SCaAY#L22swMzsea5O-EuD_KQf6kuAu5pNkit_9p07qFXQ80U))      |
| Urban1 - South Korea        | [Link](https://github.com/snu-adsl/DDP-GCN/tree/main/dataset) ([Mirror](https://mega.nz/file/gVAlxCTb#wI_29erVJlstayKcLKdAj9p0gdYTxhcbrCc509w-Qbs))       |
| NYC Citi Bike - New York    | [Link](https://github.com/Essaim/CGCDemandPrediction/tree/main/data) ([Mirror](https://mega.nz/file/ZcZVHAKa#YxfaSlsIKsxjLWtTmbUOEaonCK7HmqKmuiCpnP06D1E)) |
| PeMSD4 - California         | [Link](https://github.com/jeongwhanchoi/STG-NCDE/tree/main/data) ([Mirror](https://mega.nz/file/oExDgC6Z#2NdQ28ogIYXc5wAeVTclG2DQmW91FrgOGhNQ39kmlPE))      |
| SZ-taxi - Shenzhen          | [Link](https://github.com/lehaifeng/T-GCN/tree/master/data) ([Mirror](https://mega.nz/file/RF43ka4R#u2DgbzTQUTj1Ya_ON95vQn6QA9dmpNLrsDwkKLyV8RQ))       |
| METR-LA - Los Angeles       | [Link](https://github.com/liyaguang/DCRNN) ([Mirror](https://mega.nz/file/lUgFmKIL#YnrZoY7xy2XoYZ_cpfe-F-1WMXUIOS0d8-nCq4KuBfY))      |
| PEMS-BAY - California       | [Link](https://github.com/liyaguang/DCRNN) ([Mirror](https://mega.nz/file/dN5VQaob#m9E9WQbgtwYFIWveEmFQPI8I9Z_spBJkZW7LT2GGuGE))      |
| NYC Citi Bike - New York    | [Link](https://github.com/FIBLAB/3D-DGCN/tree/master/flow) ([Mirror](https://mega.nz/file/dMo1mSQA#op2C4Rjp7x5UifsEEj8_1LmlSV-6iSK8Qhv7SpLPqm0))    |
| Seattle loop data - Seattle | [Link](https://github.com/Vadermit/TransPAI/tree/master/datasets/Seattle_loop-data-set) ([Mirror](https://mega.nz/file/0NZHgILC#Y5f7XBrQkAgTguZaLNWNGjBN5Z_uMSZjPfz8TFapObw))       |


