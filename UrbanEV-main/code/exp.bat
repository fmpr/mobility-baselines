@echo off

set models=ar lo arima fcnn lstm gcn gcnlstm astgcn
set pred_lens=3 6 9 12
set folds=1 2 3 4 5 6
set EPOCH=20

for %%m in (%models%) do (
    for %%l in (%pred_lens%) do (
        for %%f in (%folds%) do (
                python main.py --model %%m --pred_len %%l --fold %%f --epoch %EPOCH%
            )
        )
)
echo All experiments completed.
