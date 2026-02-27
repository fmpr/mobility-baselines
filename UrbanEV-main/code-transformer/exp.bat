@echo off

set pre_lens=3 6 9 12
set folds=1 2 3 4 5 6
set EPOCH=1

for %%l in (%pre_lens%) do (
    for %%f in (%folds%) do (
        python run.py ^
            --seq_len 12 ^
            --label_len 12 ^
            --epoch %EPOCH% ^
            --model TimeXer ^
            --pred_len %%l ^
            --fold %%f ^
    )
)

for %%l in (%pre_lens%) do (
    for %%f in (%folds%) do (
        python run.py ^
            --seq_len 12 ^
            --label_len 12 ^
            --epoch %EPOCH% ^
            --model TimesNet ^
            --pred_len %%l ^
            --fold %%f ^

    )
)

echo All tasks completed!
pause
