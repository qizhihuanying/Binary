@echo off
setlocal enabledelayedexpansion

:: Create logs directory
if not exist logs mkdir logs
if not exist logs\intfloat mkdir logs\intfloat

:: Set Python path
set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

:: Set experiment parameters
set LRS=1e-6 2e-6 3e-6 5e-6 1e-7 3e-7 5e-7 1e-5
set L2S=0

:: Run experiments with different parameters
for %%a in (0 1 2) do (
    for %%l in (%LRS%) do (
        for %%w in (%L2S%) do (
            echo Starting experiment - lr: %%l, l2: %%w
            
            %PYTHON_PATH% main.py ^
                --local_model_names intfloat/multilingual-e5-base ^
                --langs zh ^
                --use_binary_head ^
                --epochs 10 ^
                --lr %%l ^
                --l2 %%w ^
                --batch_size 32 ^
                --train_sample_ratio 1.0 ^
                --val_ratio 0.2 ^
                --test_ratio 1.0 ^
                --base_trainable_layers 0
                
            echo Experiment completed - lr: %%l, l2: %%w
        )
    )
)

echo All experiments completed!
