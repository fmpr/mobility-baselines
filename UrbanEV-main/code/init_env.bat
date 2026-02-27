@echo off
@chcp 65001 >nul
setlocal enabledelayedexpansion

REM 选择是否使用清华镜像
echo.
echo "[1] 使用清华镜像源安装 pip 包"
echo "[2] 使用默认 pip 源"
set /p mirror="请选择是否启用清华镜像 (1/2): "

if "%mirror%"=="1" (
    set "PIP_INDEX=-i https://pypi.tuna.tsinghua.edu.cn/simple"

    echo 已启用清华镜像源。
) else (
    set "PIP_INDEX= "
    echo 使用默认 pip 源。
)

REM 检查环境是否已存在
call conda env list | findstr /C:"UrbanEV" >nul
if %errorlevel%==0 (
    echo 环境 UrbanEV 已存在，跳过创建。
) else (
    echo 正在创建 conda 环境 UrbanEV...
    call conda create -n UrbanEV python=3.8 --yes
)

REM 激活环境
call conda activate UrbanEV

REM 选择安装 CPU / GPU 版本
echo.
echo [1] 安装 CPU 版本
echo [2] 安装 GPU 版本 (CUDA 11.8)
set /p choice="请选择要安装的版本 (1/2): "


if "%choice%"=="1" (
    echo "安装 PyTorch CPU 版本..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cpu
    pip install scipy %PIP_INDEX%
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
) else if "%choice%"=="2" (
    echo "安装 PyTorch GPU 版本 (CUDA 11.8)..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    pip install scipy %PIP_INDEX%
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
) else (
    echo 输入无效，退出。
    exit /b
)

REM 安装其他 pip 包
pip install torch_geometric %PIP_INDEX%
pip install torch-geometric-temporal %PIP_INDEX%
pip install statsmodels %PIP_INDEX%
pip install scikit-learn %PIP_INDEX%
pip install openpyxl %PIP_INDEX%
pip install patool %PIP_INDEX%
pip install sktime %PIP_INDEX%
pip install matplotlib %PIP_INDEX%
pip install reformer_pytorch %PIP_INDEX%

echo.
echo ✅ 所有安装任务已完成。
pause
