@echo off
REM install_gpu.bat - Script to install llama-cpp-python with multi-GPU support

REM Check if CUDA is installed
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: NVIDIA CUDA toolkit not found!
    echo Please install CUDA toolkit first: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)

REM Install regular dependencies
pip install -r requirements.txt

REM Install llama-cpp-python with multi-GPU support
echo Installing llama-cpp-python with multi-GPU support...

set CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all
set FORCE_CMAKE=1
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.2.72

echo Installation complete!
echo To verify installation, run: python test_gpu.py