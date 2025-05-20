"""
test_gpu.py - Test script to verify multi-GPU support in llama-cpp-python
"""
import os
import torch
import subprocess
from llama_cpp import Llama

def check_gpu_info():
    """Print GPU information using nvidia-smi and PyTorch"""
    # Check using nvidia-smi
    try:
        print("=== GPU Information using nvidia-smi ===")
        subprocess.run(["nvidia-smi"], check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Could not run nvidia-smi")
    
    # Check using PyTorch
    if torch.cuda.is_available():
        print("\n=== GPU Information using PyTorch ===")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\nPyTorch does not detect any CUDA-capable GPUs")

def test_single_gpu():
    """Test loading model on a single GPU"""
    print("\n=== Testing Single GPU ===")
    
    # Check if model exists
    model_path = os.path.join("models", "meta-llama-3-8b-instruct.Q4_K_M.gguf")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download a model and update the path in this script")
        return
    
    # Load model on first GPU
    print("Loading model on GPU 0...")
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            main_gpu=0,
            tensor_split=None,  # Use only one GPU
            verbose=True
        )
        
        # Generate text to verify GPU usage
        prompt = "Hello, I'm testing GPU acceleration. Can you respond briefly?"
        print(f"\nTesting with prompt: '{prompt}'")
        response = llm(prompt, max_tokens=20)
        print(f"Response: {response['choices'][0]['text']}")
        print("Single GPU test successful!")
    except Exception as e:
        print(f"Error during single GPU test: {e}")

def test_multi_gpu():
    """Test loading model with multiple GPUs"""
    if torch.cuda.device_count() < 2:
        print("\n=== Multi-GPU Test Skipped ===")
        print("Not enough GPUs available for multi-GPU test")
        return
    
    print("\n=== Testing Multi-GPU ===")
    model_path = os.path.join("models", "meta-llama-3-8b-instruct.Q4_K_M.gguf")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    try:
        # Equally split model between available GPUs
        num_gpus = torch.cuda.device_count()
        split = [1.0/num_gpus] * num_gpus
        
        print(f"Loading model split across {num_gpus} GPUs with tensor_split={split}...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            tensor_split=split,
            verbose=True
        )
        
        prompt = "Please give a brief response to test multi-GPU operation."
        print(f"\nTesting with prompt: '{prompt}'")
        response = llm(prompt, max_tokens=20)
        print(f"Response: {response['choices'][0]['text']}")
        print("Multi-GPU test successful!")
    except Exception as e:
        print(f"Error during multi-GPU test: {e}")

if __name__ == "__main__":
    print("GPU Support Test for llama-cpp-python")
    print("=====================================")
    
    # Check GPU information
    check_gpu_info()
    
    # Test with single GPU
    test_single_gpu()
    
    # Test with multiple GPUs if available
    test_multi_gpu()