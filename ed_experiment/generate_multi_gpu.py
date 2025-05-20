#!/usr/bin/env python
"""
generate_multi_gpu.py - Run language model inference on multiple GPUs in parallel

This script divides prompts between available GPUs and runs inference
in parallel to generate token logits for Entropic Deviation analysis.
"""
import argparse
import json
import os
import itertools
import torch
import numpy as np
import tqdm
import logging
import threading
import time
import subprocess
import sys
from datetime import datetime
from llama_cpp import Llama

# Configure logger
def setup_logger(log_file=None):
    """Set up logging to console and optionally to file"""
    logger = logging.getLogger("multi_gpu_inference")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def check_gpu_usage(gpu_id):
    """Check GPU utilization and memory usage with nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        usage, memory = map(int, result.stdout.strip().split(','))
        return usage, memory
    except Exception as e:
        logger.error(f"Error checking GPU {gpu_id} usage: {e}")
        return -1, -1

def load_prompts(path):
    """Load prompts from file. Handles both JSON and plain text formats."""
    logger.info(f"Loading prompts from file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] if line.strip().startswith("{")
                      else line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        raise

def split_prompts(prompts, gpu_split_ratio=0.5):
    """Split prompts between GPUs according to the provided ratio."""
    split_point = int(len(prompts) * gpu_split_ratio)
    return prompts[:split_point], prompts[split_point:]

def process_batch(llm, prompts_batch, temps, max_tokens, out_file_prefix, gpu_id):
    """Process a batch of prompts on a specific GPU."""
    logger.info(f"GPU {gpu_id}: Starting to process {len(prompts_batch)} prompts")
    start_time = time.time()
    
    # Check initial GPU usage to confirm it's being used
    gpu_usage, gpu_memory = check_gpu_usage(gpu_id)
    logger.info(f"GPU {gpu_id} initial usage: {gpu_usage}%, memory: {gpu_memory}MB")
    
    out_logits, meta = [], []
    
    # Temp file for periodic saving
    temp_out_file = f"{out_file_prefix}_gpu{gpu_id}_temp.pt"
    save_interval = max(1, len(list(itertools.product(temps, prompts_batch))) // 10)
    
    # Counter for combinations
    total_combinations = len(list(itertools.product(temps, prompts_batch)))
    processed = 0
    
    for t, prompt in tqdm.tqdm(list(itertools.product(temps, prompts_batch)), 
                             desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            # Check GPU utilization every 5 prompts
            if processed % 5 == 0:
                gpu_usage, gpu_memory = check_gpu_usage(gpu_id)
                logger.info(f"GPU {gpu_id} current usage: {gpu_usage}%, memory: {gpu_memory}MB")
            
            # Tokenize input
            input_tokens = llm.tokenize(prompt.encode('utf-8'))
            
            # Reset model state
            llm.reset()

            # Measure generation time
            gen_start = time.time()
            
            # Generate response with logits
            response = llm.create_completion(
                prompt, 
                max_tokens=max_tokens, 
                temperature=t,
                logprobs=5  # This parameter activates logprobs calculation from logits
            )
            
            gen_time = time.time() - gen_start
            
            processed += 1
            if processed % 5 == 0:  # Log every 5 prompts for readability
                logger.info(f"GPU {gpu_id} - Processed {processed}/{total_combinations}: '{prompt[:30]}...' (Temp: {t}, Time: {gen_time:.2f}s)")
            
            # Extract logits from model
            if hasattr(llm, "_ctx") and hasattr(llm, "scores"):
                logits_array = np.array(llm.scores, dtype=np.float32)
                logits_tensor = torch.from_numpy(logits_array)
                
                out_logits.append(logits_tensor)
                meta.append({
                    "prompt": prompt, 
                    "temp": t, 
                    "seq_len": logits_array.shape[0], 
                    "gpu_id": gpu_id,
                    "gen_time": gen_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Periodic saving
                if processed % save_interval == 0:
                    try:
                        torch.save({"logits": out_logits, "meta": meta}, temp_out_file)
                        logger.info(f"GPU {gpu_id} - Saved {processed}/{total_combinations} results to temp file")
                    except Exception as e:
                        logger.error(f"GPU {gpu_id} - Error saving temp file: {e}")
            else:
                logger.warning(f"GPU {gpu_id} - Cannot access logits for prompt: '{prompt[:30]}...'")
                logger.debug(f"Available attributes: {[attr for attr in dir(llm) if not attr.startswith('_')]}")
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} - Error processing prompt '{prompt[:30]}...': {e}")
    
    # Final save for this GPU
    final_out_file = f"{out_file_prefix}_gpu{gpu_id}.pt"
    try:
        if out_logits:
            torch.save({"logits": out_logits, "meta": meta}, final_out_file)
            logger.info(f"GPU {gpu_id} - Saved final file {final_out_file} with {len(out_logits)} results")
            
            if os.path.exists(temp_out_file):
                os.remove(temp_out_file)
                logger.info(f"GPU {gpu_id} - Removed temp file {temp_out_file}")
        else:
            logger.warning(f"GPU {gpu_id} - No results to save")
    except Exception as e:
        logger.error(f"GPU {gpu_id} - Error saving final file: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"GPU {gpu_id} - Processing complete in {elapsed_time:.2f} seconds")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate token logits from language models running on multiple GPUs in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompts", required=True, help="Path to prompts file")
    parser.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0, 1.3], 
                      help="List of temperatures to use")
    parser.add_argument("--max_tokens", type=int, default=128, 
                      help="Maximum number of tokens to generate")
    parser.add_argument("--n_ctx", type=int, default=2048, 
                      help="Context size for the model")
    parser.add_argument("--out", default="logits", 
                      help="Output filename prefix")
    parser.add_argument("--log", default=None, 
                      help="Path to log file (default: console only)")
    parser.add_argument("--gpu_split", type=float, default=0.5, 
                      help="Ratio for splitting prompts between GPUs (0-1.0)")
    parser.add_argument("--chat_format", default="llama-3", 
                      help="Chat format (e.g., llama-3, chatml)")
    parser.add_argument("--combine", action="store_true", 
                      help="Combine results from both GPUs into one file")
    parser.add_argument("--n_batch", type=int, default=1024, 
                      help="Batch size for the model")
    parser.add_argument("--f16_kv", action="store_true", 
                      help="Use half-precision for key/value cache")
    args = parser.parse_args()

    # Set up logger
    global logger
    logger = setup_logger(args.log)
    logger.info(f"Starting processing with parameters: {vars(args)}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.error("CUDA not available! CUDA support is required.")
        return

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Created output directory: {out_dir}")

    # Load prompts
    try:
        all_prompts = load_prompts(args.prompts)
        if not all_prompts:
            logger.error("No prompts to process")
            return
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return
    
    # Split prompts for two GPUs
    prompts_gpu0, prompts_gpu1 = split_prompts(all_prompts, args.gpu_split)
    logger.info(f"Assigned {len(prompts_gpu0)} prompts to GPU 0 and {len(prompts_gpu1)} to GPU 1")

    try:
        # Check initial GPU usage
        for gpu_id in range(2):
            usage, memory = check_gpu_usage(gpu_id)
            logger.info(f"Initial GPU {gpu_id} usage: {usage}%, memory: {memory}MB")
        
        # Initialize model on GPU 0
        logger.info("Initializing model on GPU 0...")
        llm_gpu0 = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,      # All layers on GPU
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,      # Critical for logits
            main_gpu=0,           # Assign to GPU 0
            tensor_split=[1.0, 0.0],  # 100% on GPU 0, 0% on GPU 1
            verbose=True,
            n_batch=args.n_batch,
            f16_kv=args.f16_kv    # Half-precision for key/value cache
        )
        
        # Test GPU 0
        logger.info("Testing computation on GPU 0...")
        test_prompt = "Test GPU 0"
        llm_gpu0(test_prompt, max_tokens=10)
        usage0, memory0 = check_gpu_usage(0)
        logger.info(f"After test - GPU 0 usage: {usage0}%, memory: {memory0}MB")
        
        # Initialize model on GPU 1
        logger.info("Initializing model on GPU 1...")
        llm_gpu1 = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,      # All layers on GPU
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,      # Critical for logits
            main_gpu=1,           # Assign to GPU 1
            tensor_split=[0.0, 1.0],  # 0% on GPU 0, 100% on GPU 1
            verbose=True,
            n_batch=args.n_batch,
            f16_kv=args.f16_kv    # Half-precision for key/value cache
        )
        
        # Test GPU 1
        logger.info("Testing computation on GPU 1...")
        test_prompt = "Test GPU 1"
        llm_gpu1(test_prompt, max_tokens=10)
        usage1, memory1 = check_gpu_usage(1)
        logger.info(f"After test - GPU 1 usage: {usage1}%, memory: {memory1}MB")
        
        # Check if GPUs are being used
        if usage0 <= 5 and usage1 <= 5:
            logger.warning("Warning: Very low GPU usage. Model may not be using GPU acceleration!")
        
        # Run parallel processing on two GPUs
        logger.info("Starting processing threads...")
        
        thread_gpu0 = threading.Thread(
            target=process_batch, 
            args=(llm_gpu0, prompts_gpu0, args.temps, args.max_tokens, args.out, 0)
        )
        
        thread_gpu1 = threading.Thread(
            target=process_batch, 
            args=(llm_gpu1, prompts_gpu1, args.temps, args.max_tokens, args.out, 1)
        )
        
        # Start threads
        thread_gpu0.start()
        thread_gpu1.start()
        
        # Monitor resources during processing
        while thread_gpu0.is_alive() or thread_gpu1.is_alive():
            for gpu_id in range(2):
                usage, memory = check_gpu_usage(gpu_id)
                logger.info(f"Monitoring - GPU {gpu_id}: {usage}% usage, {memory}MB memory")
            time.sleep(30)  # Update every 30 seconds
        
        # Wait for both threads to complete
        thread_gpu0.join()
        thread_gpu1.join()
        
        logger.info("Processing on both GPUs complete.")
        
        # Optionally combine results from both GPUs
        if args.combine:
            try:
                logger.info("Combining results from both GPUs...")
                results_gpu0 = torch.load(f"{args.out}_gpu0.pt")
                results_gpu1 = torch.load(f"{args.out}_gpu1.pt")
                
                # Check if we have logits in both files
                if "logits" in results_gpu0 and "logits" in results_gpu1:
                    combined_logits = results_gpu0["logits"] + results_gpu1["logits"]
                    combined_meta = results_gpu0["meta"] + results_gpu1["meta"]
                    
                    combined_file = f"{args.out}_combined.pt"
                    torch.save({"logits": combined_logits, "meta": combined_meta}, combined_file)
                    logger.info(f"Combined results from both GPUs saved to {combined_file}")
                else:
                    logger.error("Missing logits in one or both result files")
            except Exception as e:
                logger.error(f"Error combining results: {e}")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()