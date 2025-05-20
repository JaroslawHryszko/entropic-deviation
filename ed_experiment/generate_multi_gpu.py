#!/usr/bin/env python
"""
generate_multi_gpu.py - Memory-optimized version for Entropic Deviation experiments

This script processes prompts in smaller batches to avoid OOM issues, with memory
monitoring and adaptive pacing to ensure experiment completion, even if slower.
"""
import argparse
import json
import os
import torch
import numpy as np
import tqdm
import logging
import threading
import time
import subprocess
import sys
import gc
import psutil
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

def check_memory():
    """Check system memory usage"""
    memory = psutil.virtual_memory()
    return memory.percent, memory.available / (1024 * 1024 * 1024)  # percentage used and GB available

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

def load_prompts(path, start_idx=0, max_prompts=None):
    """Load prompts from file with option to limit count."""
    logger.info(f"Loading prompts from file: {path} (start_idx={start_idx}, max_prompts={max_prompts})")
    try:
        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if line.strip():
                    if line.strip().startswith("{"):
                        prompts.append(json.loads(line)["prompt"])
                    else:
                        prompts.append(line.strip())
                if max_prompts is not None and len(prompts) >= max_prompts:
                    break
        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        raise

def split_prompts(prompts, gpu_split_ratio=0.5):
    """Split prompts between GPUs according to the provided ratio."""
    split_point = int(len(prompts) * gpu_split_ratio)
    return prompts[:split_point], prompts[split_point:]

def process_batch(llm, prompts_batch, temps, max_tokens, out_file_prefix, gpu_id, 
                  micro_batch_size=2, save_interval=5, sleep_time=1.0):
    """Process a batch of prompts on a specific GPU with micro-batching and memory management."""
    logger.info(f"GPU {gpu_id}: Starting to process {len(prompts_batch)} prompts")
    start_time = time.time()
    
    # Initial GPU check
    gpu_usage, gpu_memory = check_gpu_usage(gpu_id)
    logger.info(f"GPU {gpu_id} initial usage: {gpu_usage}%, memory: {gpu_memory}MB")
    
    out_logits, meta = [], []
    temp_out_file = f"{out_file_prefix}_gpu{gpu_id}_temp.pt"
    
    # Process prompts in small micro-batches to control memory usage
    total_combinations = len(temps) * len(prompts_batch)
    processed = 0
    
    # Create combinations but don't materialize the whole list at once
    all_combinations = [(t, prompt) for t in temps for prompt in prompts_batch]
    
    # Process in micro-batches
    for batch_start in range(0, len(all_combinations), micro_batch_size):
        batch_end = min(batch_start + micro_batch_size, len(all_combinations))
        current_batch = all_combinations[batch_start:batch_end]
        
        mem_percent, mem_avail = check_memory()
        if mem_percent > 90:  # System memory critical
            logger.warning(f"System memory high ({mem_percent}%), pausing for 60s")
            time.sleep(60)  # Pause for a minute to let memory stabilize
            gc.collect()  # Force garbage collection
        
        for t, prompt in current_batch:
            try:
                # Space out processing to reduce memory pressure
                time.sleep(sleep_time)  # Add small delay between prompts
                
                # Check system memory
                mem_percent, mem_avail = check_memory()
                if mem_percent > 85:
                    pause_time = (mem_percent - 80) * 2  # Adaptive pause based on memory pressure
                    logger.warning(f"Memory pressure high ({mem_percent}%), pausing for {pause_time}s")
                    time.sleep(pause_time)
                    gc.collect()
                
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
                if processed % 5 == 0:  # Log progress every 5 prompts
                    logger.info(f"GPU {gpu_id} - {processed}/{total_combinations}: '{prompt[:20]}...' "
                               f"(Temp: {t}, Time: {gen_time:.2f}s, Mem: {mem_percent}%)")
                
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
                    
                    # Save results more frequently to avoid losing progress
                    if processed % save_interval == 0:
                        try:
                            torch.save({"logits": out_logits, "meta": meta}, temp_out_file)
                            logger.info(f"GPU {gpu_id} - Saved progress ({processed}/{total_combinations})")
                        except Exception as e:
                            logger.error(f"GPU {gpu_id} - Error saving temp file: {e}")
                else:
                    logger.warning(f"GPU {gpu_id} - Cannot access logits for prompt: '{prompt[:20]}...'")
            
            except Exception as e:
                logger.error(f"GPU {gpu_id} - Error processing prompt '{prompt[:20]}...': {e}")
                # Continue processing despite errors
        
        # Clear some memory after each micro-batch
        if hasattr(llm, "eval"):
            llm.eval("clear_cache();")  # Clear KV cache if possible
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Final save for this GPU
    final_out_file = f"{out_file_prefix}_gpu{gpu_id}.pt"
    try:
        if out_logits:
            torch.save({"logits": out_logits, "meta": meta}, final_out_file)
            logger.info(f"GPU {gpu_id} - Saved final file with {len(out_logits)} results")
            
            if os.path.exists(temp_out_file):
                os.remove(temp_out_file)
        else:
            logger.warning(f"GPU {gpu_id} - No results to save")
    except Exception as e:
        logger.error(f"GPU {gpu_id} - Error saving final file: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"GPU {gpu_id} - Processing complete in {elapsed_time:.2f} seconds")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Memory-optimized script for Entropic Deviation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompts", required=True, help="Path to prompts file")
    parser.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0, 1.3], 
                      help="List of temperatures to use")
    parser.add_argument("--max_tokens", type=int, default=64,  # Reduced from 128
                      help="Maximum number of tokens to generate")
    parser.add_argument("--n_ctx", type=int, default=512,  # Reduced from 2048
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
    parser.add_argument("--n_batch", type=int, default=32,  # Reduced from 1024
                      help="Batch size for the model")
    parser.add_argument("--f16_kv", action="store_true", default=True,  # Default to True
                      help="Use half-precision for key/value cache")
    parser.add_argument("--micro_batch", type=int, default=2,
                      help="Number of prompts to process in each micro-batch")
    parser.add_argument("--save_interval", type=int, default=5,
                      help="Save progress every N prompts")
    parser.add_argument("--max_prompts", type=int, default=None,
                      help="Maximum number of prompts to process (for testing)")
    parser.add_argument("--start_idx", type=int, default=0,
                      help="Starting index in prompts file (for resuming)")
    parser.add_argument("--sleep_time", type=float, default=1.0,
                      help="Seconds to wait between prompts (higher = less memory pressure)")
    parser.add_argument("--n_gpu_layers", type=int, default=32,  # Process fewer layers on GPU
                      help="Number of layers to offload to GPU (-1 for all)")
    args = parser.parse_args()

    # Set up logger
    global logger
    logger = setup_logger(args.log)
    logger.info(f"Starting memory-optimized processing with parameters: {vars(args)}")
    
    # Check system memory
    mem_percent, mem_avail = check_memory()
    logger.info(f"Initial system memory: {mem_percent}% used, {mem_avail:.2f}GB available")

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.error("CUDA not available! CUDA support is required.")
        return

    # Create output directory if needed
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Created output directory: {out_dir}")

    # Load prompts (memory-optimized: limit if needed)
    try:
        all_prompts = load_prompts(args.prompts, args.start_idx, args.max_prompts)
        if not all_prompts:
            logger.error("No prompts to process")
            return
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return
    
    # Split prompts for GPUs
    prompts_gpu0, prompts_gpu1 = split_prompts(all_prompts, args.gpu_split)
    logger.info(f"Assigned {len(prompts_gpu0)} prompts to GPU 0 and {len(prompts_gpu1)} to GPU 1")

    try:
        # Initialize model on GPU 0
        logger.info("Initializing model on GPU 0...")
        llm_gpu0 = Llama(
            model_path=args.model, 
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,
            main_gpu=0,
            tensor_split=[1.0, 0.0],
            verbose=True,
            n_batch=args.n_batch,
            f16_kv=args.f16_kv
        )
        
        # Initialize model on GPU 1
        logger.info("Initializing model on GPU 1...")
        llm_gpu1 = Llama(
            model_path=args.model, 
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,
            main_gpu=1,
            tensor_split=[0.0, 1.0],
            verbose=True,
            n_batch=args.n_batch,
            f16_kv=args.f16_kv
        )
        
        # Run parallel processing
        logger.info("Starting processing threads...")
        
        thread_gpu0 = threading.Thread(
            target=process_batch, 
            args=(llm_gpu0, prompts_gpu0, args.temps, args.max_tokens, args.out, 0, 
                  args.micro_batch, args.save_interval, args.sleep_time)
        )
        
        thread_gpu1 = threading.Thread(
            target=process_batch, 
            args=(llm_gpu1, prompts_gpu1, args.temps, args.max_tokens, args.out, 1,
                  args.micro_batch, args.save_interval, args.sleep_time)
        )
        
        # Start threads
        thread_gpu0.start()
        thread_gpu1.start()
        
        # Monitor resources during processing (less frequent to reduce overhead)
        while thread_gpu0.is_alive() or thread_gpu1.is_alive():
            mem_percent, mem_avail = check_memory()
            logger.info(f"System memory: {mem_percent}% used, {mem_avail:.2f}GB available")
            
            for gpu_id in range(2):
                usage, memory = check_gpu_usage(gpu_id)
                logger.info(f"GPU {gpu_id}: {usage}% usage, {memory}MB memory")
            
            # Sleep longer between checks to reduce overhead
            time.sleep(60)
        
        # Wait for both threads to complete
        thread_gpu0.join()
        thread_gpu1.join()
        
        logger.info("Processing on both GPUs complete.")
        
        # Combine results (if requested)
        if args.combine:
            try:
                logger.info("Combining results from both GPUs...")
                results_gpu0 = torch.load(f"{args.out}_gpu0.pt")
                results_gpu1 = torch.load(f"{args.out}_gpu1.pt")
                
                combined_logits = results_gpu0["logits"] + results_gpu1["logits"]
                combined_meta = results_gpu0["meta"] + results_gpu1["meta"]
                
                combined_file = f"{args.out}_combined.pt"
                torch.save({"logits": combined_logits, "meta": combined_meta}, combined_file)
                logger.info(f"Combined results saved to {combined_file}")
            except Exception as e:
                logger.error(f"Error combining results: {e}")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()