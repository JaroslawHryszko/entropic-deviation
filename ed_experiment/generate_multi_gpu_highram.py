#!/usr/bin/env python
"""
generate_multi_gpu.py - Run language model inference on multiple GPUs in parallel

EXPERIMENT SUMMARY:
1. Study of entropic deviation (ED) in large language models
2. ED measures how much a model's token probability distribution deviates from uniform distribution
3. This may indicate "proto-agency" or non-random behavior in language models
4. Experiment collects logits across multiple temperatures (0.7, 1.0, 1.3)
5. Uses domain-diverse prompts (Wikipedia, News, Fiction, Code)
6. Runs parallel inference on multiple GPUs for efficiency
7. Preserves all data needed for statistical falsification tests (F1-F8)
8. Maintains experiment integrity with consistent parameters

Author: Jarosław Hryszko (jaroslaw.hryszko@uj.edu.pl)
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

def load_prompts(path):
    """
    Load prompts from file. Handles both JSON and plain text formats.
    
    EXPERIMENT POINT 5: Uses domain-diverse prompts stored in a JSONL file
    with potential domain labels (Wikipedia, News, Fiction, Code)
    """
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
    """
    Process a batch of prompts on a specific GPU.
    
    EXPERIMENT POINTS 1-4, 6-7: This function:
    - Runs model inference across multiple prompts and temperatures
    - Collects logits for every token generated
    - Preserves association between logits, prompts, and temperature settings
    - Ensures results from each GPU are saved separately for later analysis
    """
    logger.info(f"GPU {gpu_id}: Starting to process {len(prompts_batch)} prompts")
    start_time = time.time()
    
    out_logits, meta = [], []
    
    # Temp file for periodic saving
    temp_out_file = f"{out_file_prefix}_gpu{gpu_id}_temp.pt"
    final_out_file = f"{out_file_prefix}_gpu{gpu_id}.pt"
    
    # Check if we have previously saved results to resume from
    if os.path.exists(temp_out_file):
        try:
            logger.info(f"Found temporary results file {temp_out_file}, attempting to resume")
            temp_data = torch.load(temp_out_file)
            out_logits = temp_data.get("logits", [])
            meta = temp_data.get("meta", [])
            processed_pairs = {(m["prompt"], m["temp"]) for m in meta}
            logger.info(f"Resuming from {len(out_logits)} previously processed combinations")
        except Exception as e:
            logger.error(f"Error loading temporary results: {e}. Starting from scratch.")
            processed_pairs = set()
    else:
        processed_pairs = set()
    
    # Create list of combinations to process (excluding already processed ones)
    combinations = [(t, p) for t, p in itertools.product(temps, prompts_batch) 
                    if (p, t) not in processed_pairs]
    
    # Save interval - after every ~10% of the remaining combinations
    save_interval = max(1, len(combinations) // 10)
    
    # Process combinations
    for idx, (t, prompt) in enumerate(tqdm.tqdm(combinations, desc=f"GPU {gpu_id}", position=gpu_id)):
        try:
            # Tokenize input
            input_tokens = llm.tokenize(prompt.encode('utf-8'))
            
            # Reset model state
            llm.reset()

            # Generate response with logits
            # EXPERIMENT POINT 2: Collecting token probability distributions
            response = llm.create_completion(
                prompt, 
                max_tokens=max_tokens, 
                temperature=t,  # EXPERIMENT POINT 4: Use varying temperatures
                logprobs=5      # This activates logprobs calculation from logits
            )
            
            # Extract logits from model
            # EXPERIMENT POINT 1: Storing logits for entropy deviation calculation
            if hasattr(llm, "_ctx") and hasattr(llm, "scores"):
                logits_array = np.array(llm.scores, dtype=np.float32)
                logits_tensor = torch.from_numpy(logits_array)
                
                # Save logits and metadata
                out_logits.append(logits_tensor)
                meta.append({
                    "prompt": prompt, 
                    "temp": t, 
                    "seq_len": logits_array.shape[0], 
                    "gpu_id": gpu_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log progress periodically
                if (idx + 1) % 5 == 0:
                    logger.info(f"GPU {gpu_id} - Processed {idx + 1}/{len(combinations)}: "
                                f"prompt length: {len(prompt.split())}, temp: {t}, "
                                f"output tokens: {logits_array.shape[0]}")
                
                # Periodic saving to prevent data loss
                if (idx + 1) % save_interval == 0:
                    try:
                        torch.save({"logits": out_logits, "meta": meta}, temp_out_file)
                        logger.info(f"GPU {gpu_id} - Saved {len(out_logits)} results to temp file")
                    except Exception as e:
                        logger.error(f"GPU {gpu_id} - Error saving temp file: {e}")
            else:
                logger.warning(f"GPU {gpu_id} - Cannot access logits for prompt: '{prompt[:30]}...'")
                
        except Exception as e:
            logger.error(f"GPU {gpu_id} - Error processing prompt '{prompt[:30]}...': {e}")
            # Continue with next prompt rather than terminating entire batch
    
    # Final save for this GPU's results
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
    """
    Main function to parse arguments and orchestrate parallel processing
    on multiple GPUs.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate token logits from language models running on multiple GPUs in parallel "
                    "for entropic deviation analysis",
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
    parser.add_argument("--resume", action="store_true",
                      help="Resume from previous run if temporary files exist")
    args = parser.parse_args()

    # Set up logger
    global logger
    logger = setup_logger(args.log)
    logger.info(f"Starting entropic deviation data collection with parameters: {vars(args)}")

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
        # Initialize model on GPU 0
        logger.info("Initializing model on GPU 0...")
        llm_gpu0 = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,        # All layers on GPU
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,        # CRITICAL: Collect logits for all tokens
            main_gpu=0,             # Assign to GPU 0
            tensor_split=[1.0, 0.0], # 100% on GPU 0
            verbose=True,
            n_batch=args.n_batch,
            use_mlock=True          # Prevent memory from being swapped out
        )
        
        # Quick model test to verify GPU 0 is working
        logger.info("Testing model on GPU 0...")
        test_result = llm_gpu0("Test.", max_tokens=5)
        logger.info(f"GPU 0 test output: {test_result['choices'][0]['text']}")
        
        # Initialize model on GPU 1
        logger.info("Initializing model on GPU 1...")
        llm_gpu1 = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,
            main_gpu=1,             # Assign to GPU 1
            tensor_split=[0.0, 1.0], # 100% on GPU 1
            verbose=True,
            n_batch=args.n_batch,
            use_mlock=True
        )
        
        # Quick model test to verify GPU 1 is working
        logger.info("Testing model on GPU 1...")
        test_result = llm_gpu1("Test.", max_tokens=5)
        logger.info(f"GPU 1 test output: {test_result['choices'][0]['text']}")
        
        # EXPERIMENT POINT 6: Run parallel processing on two GPUs
        logger.info("Starting parallel processing threads...")
        
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
        
        # Wait for both threads to complete
        thread_gpu0.join()
        thread_gpu1.join()
        
        logger.info("Processing on both GPUs complete.")
        
        # EXPERIMENT POINT 7: Combine results from both GPUs if requested
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
                    logger.info(f"Total combined samples: {len(combined_logits)}")
                else:
                    logger.error("Missing logits in one or both result files")
            except Exception as e:
                logger.error(f"Error combining results: {e}")
        
        logger.info("==== Experiment data collection complete ====")
        logger.info("Next steps: Run 'ed.py' to compute entropic deviation metrics from collected logits")
        logger.info("          Then run 'stats.py' to perform statistical tests (F1-F8)")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()