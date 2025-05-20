#!/usr/bin/env python
"""
generate_single_gpu.py - Run language model inference on a single GPU

EXPERIMENT SUMMARY:
1. Study of entropic deviation (ED) in large language models
2. ED measures how much a model's token probability distribution deviates from uniform distribution
3. This may indicate "proto-agency" or non-random behavior in language models
4. Experiment collects logits across multiple temperatures (0.7, 1.0, 1.3)
5. Uses domain-diverse prompts (Wikipedia, News, Fiction, Code)
6. Runs inference on a single GPU for simplicity and reliability
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
import time
from datetime import datetime
from llama_cpp import Llama

# Configure logger
def setup_logger(log_file=None):
    """Set up logging to console and optionally to file"""
    logger = logging.getLogger("single_gpu_inference")
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

def process_prompts(llm, prompts, temps, max_tokens, out_file):
    """
    Process all prompts on a single GPU.
    
    EXPERIMENT POINTS 1-4, 7: This function:
    - Runs model inference across multiple prompts and temperatures
    - Collects logits for every token generated
    - Preserves association between logits, prompts, and temperature settings
    - Ensures results are saved for later analysis
    """
    logger.info(f"Starting to process {len(prompts)} prompts with {len(temps)} temperatures")
    start_time = time.time()
    
    out_logits, meta = [], []
    
    # Temp file for periodic saving
    temp_out_file = f"{os.path.splitext(out_file)[0]}_temp.pt"
    
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
    combinations = [(t, p) for t, p in itertools.product(temps, prompts) 
                    if (p, t) not in processed_pairs]
    
    total_combinations = len(combinations)
    logger.info(f"Processing {total_combinations} prompt-temperature combinations")
    
    # Save interval - after every ~5% of the remaining combinations or every 20, whichever is less
    save_interval = min(20, max(1, total_combinations // 20))
    
    # Process combinations
    for idx, (t, prompt) in enumerate(tqdm.tqdm(combinations, desc="Processing")):
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
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log progress periodically
                if (idx + 1) % 5 == 0:
                    logger.info(f"Processed {idx + 1}/{total_combinations}: "
                                f"prompt length: {len(prompt.split())}, temp: {t}, "
                                f"output tokens: {logits_array.shape[0]}")
                
                # Periodic saving to prevent data loss
                if (idx + 1) % save_interval == 0:
                    try:
                        torch.save({"logits": out_logits, "meta": meta}, temp_out_file)
                        logger.info(f"Saved {len(out_logits)} results to temp file")
                    except Exception as e:
                        logger.error(f"Error saving temp file: {e}")
            else:
                logger.warning(f"Cannot access logits for prompt: '{prompt[:30]}...'")
                
        except Exception as e:
            logger.error(f"Error processing prompt '{prompt[:30]}...': {e}")
            # Continue with next prompt rather than terminating entire batch
    
    # Final save of results
    try:
        if out_logits:
            torch.save({"logits": out_logits, "meta": meta}, out_file)
            logger.info(f"Saved final file {out_file} with {len(out_logits)} results")
            
            if os.path.exists(temp_out_file):
                os.remove(temp_out_file)
                logger.info(f"Removed temp file {temp_out_file}")
        else:
            logger.warning(f"No results to save")
    except Exception as e:
        logger.error(f"Error saving final file: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

def main():
    """
    Main function to parse arguments and run the model on a single GPU.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate token logits from language models running on a single GPU "
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
    parser.add_argument("--out", default="logits.pt", 
                      help="Output filename")
    parser.add_argument("--log", default=None, 
                      help="Path to log file (default: console only)")
    parser.add_argument("--chat_format", default="llama-3", 
                      help="Chat format (e.g., llama-3, chatml)")
    parser.add_argument("--n_batch", type=int, default=1024, 
                      help="Batch size for the model")
    parser.add_argument("--gpu_id", type=int, default=0,
                      help="GPU ID to use (if multiple GPUs are available)")
    parser.add_argument("--max_prompts", type=int, default=None,
                      help="Maximum number of prompts to process (for testing)")
    args = parser.parse_args()

    # Set up logger
    global logger
    logger = setup_logger(args.log)
    logger.info(f"Starting entropic deviation data collection with parameters: {vars(args)}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA available: {device_count} devices")
        if args.gpu_id >= device_count:
            logger.error(f"Requested GPU ID {args.gpu_id} is not available. "
                        f"Only {device_count} GPUs detected.")
            return
        logger.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
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
            
        # Apply prompt limit if specified (useful for testing)
        if args.max_prompts is not None and args.max_prompts > 0:
            if len(all_prompts) > args.max_prompts:
                logger.info(f"Limiting to first {args.max_prompts} prompts (of {len(all_prompts)})")
                all_prompts = all_prompts[:args.max_prompts]
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return

    try:
        # Initialize model on selected GPU
        logger.info(f"Initializing model on GPU {args.gpu_id}...")
        llm = Llama(
            model_path=args.model, 
            n_gpu_layers=-1,        # All layers on GPU
            n_ctx=args.n_ctx, 
            chat_format=args.chat_format,
            logits_all=True,        # CRITICAL: Collect logits for all tokens
            main_gpu=args.gpu_id,   # Specify which GPU to use
            verbose=True,
            n_batch=args.n_batch,
            use_mlock=True          # Prevent memory from being swapped out
        )
        
        # Quick model test to verify GPU is working
        logger.info("Testing model...")
        test_result = llm("Test.", max_tokens=5)
        logger.info(f"Test output: {test_result['choices'][0]['text']}")
        
        # Process all prompts
        process_prompts(llm, all_prompts, args.temps, args.max_tokens, args.out)
        
        logger.info("==== Experiment data collection complete ====")
        logger.info("Next steps: Run 'ed.py' to compute entropic deviation metrics from collected logits")
        logger.info("          Then run 'stats.py' to perform statistical tests (F1-F8)")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()