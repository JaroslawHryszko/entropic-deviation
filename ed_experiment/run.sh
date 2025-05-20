python generate_multi_gpu_lowmem.py \
    --model models/meta-llama-3-8b-instruct.Q4_K_M.gguf \
    --prompts prompts/prompts.jsonl \
    --max_tokens 64 \
    --n_batch 32 \
    --reset_interval 10 \
    --sleep_time 2.0 \
    --out results/logits \
    --combine