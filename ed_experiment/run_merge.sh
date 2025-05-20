python merge_checkpoints.py --pattern "prefix_gpu*_chkpt_*.pt" --output merged_logits.pt
python ed.py --logits merged_logits.pt --out ed_results.csv
