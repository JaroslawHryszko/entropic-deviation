import torch
data = torch.load("logits_gpu0_chkpt_5.pt", map_location="cpu")
meta = data["meta"]
print(meta[0])