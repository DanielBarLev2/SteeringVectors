import torch

from config.names import SV_PATH
from src.llama7.LlamaWrapper import LlamaWrapper

llama7 = LlamaWrapper(device_map="cpu",  # keep on CPU for simplicity
                      torch_dtype=torch.bfloat16)  # weights bf16, but will project to float32

r = llama7.load_sv(path=SV_PATH, dtype=torch.float32, map_location="cpu")

rms_r = llama7.model.model.norm(r)  # [4096, 1]
W_U = llama7.model.lm_head.weight.float()  # [vocab, 4096]
delta = (W_U @ rms_r).to(torch.float32)  # [vocab]

top_k = 30

# Promoted (most positive delta-logit)
vals_pos, ids_pos = torch.topk(delta, k=top_k)
pieces_pos = llama7.tokenizer.convert_ids_to_tokens(ids_pos.tolist())

print("Top promoted tokens:")
for tok_id, val in zip(ids_pos.tolist(), vals_pos.tolist()):
    decoded = llama7.tokenizer.decode([tok_id])
    print(f"{decoded!r:<20} id={tok_id:<6} Δlogit={val:.6f}")

print()

# Suppressed (most negative delta-logit)
vals_neg, ids_neg = torch.topk(-delta, k=top_k)
pieces_neg = llama7.tokenizer.convert_ids_to_tokens(ids_neg.tolist())

print("Top suppressed tokens:")
for tok_id, neg_val in zip(ids_neg.tolist(), vals_neg.tolist()):
    val = -neg_val
    decoded = llama7.tokenizer.decode([tok_id])
    print(f"{decoded!r:<20} id={tok_id:<6} Δlogit={val:.6f}")



