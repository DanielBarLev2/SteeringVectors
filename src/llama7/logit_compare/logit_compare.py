import json
import torch
from pathlib import Path

from src.llama7.LlamaWrapper import LlamaWrapper
from src.llama7.steer import register_residual_hooks
from config.names import SV_PATH

PROMPT = "What is the average number of hours of sleep a person should get?"
OUT_JSON = "logit_compare.json"
MAX_NEW_TOKENS = 16

torch.set_grad_enabled(False)
llama7 = LlamaWrapper()
tok = llama7.tokenizer
model = llama7.model.eval()

r = llama7.load_sv(path=SV_PATH, dtype=torch.float32, map_location="cpu")

input_ids, attn_mask, prompt_len = llama7.to_tokens(PROMPT)

# base completion (no hooks)
with torch.inference_mode():
    gen_out = model.generate(input_ids=input_ids,
                             attention_mask=attn_mask,
                             do_sample=False,
                             use_cache=True,
                             top_p=None,
                             max_new_tokens=MAX_NEW_TOKENS,
                             return_dict_in_generate=True,
                             output_scores=False,
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)

seq = gen_out.sequences[0]
base_text = tok.decode(seq, skip_special_tokens=True)

gen_ids = seq[prompt_len:]
gen_len = int(gen_ids.size(0))

# Pre-decode response tokens for display (keep exact tokenization)
resp_tokens_text = [tok.decode([int(t)], skip_special_tokens=True) for t in gen_ids.tolist()]

# steered
per_step_top = []
for k in range(gen_len):
    # prefix: prompt + first k generated tokens (from base trajectory)
    prefix_ids = torch.cat([input_ids, gen_ids[:k].unsqueeze(0)], dim=1)
    prefix_mask = torch.ones_like(prefix_ids)

    base_logits = llama7.next_token_logits(prefix_ids, prefix_mask)

    hooks = register_residual_hooks(model, r_vec=r, alpha=1, start_pos=prompt_len)
    try:
        steer_logits = llama7.next_token_logits(prefix_ids, prefix_mask)
    finally:
        for h in hooks:
            h.remove()

    delta = (steer_logits - base_logits).to(torch.float32)
    vals_pos, ids_pos = torch.topk(delta, k=5)

    step_items = []
    for v, i in zip(vals_pos.tolist(), ids_pos.tolist()):
        step_items.append({"label": tok.decode([i], skip_special_tokens=True),
                           "delta": float(v),
                           "token_id": int(i)})
    per_step_top.append(step_items)

# save JSON
data = {"prompt": PROMPT,
        "alpha": 1,
        "topk_per_step": 5,
        "response_tokens": resp_tokens_text,  # list[str] — per token text
        "per_step_top": per_step_top,  # list[list[dict{label, delta, token_id}]]
        "base_text": base_text,  # full base completion (pretty)
        }
Path(OUT_JSON).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote delta-logit data → {OUT_JSON}")
