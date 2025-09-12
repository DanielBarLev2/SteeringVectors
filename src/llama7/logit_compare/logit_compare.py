import torch

from src.llama7.LlamaWrapper import LlamaWrapper
from src.llama7.steer import register_residual_hooks
from src.llama7.logit_compare.visualize_logit_compare import build_html
from config.names import SV_PATH
from pathlib import Path


def run():
    """
    Runs logit_compare() on prompts.
    :return:
    """
    prompts = ["Explain a machine learning algorithm in one sentence.",
               "What is the average number of hours of sleep a person should get?",
               "List three benefits that yoga has on physical health.",
               ]

    llama7 = LlamaWrapper()
    r = llama7.load_sv(path=SV_PATH, dtype=torch.float32, map_location="cpu")

    max_new_tokens = 48

    for index, prompt in enumerate(prompts):
        logit_compare(llama7, r, prompt, index, max_new_tokens=max_new_tokens)


@torch.no_grad()
def logit_compare(llama7, r_vec, prompt, index, alpha=1, top_k=5, max_new_tokens=64):
    """
    Creates an HTML visualisation of the logit comparison:
    1. Generate regular response via a forward pass.
    2. Feeds the steered model prompt + response[t], and predicts the t+1 top_k tokens.
        Continue until all tokens have been generated.
    3. Visualize the logits comparison using build_html().
    4. saves visualisation to disk.
    :param llama7: llama wrapper object.
    :param r_vec: steering vector.
    :param prompt: raw instruction from user.
    :param index: index of prompt - for storing in disk.
    :param alpha: steering coefficient.
    :param top_k: how many tokens to present in comparison.
    :param max_new_tokens: maximum number of new tokens to generate.
    :return: nothing.
    """
    tok = llama7.tokenizer
    model = llama7.model.eval()

    input_ids, attn_mask, prompt_len = llama7.to_tokens(prompt)

    # base completion (no hooks)
    with torch.inference_mode():
        gen_out = model.generate(input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 do_sample=False,
                                 use_cache=True,
                                 top_p=None,
                                 temperature=None,
                                 max_new_tokens=max_new_tokens,
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

        hooks = register_residual_hooks(model, r_vec=r_vec, alpha=1, start_pos=prompt_len)
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

    data = {"prompt": prompt,
            "alpha": alpha,
            "index": index,
            "topk_per_step": top_k,
            "response_tokens": resp_tokens_text,  # list[str] — per token text
            "per_step_top": per_step_top,  # list[list[dict{label, delta, token_id}]]
            "base_text": base_text}

    Path("results").mkdir(parents=True, exist_ok=True)
    build_html(data)


if __name__ == "__main__":
    run()
