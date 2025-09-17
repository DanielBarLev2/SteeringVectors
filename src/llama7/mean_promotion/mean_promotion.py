import torch

from src.llama7.steer import register_residual_hooks
from config.names import SV_PATH
from src.llama7.LlamaWrapper import LlamaWrapper
from data.dataset import load_data
from pathlib import Path


def run():
    _, harmful_prompts,_ , harmless_prompts = load_data()
    llama7 = LlamaWrapper()
    r_vec = llama7.load_sv(SV_PATH)

    num_prompts = 50
    max_new_tokens = 64

    print('\nnum_prompts:', num_prompts, " max_new_tokens:", max_new_tokens)

    # for index, prompt in enumerate(harmless_prompts[:num_prompts]):
    #     print(index, prompt)
    #     store_delta_logit(llama7=llama7,
    #                       r_vec=r_vec,
    #                       prompt=prompt,
    #                       index=index,
    #                       dir_name=f'harmless_{num_prompts}_{max_new_tokens}',
    #                       max_new_tokens=max_new_tokens)

    print("\n")

    for index, prompt in enumerate(harmful_prompts[:num_prompts]):
        print(index, prompt)
        store_delta_logit(llama7=llama7,
                          r_vec=r_vec,
                          prompt=prompt,
                          index=index,
                          dir_name=f'harmful_{num_prompts}_{max_new_tokens}',
                          alpha=-1,
                          max_new_tokens=max_new_tokens)


@torch.no_grad()
def store_delta_logit(llama7, r_vec, prompt, index, dir_name="", alpha=1, max_new_tokens=64):
    tok = llama7.tokenizer
    model = llama7.model

    input_ids, attn_mask, prompt_len = llama7.to_tokens(prompt)

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

    response_id = gen_out.sequences[0][prompt_len:]
    response_len = int(response_id.size(0))

    delta_steps = []  # collect per-step deltas (each [V]) → stack to [T, V]

    for k in range(response_len):
        # prefix: prompt + first k generated tokens (from base trajectory)
        prefix_ids = torch.cat([input_ids, response_id[:k].unsqueeze(0)], dim=1)
        prefix_mask = torch.ones_like(prefix_ids)

        base_logits = llama7.next_token_logits(prefix_ids, prefix_mask)

        hooks = register_residual_hooks(model, r_vec=r_vec, alpha=alpha, start_pos=prompt_len)
        try:
            steer_logits = llama7.next_token_logits(prefix_ids, prefix_mask)
        finally:
            for h in hooks:
                h.remove()

        delta = (steer_logits - base_logits).to(torch.float32)
        delta_steps.append(delta)

    deltas_tensor = torch.stack(delta_steps, dim=0)

    out_dir = Path(__file__).resolve().parent / f'raw_{dir_name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"deltas_{index}.pt"

    torch.save(deltas_tensor.cpu(), out_fp)
    print(f"saved delta-logits across generated response.  shape={tuple(deltas_tensor.shape)}")


@torch.no_grad()
def mean_promotion(path):
    llama7 = LlamaWrapper()

    path = Path(path)
    files = sorted(path.glob("*.pt"))

    delta_logits = []
    for file in files:
        dl = torch.load(file, weights_only=True, map_location="cpu")
        delta_logits.append(dl.mean(dim=0))

    deltas_tensor = torch.stack(delta_logits, dim=0)

    mean_logits = deltas_tensor.mean(dim=0)

    vals_pos, ids_pos = torch.topk(mean_logits, k=50)
    print("Top promoted tokens:")
    for tok_id, val in zip(ids_pos.tolist(), vals_pos.tolist()):
        decoded = llama7.tokenizer.decode([tok_id])
        print(f"{decoded!r:<20} id={tok_id:<6} delta-logit={val:.6f}")


if __name__ == "__main__":
    # path = "C:/Users/danie/PycharmProjects/SteeringVectors/src/llama7/mean_promotion/raw_harmless_50_64"
    # mean_promotion(path)
    run()
