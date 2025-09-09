from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from config.names import *
from pathlib import Path


def load_sv(path: Path) -> torch.Tensor:
    """
    loads a steering vector from a path, normalize it and allocate contiguous space in memory.
    :param path: path to steering vector.
    :return: tensor of size [4096]
    """
    r = torch.load(str(path), weights_only=True, map_location="cpu")
    r /= r.norm()

    return r.contiguous()


def reformat_prompt(tokenizer, user_text: str) -> str:
    """
    Reformats a prompt. Adds chat template and assign role.
    Format: "[INST] {prompt} [/INST] "
    :param tokenizer: llama-2 tokenizer.
    :param user_text: raw instruction from user.
    :return: reformated prompt
    """
    massage = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(massage, tokenize=False, add_generation_prompt=True)


def register_residual_hooks(model, r_vec: torch.Tensor, alpha: float, start_pos: int):
    """
    Install forward hooks that add alpha * r_vec to the residual stream output
     of every decoder layer, but only at generated token positions.

    Steering via activation addition: inject a fixed direction r_vec into the
     layer outputs that flow through the transformer stack (residual stream).
    By masking out the prompt positions, we leave the prompt untouched and steer
     only the tokens produced by decoding.

    Register a PyTorch forward hook on each LlamaDecoderLayer:
    On each forward pass the hook receives the layer output hs with shape (B, T, H) during the prefill step,
     and (B, 1, H) later on during incremental decoding with KV cache.
    Build a mask that is zero on prompt tokens and one on generated tokens:
        - Prefill (first pass): T = prompt_len, so all zeros (no steering).
        - Decoding steps: T = 1, so it is always a generated position → ones.
    Compute delta = alpha * r_vec (broadcast to (1, T, H)) and add it to hs at masked positions,
     then return the modified tensor as the new layer output.

    :param model: llama-2 model.
    :param r_vec: steering vector.
    :param alpha: steering scalar.
    :param start_pos: Number of non-pad tokens.
    :return: hooks : list[torch.utils.hooks.RemovableHandle] -> Handles for all installed hooks.
    """
    hooks = []

    # broadcast to (B, T, H)
    r_vec = r_vec.view(1, 1, -1)

    def make_hook(a):
        def hook_fn(module, inputs, output):
            # output could be Tensor or (Tensor, *rest)
            if isinstance(output, tuple):
                hs, *rest = output
            else:
                hs, rest = output, None

            # hs shape is (B, T, H) during the first pass (no cache),
            # and (B, 1, H) on subsequent cached steps.
            if hs.ndim == 3:
                B, T, H = hs.shape
                # mask so we only perturb generated tokens
                if T > 1:
                    mask = torch.zeros((1, T, 1), device=hs.device, dtype=hs.dtype)
                    mask[:, start_pos:, :] = 1
                else:
                    # during cached decoding, T=1 and it's a generated token
                    mask = torch.ones((1, 1, 1), device=hs.device, dtype=hs.dtype)
                delta = a * r_vec.to(device=hs.device, dtype=hs.dtype)
                hs = hs + mask * delta
            elif hs.ndim == 2:
                # rare path: (B, H)
                delta = a * r_vec.to(device=hs.device, dtype=hs.dtype).squeeze(1)
                hs = hs + delta
            else:
                # unexpected -> do nothing
                return output

            return (hs, *rest) if rest is not None else hs

        return hook_fn

    # apply hooks to all layer
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(make_hook(alpha)))

    return hooks


def to_tokens(tokenizer, user_text: str, device=DEVICE):
    """
    Reformats and tokenize a user prompt with the Llama-2 tokenizer.
    :param tokenizer: llama-2 tokenizer.
    :param user_text: raw instruction from user.
    :param device: model device.
    :return: 1. input_ids: Tokenized prompt IDs.
             2. attn_mask: Attention mask aligned with input_ids.
             3. prompt_len: Number of non-pad tokens.
                * steering vectors affect only new tokens.
    """
    text = reformat_prompt(tokenizer, user_text)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    prompt_len = int(attn_mask[0].sum().item())

    return input_ids, attn_mask, prompt_len


def generate_once(model, tokenizer, input_ids, attn_mask, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate one completion from Llama-2-7b-chat-hf using greedy decoding.
    :param model: llama-2-7b-chat-hf
    :param tokenizer: llama-2 tokenizer.
    :param input_ids: Tokenized prompt IDs.
    :param attn_mask: Attention mask aligned with input_ids.
    :param max_new_tokens:  Maximum number of new tokens to generate.
    :return: 1. decoded_text: The decoded response from Llama-2-7b-chat-hf.
             2. prompt_len: The tokenized prompt length (start position of generation)

    Notes: prompt_len is used downstream to apply steering only to generated tokens
           (positions >= prompt_len) while leaving the prompt unmodified.
    """
    with torch.inference_mode():
        out_ids = model.generate(input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 do_sample=False,
                                 temperature=None,
                                 top_p=None,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main():
    set_seed(0)

    # defines tokenizer and set pading
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # loads model and set to evaluation
    model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    # loads r (steering vector) and normalizes it
    r = load_sv(SV_PATH)
    r /= r.norm()

    PROMPT = "List three benefits that yoga has on physical health."
    input_ids, attn_mask, prompt_len = to_tokens(tokenizer, PROMPT, model.device)

    # Baseline
    response = generate_once(model, tokenizer, input_ids, attn_mask)
    print(" --- No steering:")
    print(response)

    # Steered response
    alpha = 1.0
    hooks = register_residual_hooks(model, r, alpha, start_pos=prompt_len)
    try:
        steered_response = generate_once(model, tokenizer, input_ids, attn_mask)
    finally:
        for h in hooks:
            h.remove()

    print(f'--- Steered (alpha={alpha}):')
    print(steered_response)


if __name__ == "__main__":
    main()
