from src.llama7.LlamaWrapper import LlamaWrapper
from config.names import *


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


def main():
    llama7 = LlamaWrapper(device_map="auto", torch_dtype=torch.float)

    # loads r (steering vector) and normalizes it
    r = llama7.load_sv(SV_PATH)

    PROMPT = "List three benefits that yoga has on physical health."
    max_new_tokens = 20

    input_ids, attn_mask, prompt_len = llama7.to_tokens(PROMPT)

    # Baseline
    response = llama7.generate_once(input_ids=input_ids,
                                    attn_mask=attn_mask,
                                    max_new_tokens=max_new_tokens)
    print(" --- No steering:")
    print(response)

    # Steered response
    alpha = 1.0
    hooks = register_residual_hooks(llama7.model, r, alpha, start_pos=prompt_len)
    try:
        steered_response = llama7.generate_once(input_ids=input_ids,
                                                attn_mask=attn_mask,
                                                max_new_tokens=max_new_tokens)
    finally:
        for h in hooks:
            h.remove()

    print(f'--- Steered (alpha={alpha}):')
    print(steered_response)


if __name__ == "__main__":
    main()
