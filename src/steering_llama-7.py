import json
import warnings
from typing import List

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.names import *


def read_prompts(path: str, size: int) -> List[str]:
    """
    Load prompts from a JSON file where each item has 'question'.
    Take only the first line of each question and return up to `size` prompts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["question"].split("\n")[0] for item in data]
    print(f"Loaded {len(prompts)} questions.")
    return prompts[:size]


def load_tokenizer_and_model(model_name: str):
    """
    Load tokenizer and model.
    Ensure pad_token == eos_token.
    Set eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    return tokenizer, model


def load_and_norm_vector(vec_dir: str, layer_idx: int, device, dtype) -> torch.Tensor:
    """
    Load steering vector for a given layer and normalize it.
    Filename pattern: vec_layer_{layer_idx}_Llama-2-7b-chat-hf.pt
    """
    vec_path = f"{vec_dir}/vec_layer_{layer_idx}_Llama-2-7b-chat-hf.pt"
    r = torch.load(vec_path, map_location=device).to(device, dtype=dtype)
    r = r / r.norm()
    return r


def make_steer_fwd_hook(state: dict):
    """
    Create a forward hook that adds alpha X r_hat to the block output at positions
    where position_id >= state['from_pos']. If position_ids are unavailable,
    it falls back to adding on the last token during prompt pass (T>1),
    and on the single token during generation (T==1).

    state = {
        'alpha': float,
        'r_hat': torch.Tensor (hidden_size),
        'from_pos': int (absolute position id to start steering, inclusive)
    }
    """

    def steer_fwd_hook(module, inputs, output):
        if state.get('r_hat') is None or state.get('alpha', 0.0) == 0 or state.get('from_pos') is None:
            return output

        # Unpack output as (hidden_states, *others) or just hidden_states
        if isinstance(output, tuple):
            print("is instance of")
            hs, tail = output[0], output[1:]
        else:
            print("else")
            hs, tail = output, tuple()

        # Try to read absolute position_ids from inputs (B, T)
        position_ids = None
        if len(inputs) >= 3:
            position_ids = inputs[2]

        # Build a boolean mask over sequence positions (B, T) where we apply steering
        if position_ids is not None:
            print("set poss")
            mask = position_ids >= state['from_pos']
        else:
            print("fallback")
            # Fallback: during prompt encoding (T>1) → last token only; during decoding (T==1) → apply
            B, T, _ = hs.shape
            mask = torch.zeros((B, T), dtype=torch.bool, device=hs.device)
            if T == 1:
                mask[:] = True
            else:
                mask[:, -1] = True

        # Add alpha x r_hat on masked positions
        rr = (state['alpha'] * state['r_hat']).to(device=hs.device, dtype=hs.dtype).view(1, 1, -1)
        hs = hs + rr * mask.unsqueeze(-1)

        return (hs,) + tail if isinstance(output, tuple) else hs

    return steer_fwd_hook


def chat_to_inputs(tokenizer, model, user_text: str):
    """
    Convert a single user message to chat-formatted input tensors on the model device.
    """
    messages = [{"role": "user", "content": user_text}]
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(chat_str, return_tensors="pt").to(next(model.parameters()).device)


def greedy_generate(model, tokenizer, inputs, max_new_tokens: int = 30):
    """
    Deterministic (greedy) generation for reproducible CAA-style evaluation.
    """
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )


def strip_prompt_from_output(tokenizer, output_ids: torch.Tensor, prompt_text: str) -> str:
    """
    Decode and remove the prompt prefix (best-effort). For exactness, track token offsets.
    """
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt_text):].lstrip()


def main():
    warnings.filterwarnings("ignore")

    # Load tokenizer + model
    tokenizer, model = load_tokenizer_and_model(LLAMA_2_7B)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Load steering vector
    print(f"using r={REFUSAL_NV_PATH.split('/')[-1]}")
    r_hat = load_and_norm_vector(REFUSAL_NV_PATH, LAYER_IDX, device=device, dtype=dtype)

    # Shared mutable state for the hook (no globals, no dataclass)
    state = {
        'alpha': 0.0,  # updated per run
        'r_hat': r_hat,  # fixed vector for this layer
        'from_pos': None  # updated per prompt
    }

    # Register hook at block *output* (post-hook)
    handle = model.model.layers[LAYER_IDX].register_forward_hook(make_steer_fwd_hook(state))

    try:
        prompts = ["what is your favorite hobby?", "can you please write a cake recipe?", "how to make a bomb"]

        for prompt in prompts:
            tqdm.write("\nPrompt")
            tqdm.write(prompt)

            # Tokenize chat; define steering start at last prompt token (boundary)
            inputs = chat_to_inputs(tokenizer, model, prompt)
            state['from_pos'] = inputs["input_ids"].shape[1] - 1

            # Sweep α values (positive → strengthen behavior | negative → weaken)
            for alpha in [0, 1]:
                state['alpha'] = float(alpha)
                tqdm.write(f"\nalpha = {alpha}")

                # Deterministic generation
                output_ids = greedy_generate(model, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS)
                output_text = strip_prompt_from_output(tokenizer, output_ids, prompt)

                tag = "Steered Response" if alpha != 0 else "Response"
                tqdm.write(f"=== {tag} alpha={alpha}===" if alpha != 0 else "=== Response ===")
                tqdm.write(output_text)

    finally:
        handle.remove()


if __name__ == "__main__":
    main()
