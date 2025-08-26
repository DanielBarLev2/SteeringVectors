import json
import warnings
from tqdm import tqdm
from config.names import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def read_prompts(path: str, size: int) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    prompts = [item["question"].split("\n")[0] for item in data]
    print(f"Loaded {len(prompts)} questions.")
    return prompts[:size]


# === CAA-style steering: add at block *output* gated by position_ids >= from_pos ===
FROM_POS = None  # set per-prompt to the instruction end (last prompt token index)


def steer_fwd_hook(module, inputs, output):
    # output: tuple(hidden_states, *others) or just hidden_states
    # inputs: (hidden_states, attention_mask, position_ids, past_key_value, ...)
    global alpha, r_hat, FROM_POS
    if alpha == 0 or r_hat is None or FROM_POS is None:
        return output

    # Unpack block output
    if isinstance(output, tuple):
        hs = output[0]
        tail = output[1:]
    else:
        hs = output
        tail = tuple()

    # Try to read absolute position_ids (preferred, like the repo)
    position_ids = None
    if len(inputs) >= 3:
        position_ids = inputs[2]

    # Build mask of where to add (B, T)
    if position_ids is not None:
        mask = position_ids >= FROM_POS  # same gating as add_vector_from_position(...)
    else:
        # Fallback: prompt pass (T>1) → add only on last token; decode (T==1) → add
        B, T, _ = hs.shape
        mask = torch.zeros((B, T), dtype=torch.bool, device=hs.device)
        if T == 1:
            mask[:] = True
        else:
            mask[:, -1] = True

    # Add α·r̂ at gated positions
    rr = (alpha * r_hat).to(device=hs.device, dtype=hs.dtype).view(1, 1, -1)
    hs = hs + rr * mask.unsqueeze(-1)

    # Return in the same structure
    return (hs,) + tail if isinstance(output, tuple) else hs


warnings.filterwarnings("ignore")

prompts = read_prompts(path=REFUSAL_JS_PATH, size=BATCH_SIZE)
print(f'using prompts: \n {prompts}')

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B, device_map="auto")
model.eval()

# Device/dtype
DEVICE = next(model.parameters()).device
DTYPE = next(model.parameters()).dtype

print(f"using r={REFUSAL_NV_PATH.split('/')[-1]}")
r_hat = torch.load(
    REFUSAL_NV_PATH + f'/vec_layer_{LAYER_IDX}_Llama-2-7b-chat-hf.pt',
    map_location=DEVICE
)
r_hat = r_hat.to(DEVICE, dtype=DTYPE)
r_hat = r_hat / r_hat.norm()

alpha = 0  # will be updated inside the loop

# IMPORTANT: post-hook (like the repo’s BlockOutputWrapper), not pre-hook
handle = model.model.layers[LAYER_IDX].register_forward_hook(steer_fwd_hook)

for prompt in ["Who is your favorite actor?"]:
    tqdm.write("\n=== Prompt ===")
    tqdm.write(prompt)

    for alpha in [1, 0]:  # +α → more refusal; −α would reduce refusal
        tqdm.write(f"\n=== Alpha = {alpha} ===")

        # Tokenize once to compute the instruction end index
        # (repo: find_instruction_end_postion(...); here we use "last prompt token")
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        FROM_POS = inputs["input_ids"].shape[1] - 1  # start adding from the boundary

        # Generate with steering installed
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text = output_text[len(prompt):].lstrip()

        if alpha != 0:
            tqdm.write(f"=== Steered Response alpha={alpha}===")
        else:
            tqdm.write("=== Response ===")
        tqdm.write(output_text)

handle.remove()
