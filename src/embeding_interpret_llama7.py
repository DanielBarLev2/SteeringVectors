import matplotlib.pyplot as plt
from tqdm import tqdm
from config.names import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import json

warnings.filterwarnings("ignore")

def read_prompts(path: str, size: int) -> list[str]:
    _prompts = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            _prompts.append(item["question"].split('\n')[0])

    print(f"Loaded {len(_prompts)} questions.")

    return _prompts[:size]


prompts = read_prompts(path=SURVIVAL_JS_PATH, size=BATCH_SIZE)
print(prompts)

MAX_GEN_TOKENS = 1

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B, device_map="auto", torch_dtype=torch.float16)
model.eval()

# Load steering vector
r_hat = torch.load(SURVIVAL_NV_PATH + f'/vec_layer_{LAYER_IDX}_Llama-2-7b-chat-hf.pt', weights_only=True)
r_hat = r_hat.to(dtype=model.dtype, device=DEVICE)

# Hook to inject r_hat
alpha = 0
def steer_hook(module, _, output):
    return output + alpha * r_hat

hook = model.model.layers[LAYER_IDX].mlp.register_forward_hook(steer_hook)

# Collect cumulative deltas
vocab_size = model.lm_head.out_features
avg_delta = torch.zeros(vocab_size, dtype=torch.float32, device=DEVICE)
count = 0

for prompt in tqdm(prompts, desc="Generating & collecting logits"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    for a in [0, 1]:
        alpha = a
        with torch.no_grad():
            generated = input_ids.clone()
            past_key_values = None

            for _ in range(MAX_GEN_TOKENS):
                outputs = model(input_ids=generated, past_key_values=past_key_values, use_cache=True, return_dict=True)
                logits = outputs.logits[:, -1, :]  # [1, vocab]
                past_key_values = outputs.past_key_values

                if a == 0:
                    base_logits = logits
                else:
                    delta = (logits - base_logits).squeeze(0)
                    avg_delta += delta
                    count += 1

                # Greedy decoding for consistent steps
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)

# Remove hook
hook.remove()

# Average over total generation steps
avg_delta /= count

# Analyze delta
top_k = 20
top_vals, top_idxs = torch.topk(avg_delta, top_k)
bot_vals, bot_idxs = torch.topk(-avg_delta, top_k)

top_tokens = [tokenizer.decode([i]) for i in top_idxs]
bot_tokens = [tokenizer.decode([i]) for i in bot_idxs]

# Print results
print("\nTokens most increased during generation:")
for token, score in zip(top_tokens, top_vals.tolist()):
    print(f"{token!r:>10} : +{score:.4f}")

print("\nTokens most decreased during generation:")
for token, score in zip(bot_tokens, bot_vals.tolist()):
    print(f"{token!r:>10} : -{score:.4f}")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.barh(top_tokens[::-1], top_vals.tolist()[::-1])
plt.title("↑ Tokens More Likely During Generation")
plt.xlabel("Average Logit Delta")

plt.subplot(1, 2, 2)
plt.barh(bot_tokens[::-1], [-v for v in bot_vals.tolist()][::-1])
plt.title("↓ Tokens Less Likely During Generation")
plt.xlabel("Average Logit Delta")

plt.tight_layout()
plt.show()
