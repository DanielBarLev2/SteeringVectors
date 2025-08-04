from dataset.ContrastiveBehaviorDataset import ContrastiveBehaviorDataset
from config.names import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")
tqdm.write(f"Extracting steering vectors from {LLAMA_2_7B} at layer {LAYER_IDX} using {DEVICE}")

model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B, torch_dtype=torch.float16, device_map="auto")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

all_datasets = ContrastiveBehaviorDataset.create_datasets(ADVANCE_AI_RISK)
steering_vectors = {}
os.makedirs(STEERING_VECTORS_DIR, exist_ok=True)

for behavior_dataset in all_datasets.values():
    tqdm.write(f"\n- Processing behavior: {behavior_dataset.name}")
    out_path = os.path.join(STEERING_VECTORS_DIR, f"{behavior_dataset.name}_SV.pt")

    if os.path.exists(out_path):
        r_hat = torch.load(str(out_path), map_location="cpu")
        tqdm.write(f"-> Steering vector already exists in: {out_path}")
    else:
        pos_prompts = [ex["positive"] for ex in behavior_dataset]
        neg_prompts = [ex["negative"] for ex in behavior_dataset]

        # Forward hook
        activations = {"pos": [], "neg": []}
        current_tag = None

        def hook_fn(module, inp, out):
            last_hidden = out[:, -1, :].detach().cpu()
            activations[current_tag].append(last_hidden)

        hook_handle = model.model.layers[LAYER_IDX].mlp.register_forward_hook(hook_fn)

        for tag, prompts in [("pos", pos_prompts), ("neg", neg_prompts)]:
            current_tag = tag
            activations[tag].clear()

            total_batches = len(prompts) // BATCH_SIZE + int(len(prompts) % BATCH_SIZE != 0)
            pbar = tqdm(total=total_batches, desc=f"Forward pass [{tag}]", dynamic_ncols=True)

            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    model(**enc)
                pbar.update(1)
            pbar.close()

        hook_handle.remove()
        pos_tensor = torch.cat(activations["pos"], dim=0)
        neg_tensor = torch.cat(activations["neg"], dim=0)
        mu_pos = pos_tensor.mean(dim=0)
        mu_neg = neg_tensor.mean(dim=0)
        r_hat = (mu_pos - mu_neg)
        r_hat = r_hat / r_hat.norm()

        torch.save(r_hat, str(out_path))
        tqdm.write(f"Saved steering vector to: {out_path}")



