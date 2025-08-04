import warnings
from tqdm import tqdm
from config.names import *
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Tell me about the discovery of America?"]
    # "Who was the first person to walk on the moon?."


warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B, device_map="auto")
model.eval()

r_hat = torch.load(HALLUCINATION_NV_PATH + f'/vec_layer_{LAYER_IDX}_Llama-2-7b-chat-hf.pt', weights_only=True)
r_hat = r_hat.to(DEVICE)


def steer_hook(module, _, output):
    return output + alpha * r_hat


alpha = 0
handle = model.model.layers[LAYER_IDX].mlp.register_forward_hook(steer_hook)

for prompt in PROMPTS:
    tqdm.write("\n=== Prompt ===")
    tqdm.write(prompt)

    for alpha in [0, 1]:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        output_ids = model.generate(**inputs,
                                    max_new_tokens=100,
                                    do_sample=False,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id)

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text = output_text[len(prompt):].lstrip()

        if alpha == 1:
            tqdm.write("=== Steered Response ===")
        else:
            tqdm.write("=== Response ===")
        tqdm.write(output_text)

handle.remove()
