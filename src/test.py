import torch
from config.names import *
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B, torch_dtype=torch.float16)
llama_model.to(DEVICE).eval() # Moves to GPU, disables dropout

# --- forward hook
LAYER_IDX = 14

activations = {"pos": [], "neg": []}
current_tag = None

def hook_fn(module, inp, out):
    """
    out shape: [batch, seq_len, hidden_dim]
    take only the **last** token’s activation in each batch
    """

    last_hidden = out[:, -1, :].detach().cpu()
    activations[current_tag].append(last_hidden)

layer = llama_model.model.layers[LAYER_IDX].mlp  # adjust if path differs
hook_handle = layer.register_forward_hook(hook_fn)