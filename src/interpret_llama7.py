import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.names import *


def load_vector(path: str) -> torch.Tensor:
    """Load a normalized steering vector from disk."""
    vec = torch.load(path, map_location=DEVICE)
    return vec.to(DEVICE)


def vocab_proj(vector: torch.Tensor,
               model: AutoModelForCausalLM,
               tokenizer: AutoTokenizer,
               top_k: int = 20):
    """
    Project the steering vector into vocabulary space via the unembedding head.
    Returns the top_k tokens and their scores.
    """
    with torch.no_grad():
        # Adds a batch dimension
        v_normed = model.model.norm(vector.unsqueeze(0))
        # Project to vocab
        logits = model.lm_head(v_normed)
        values, indices = torch.topk(logits, top_k, dim=-1)
        tokens = [tokenizer.decode([idx]) for idx in indices[0].tolist()]
        scores = values[0].tolist()
    return list(zip(tokens, scores))



if __name__ == '__main__':
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7B,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model.eval()

    # Example vector
    r_hat = torch.load(REFUSAL_NV_PATH + f'/vec_layer_{LAYER_IDX}_Llama-2-7b-chat-hf.pt', weights_only=True)
    r_hat = r_hat.to(dtype=model.dtype, device=DEVICE)

    # VocabProj
    vp_results = vocab_proj(r_hat, model, tokenizer, top_k=20)
    print("VocabProj top tokens:")
    for tok, score in vp_results:
        print(f"  {tok}: {score:.4f}")
