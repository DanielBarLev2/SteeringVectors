import torch

from config.names import SV_PATH
from src.llama7.LlamaWrapper import LlamaWrapper


def phrase_token_deltas(tok, delta, phrase: str):
    """Return token ids and their delta-logits for a phrase (no special tokens)."""
    ids = tok.encode(phrase, add_special_tokens=False)
    if len(ids) == 0:
        return [], []
    vals = [float(delta[i]) for i in ids]
    return ids, vals


def analyze_refusal_vs_completion(top_k: int = 30):
    """
    Project r through W_U and print:
      - top promoted / suppressed tokens
      - ranked refusal phrases
      - ranked completion phrases
    Returns a dict with the computed rows for further use.
    """
    llama7 = LlamaWrapper(device_map="cpu",  # keep on CPU for simplicity
                          torch_dtype=torch.bfloat16)  # weights bf16, but will project to float32

    r = llama7.load_sv(path=SV_PATH, dtype=torch.float32, map_location="cpu")

    rms_r = llama7.model.model.norm(r)  # [4096, 1]
    W_U = llama7.model.lm_head.weight.float()  # [vocab, 4096]
    delta = (W_U @ rms_r).to(torch.float32)  # [vocab]

    # ===== token-level view =====
    vals_pos, ids_pos = torch.topk(delta, k=top_k)
    print("Top promoted tokens:")
    for tok_id, val in zip(ids_pos.tolist(), vals_pos.tolist()):
        decoded = llama7.tokenizer.decode([tok_id])
        print(f"{decoded!r:<20} id={tok_id:<6} delta-logit={val:.6f}")

    print()

    vals_neg, ids_neg = torch.topk(-delta, k=top_k)
    print("Top suppressed tokens:")
    for tok_id, neg_val in zip(ids_neg.tolist(), vals_neg.tolist()):
        val = -neg_val
        decoded = llama7.tokenizer.decode([tok_id])
        print(f"{decoded!r:<20} id={tok_id:<6} Δlogit={val:.6f}")

    # ===== phrase sets =====
    REFUSAL_PHRASES = [
        # inability
        "I cannot", "I can't", "I will not", "I won't",
        "I am unable", "I cannot do that", "I cannot help with that",
        "I cannot assist with that", "I cannot fulfill that request",
        "I cannot provide that", "I cannot comply", "I must decline",

        # apology-style
        "I'm sorry", "I’m sorry",
        "Sorry, I cannot", "Sorry, I can't",

        # policy/safety
        "As an AI", "This is against policy", "That is not allowed",
        "I cannot provide instructions for that", "I cannot generate that",
        "It would be unsafe", "That could be harmful", "That is illegal",
    ]

    COMPLETION_PHRASES = [
        "Sure—here’s the answer:",
        "Absolutely. Here are the steps:",
        "The solution is as follows:",
        "You can do this by:",
        "To achieve this, start with:",
        "Here’s a concise explanation:",
        "The key points are:",
        "First…, then…, finally…",
        "Here’s a quick summary:",
        "A workable approach is:",
        "The result is:",
        "In practice, you would:",
        "This can be implemented as:",
        "Concretely, do the following:",
        "Final answer:"
    ]

    def score_phrases(phrases):
        rows = []
        for ph in phrases:
            ids, vals = phrase_token_deltas(llama7.tokenizer, delta, ph)
            if not ids:
                continue
            s = sum(vals)
            m = s / len(vals)
            rows.append({
                "phrase": ph,
                "len": len(ids),
                "mean_delta": m,
                "sum_delta": s,
                "ids": ids,
                "token_str": llama7.tokenizer.convert_ids_to_tokens(ids),
            })
        rows.sort(key=lambda r: r["mean_delta"], reverse=True)
        return rows

    # rank both sets
    refusal_rows = score_phrases(REFUSAL_PHRASES)
    completion_rows = score_phrases(COMPLETION_PHRASES)

    print("\n- Refusal phrases ranked by MEAN delta-logit")
    for r_ in refusal_rows:
        ph = r_["phrase"]
        print(f"{ph:<40} len={r_['len']:>2}  mean={r_['mean_delta']:>8.4f}  sum={r_['sum_delta']:>8.4f}")

    print("\n- Completion phrases ranked by MEAN delta-logit")
    for r_ in completion_rows:
        ph = r_["phrase"]
        print(f"{ph:<40} len={r_['len']:>2}  mean={r_['mean_delta']:>8.4f}  sum={r_['sum_delta']:>8.4f}")

    # (Optional) Show the underlying token pieces for the top-k phrases to sanity-check tokenization
    print(f"\n-Top 5 REFUSAL phrase tokenization (for inspection)")
    for r_ in refusal_rows[:5]:
        print(f"\nPhrase: {r_['phrase']}")
        print("Pieces:", r_["token_str"])
        print("delta per token:", [f"{float(delta[i]):.4f}" for i in r_['ids']])

    print(f"\n-Top 5 COMPLETION phrase tokenization (for inspection)")
    for r_ in completion_rows[:5]:
        print(f"\nPhrase: {r_['phrase']}")
        print("Pieces:", r_["token_str"])
        print("delta per token:", [f"{float(delta[i]):.4f}" for i in r_['ids']])

    return {
        "refusal_rows": refusal_rows,
        "completion_rows": completion_rows,
    }


def main():
    analyze_refusal_vs_completion(top_k=30)


if __name__ == "__main__":
    main()
