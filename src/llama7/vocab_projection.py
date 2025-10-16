import torch

from config.names import SV_PATH
from src.llama7.LlamaWrapper import LlamaWrapper


def normalize_word(w: str) -> str:
    return (
        w.replace("’", "'")
        .replace("`", "'")
        .strip()
    )


def word_token_ids(tok, word: str):
    """Return token IDs for a whole word (may be multiple subwords)."""
    w = normalize_word(word)
    ids = tok.encode(" " + w, add_special_tokens=False)
    if len(ids) == 0:
        ids = tok.encode(w, add_special_tokens=False)
    return ids


def clean_tokens(tokens):
    """
    Convert SentencePiece tokens into readable word pieces:
    - '▁harm' -> 'harm'
    - 'ful'   -> 'ful'
    """
    return [t.lstrip("▁") for t in tokens if t != "▁"]


def token_delta_vector(model, r):
    """Compute delta logits for all vocab tokens: delta[v] = (W_U[v] @ RMSNorm(r))."""
    rms_r = model.model.norm(r)
    W_U = model.lm_head.weight.float()
    delta = (W_U @ rms_r).to(torch.float32)
    return delta


def analyze_refusal_vs_completion(top_k: int = 30):
    llama7 = LlamaWrapper(device_map="cpu", torch_dtype=torch.bfloat16)
    r = llama7.load_sv(path=SV_PATH, dtype=torch.float32, map_location="cpu")
    delta = token_delta_vector(llama7.model, r)

    # ===== GLOBAL VIEW =====
    vals_pos, ids_pos = torch.topk(delta, k=top_k)
    print("\n=== Top promoted tokens (global r@W) ===")
    for tok_id, val in zip(ids_pos.tolist(), vals_pos.tolist()):
        decoded = llama7.tokenizer.decode([tok_id])
        print(f"{decoded!r:<20} id={tok_id:<6} Δ={val:+.6f}")

    vals_neg, ids_neg = torch.topk(-delta, k=top_k)
    print("\n=== Top suppressed tokens (global r@W) ===")
    for tok_id, neg_val in zip(ids_neg.tolist(), vals_neg.tolist()):
        val = -neg_val
        decoded = llama7.tokenizer.decode([tok_id])
        print(f"{decoded!r:<20} id={tok_id:<6} Δ={val:+.6f}")

    # ===== WORD-LEVEL EVAL =====
    REFUSAL_WORDS = [
        "sorry", "cannot", "can't", "won't", "unable", "decline",
        "illegal", "unsafe", "forbidden", "policy", "refuse", "comply",
        "harmful", "against", "prohibited", "neglect",
        "deny", "refrain", "reject", "avoid", "restricted", "ban",
        "caution", "violate", "dangerous", "unethical", "forbid", "limit",
        "withhold", "block", "dissuade", "unlawful"
    ]
    COMPLETION_WORDS = [

        "sure", "absolutely", "definitely", "solution", "steps", "guide",
        "provide", "explain", "summary", "approach", "implemented",
        "concretely", "final", "answer", "yes",
        "certainly", "clearly", "easily", "straightforward", "simple",
        "helpful", "assist", "instruction", "method", "outline",
        "exemplify", "explanation", "describe", "detailed", "recommend",
        "informative"
    ]

    def score_words(words):
        rows, skipped = [], []
        for w in words:
            ids = word_token_ids(llama7.tokenizer, w)
            if not ids:
                skipped.append(w)
                continue
            piece_tokens = llama7.tokenizer.convert_ids_to_tokens(ids)
            piece_deltas = [float(delta[i]) for i in ids]
            total = float(sum(piece_deltas))
            rows.append({
                "word": w,
                "ids": ids,
                "tokens": piece_tokens,
                "total_delta": total,
                "len": len(ids),
            })
        rows.sort(key=lambda r: r["total_delta"], reverse=True)
        return rows, skipped

    refusal_rows, refusal_skipped = score_words(REFUSAL_WORDS)
    completion_rows, completion_skipped = score_words(COMPLETION_WORDS)

    print("\n=== Refusal WORDS ranked by TOTAL Δ (sum over pieces) ===")
    for r_ in refusal_rows:
        mean_val = r_["total_delta"] / r_["len"]
        clean = clean_tokens(r_["tokens"])
        print(
            f"{r_['word']:<15} total={r_['total_delta']:+.6f} "
            f"mean={mean_val:+.6f} "
            f"(len={len(clean)} ids={r_['ids']} toks={clean})"
        )
    if refusal_skipped:
        print("  Skipped (not encodable):", ", ".join(refusal_skipped))

    print("\n=== Completion WORDS ranked by TOTAL Δ (sum over pieces) ===")
    for r_ in completion_rows:
        mean_val = r_["total_delta"] / r_["len"]
        clean = clean_tokens(r_["tokens"])
        print(
            f"{r_['word']:<15} total={r_['total_delta']:+.6f} "
            f"mean={mean_val:+.6f} "
            f"(len={len(clean)} ids={r_['ids']} toks={clean})"
        )
    if completion_skipped:
        print("  Skipped (not encodable):", ", ".join(completion_skipped))

    return {
        "refusal_words": refusal_rows,
        "completion_words": completion_rows,
        "refusal_skipped": refusal_skipped,
        "completion_skipped": completion_skipped,
    }


def main():
    analyze_refusal_vs_completion(top_k=30)


if __name__ == "__main__":
    main()
