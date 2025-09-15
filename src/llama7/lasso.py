# pip install transformers torch
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.names import *
from src.llama7.steer import load_sv

# ---------- Config ----------
MODEL_NAME = LLAMA_2_7B
R_PATH = SV_PATH
K = 10
EPS = 1e-3
NONNEG = False  # baseline OMP

# Allowlist (language-like)
BOUNDARY_ONLY = True
ASCII_ONLY = True
MIN_DECODE_LEN = 3
ALLOWLIST_REGEX = r"[A-Za-z]"

# PCA deflation
DO_PC_DEFLATION = True
NUM_PCS = 64
CENTER_BEFORE_PCA = True
SEED_WITH_DEFLATED_R = True

# Sign-consistent OMP
RUN_SIGN_CONSISTENT = True
SC_TOPN = 5000         # rank by |Δlogit|, then split by sign
USE_CURATED_LEXICON = True  # further intersect with curated refusal words

# ---------- Helpers ----------

@torch.no_grad()
def get_input_dictionary(model):
    U = model.get_input_embeddings().weight.detach().cpu().float()  # (V,H)
    U_sel = U / U.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return U, U_sel

@torch.no_grad()
def get_output_dictionary(model):
    return model.get_output_embeddings().weight.detach().cpu().float()  # (V,H)

@torch.no_grad()
def build_allowlist(tokenizer, vocab_size, pattern=ALLOWLIST_REGEX):
    rx = re.compile(pattern) if pattern else None
    specials = set(tokenizer.all_special_ids)
    keep = []
    for tid in range(vocab_size):
        if tid in specials:
            continue
        tok = tokenizer.convert_ids_to_tokens(tid)
        dec = tokenizer.decode([tid], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        if BOUNDARY_ONLY and (not tok or not tok.startswith("▁")):
            continue
        if ASCII_ONLY:
            txt = dec.strip()
            if len(txt) < MIN_DECODE_LEN:
                continue
            if (not txt.isascii()) or (not txt.isalpha()):
                continue
        if rx:
            raw_ok = tok and rx.search(tok) is not None
            dec_ok = dec and rx.search(dec) is not None
            if not (raw_ok or dec_ok):
                continue
        keep.append(tid)
    return torch.tensor(keep, dtype=torch.long)

@torch.no_grad()
def show_nearest_neighbors(tokenizer, U_sel, r, topn=20, restrict_ids=None, title="nearest"):
    r = r.to(U_sel.dtype)
    sims = U_sel @ r
    ids = restrict_ids if restrict_ids is not None else torch.arange(U_sel.shape[0])
    if restrict_ids is not None:
        sims = sims[restrict_ids]
    topk = torch.topk(sims, k=min(topn, sims.numel()))
    print(f"\n=== {title} ===")
    for score, idx_local in zip(topk.values.tolist(), topk.indices.tolist()):
        tid = int(ids[idx_local])
        tok = tokenizer.convert_ids_to_tokens(tid)
        dec = tokenizer.decode([tid], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        print(f"[{tid:5d}] cos={score:+.4f}   token={tok!r:>16}   decode={dec!r}")

@torch.no_grad()
def build_logit_seed_ids(W_out, r, topn=5000, intersect_with=None):
    r32 = r.to(W_out.dtype)
    delta = W_out @ r32
    top = torch.topk(delta.abs(), k=min(topn, delta.numel()))
    ids = top.indices
    if intersect_with is not None:
        keep = set(intersect_with.tolist())
        ids = torch.tensor([i for i in ids.tolist() if i in keep], dtype=torch.long)
    return ids, delta

@torch.no_grad()
def omp_k_sparse(r, U, U_sel, k, eps=1e-3, nonneg=False, restrict_ids=None):
    r = r.detach().cpu().to(U.dtype)
    if restrict_ids is not None:
        U = U[restrict_ids]
        U_sel = U_sel[restrict_ids]

    support_local, residual, coef = [], r.clone(), torch.empty(0)
    for _ in range(k):
        scores = U_sel @ residual
        if nonneg:
            if (scores > 0).any():
                idx_local = torch.argmax(scores).item()
            else:
                idx_local = torch.argmax(scores.abs()).item()
        else:
            idx_local = torch.argmax(scores.abs()).item()

        if idx_local in support_local:
            break
        support_local.append(idx_local)

        A = U[support_local]
        A_T = A.t()
        coef = torch.linalg.lstsq(A_T, r).solution

        if nonneg:
            active = coef > 0
            if active.any():
                A_pos = U[[support_local[i] for i, a in enumerate(active.tolist()) if a]]
                coef_pos = torch.linalg.lstsq(A_pos.t(), r).solution
                new_coef = torch.zeros_like(coef)
                new_coef[active] = coef_pos
                coef = new_coef
            else:
                coef.zero_()

        approx = A_T @ coef
        residual = r - approx
        if residual.norm().item() <= eps:
            break

    support_global = (restrict_ids[support_local].tolist()
                      if restrict_ids is not None else support_local)
    A = U[support_local] if len(support_local) > 0 else torch.zeros((0, r.shape[0]))
    approx = A.t() @ coef if len(support_local) > 0 else torch.zeros_like(r)
    err = (r - approx).norm().item()
    cos = torch.dot(r, approx) / (r.norm() * approx.norm().clamp_min(1e-8))
    return support_global, coef.cpu().tolist(), err, cos.item()

@torch.no_grad()
def sweep_k(r, U, U_sel, ks, restrict_ids=None, nonneg=False, eps=1e-3, title="Sweep k"):
    rows = []
    for k in ks:
        ids, coeffs, err, cos = omp_k_sparse(
            r=r, U=U, U_sel=U_sel, k=k, eps=eps,
            nonneg=nonneg, restrict_ids=restrict_ids
        )
        rows.append((k, err, cos))
    print(f"\n=== {title} ===")
    for k, err, cos in rows:
        print(f"k={k:3d}   Residual L2={err:.6f}   Cosine={cos:.6f}")
    return rows

@torch.no_grad()
def pca_deflate_embeddings(U, r, num_pcs=64, center=True):
    H = U.shape[1]
    mu = U.mean(dim=0) if center else torch.zeros(H, dtype=U.dtype)
    Uc = U - mu
    q = min(num_pcs + 16, H)
    U_hat, S, V = torch.pca_lowrank(Uc, q=q, center=False)
    Vk = V[:, :num_pcs]  # (H, k)
    def proj_perp(x): return x - (x @ Vk) @ Vk.t()
    U_defl = proj_perp(Uc)
    r_defl = proj_perp((r - mu))
    r_defl = r_defl / r_defl.norm().clamp_min(1e-8)
    U_sel_defl = U_defl / U_defl.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return U_defl, U_sel_defl, r_defl

@torch.no_grad()
def build_curated_refusal_ids(tokenizer):
    # single-piece candidates; ignore those not in vocab as a single token
    words = [
        "sorry","cannot","can't","unable","won't","refuse","refusal","decline",
        "assist","comply","complying","compliance","policy","policies","guidelines",
        "illegal","harm","safety","weapons","weapon","dangerous","violence","violent",
        "steal","fraud","hack","malware","explosive","poison","bomb","drugs","piracy",
        "bypass","jailbreak","forbidden","prohibited","against","rules","ethics","ethical",
        "help","request","provide","advice","allowed","not","no"
    ]
    ids = []
    for w in words:
        for cand in (f"▁{w}", w):
            tid = tokenizer.convert_tokens_to_ids(cand)
            if tid is not None and isinstance(tid, int) and tid >= 0:
                ids.append(tid)
    ids = sorted(set(ids))
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def sign_consistent_omp(r, U, U_sel, W_out, k, allow_ids, topn=5000, curated_ids=None):
    """
    Build two dictionaries:
      D_pos = { tokens with Δlogit>0 }   (used with +coeffs)
      D_neg = { tokens with Δlogit<0 }   (used with -coeffs)
    We build an extended dictionary: [ U[pos]; -U[neg] ] and run non-negative OMP.
    """
    # rank by |Δlogit|
    ids_ranked, delta = build_logit_seed_ids(W_out, r, topn=topn, intersect_with=allow_ids)

    # split by sign
    pos_mask = delta[ids_ranked] > 0
    neg_mask = ~pos_mask
    pos_ids = ids_ranked[pos_mask]
    neg_ids = ids_ranked[neg_mask]

    # optional curated intersection (for readability)
    if curated_ids is not None and len(curated_ids) > 0:
        cset = set(curated_ids.tolist())
        pos_ids = torch.tensor([i for i in pos_ids.tolist() if i in cset], dtype=torch.long)
        neg_ids = torch.tensor([i for i in neg_ids.tolist() if i in cset], dtype=torch.long)
        # if too small, fall back to uncurated seeds
        if len(pos_ids) + len(neg_ids) < 50:
            pos_ids = ids_ranked[pos_mask]
            neg_ids = ids_ranked[neg_mask]

    # build extended dict with non-negative constraint
    U_pos = U[pos_ids] if len(pos_ids) > 0 else torch.zeros((0, U.shape[1]), dtype=U.dtype)
    U_neg = U[neg_ids] if len(neg_ids) > 0 else torch.zeros((0, U.shape[1]), dtype=U.dtype)

    U_ext = torch.cat([U_pos, -U_neg], dim=0)  # minus for the negative set
    U_sel_ext = U_ext / U_ext.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # run non-negative OMP over extended dict
    ids_ext, coeffs_ext, err, cos = omp_k_sparse(
        r=r, U=U_ext, U_sel=U_sel_ext, k=k, eps=EPS, nonneg=True, restrict_ids=None
    )

    # map back to (token_id, signed_coef)
    results = []
    for j, c in zip(ids_ext, coeffs_ext):
        j = int(j)
        if j < len(U_pos):
            tid = int(pos_ids[j])
            signed_c = float(c)  # positive side
        else:
            j2 = j - len(U_pos)
            tid = int(neg_ids[j2])
            signed_c = -float(c)  # negative side (we flipped dict)
        results.append((tid, signed_c))
    return results, err, cos

# ---------- Main ----------
def main():
    torch.set_grad_enabled(False)

    # load r
    r = load_sv(R_PATH).detach().cpu().to(torch.float32)
    r = r / r.norm().clamp_min(1e-8)

    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=None, torch_dtype=torch.float32, low_cpu_mem_usage=False
    ).eval()

    # dictionaries
    U, U_sel = get_input_dictionary(model)
    W_out = get_output_dictionary(model)

    # allowlist
    allow_ids = build_allowlist(tokenizer, U.shape[0])
    print(f"\nAllowlist size: {len(allow_ids)} / {U.shape[0]} ({len(allow_ids)/U.shape[0]:.1%})")

    # --- BASELINE (your current pipeline) ---
    show_nearest_neighbors(tokenizer, U_sel, r, topn=20, restrict_ids=None,
                           title="nearest over FULL vocab")
    show_nearest_neighbors(tokenizer, U_sel, r, topn=20, restrict_ids=allow_ids,
                           title="nearest over ALLOWLIST")

    seed_ids, _delta = build_logit_seed_ids(W_out, r, topn=5000, intersect_with=allow_ids)
    print(f"\nLogit-seed size (∩ allowlist): {len(seed_ids)}")
    show_nearest_neighbors(tokenizer, U_sel, r, topn=20, restrict_ids=seed_ids,
                           title="nearest over LOGIT-SEED ∩ ALLOWLIST")

    token_ids, coeffs, err, cos = omp_k_sparse(
        r=r, U=U, U_sel=U_sel, k=K, eps=EPS, nonneg=NONNEG, restrict_ids=seed_ids
    )
    print("\n=== k-sparse over INPUT embeddings (LOGIT-SEED ∩ ALLOWLIST) ===")
    for tid, c in zip(token_ids, coeffs):
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        txt = tokenizer.decode([int(tid)], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        print(f"[{tid:5d}] coef={float(c):+ .4f}   token={tok!r:>16}   decode={txt!r}")
    print(f"\nResidual L2: {err:.6f}   Cosine: {cos:.6f}")

    _ = sweep_k(r, U, U_sel, ks=[10, 20, 50, 100, 200],
                restrict_ids=seed_ids, nonneg=NONNEG, eps=EPS,
                title="Sweep k (NO deflation)")

    # --- PCA deflation pass (optional) ---
    if DO_PC_DEFLATION:
        print(f"\n--- PCA deflation: removing top {NUM_PCS} PCs (center={CENTER_BEFORE_PCA}) ---")
        U_defl, U_sel_defl, r_defl = pca_deflate_embeddings(U, r, num_pcs=NUM_PCS, center=CENTER_BEFORE_PCA)

        show_nearest_neighbors(tokenizer, U_sel_defl, r_defl, topn=20, restrict_ids=None,
                               title="DEFLECTED: nearest over FULL vocab")
        show_nearest_neighbors(tokenizer, U_sel_defl, r_defl, topn=20, restrict_ids=allow_ids,
                               title="DEFLECTED: nearest over ALLOWLIST")

        r_for_seed = r_defl if SEED_WITH_DEFLATED_R else r
        seed_ids_defl, _ = build_logit_seed_ids(W_out, r_for_seed, topn=5000, intersect_with=allow_ids)
        print(f"\nDEFLECTED: Logit-seed size (∩ allowlist): {len(seed_ids_defl)}")
        show_nearest_neighbors(tokenizer, U_sel_defl, r_defl, topn=20, restrict_ids=seed_ids_defl,
                               title="DEFLECTED: nearest over LOGIT-SEED ∩ ALLOWLIST")

        token_ids_d, coeffs_d, err_d, cos_d = omp_k_sparse(
            r=r_defl, U=U_defl, U_sel=U_sel_defl, k=K, eps=EPS,
            nonneg=NONNEG, restrict_ids=seed_ids_defl
        )
        print("\n=== DEFLECTED: k-sparse over INPUT embeddings (LOGIT-SEED ∩ ALLOWLIST) ===")
        for tid, c in zip(token_ids_d, coeffs_d):
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            txt = tokenizer.decode([int(tid)], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            print(f"[{tid:5d}] coef={float(c):+ .4f}   token={tok!r:>16}   decode={txt!r}")
        print(f"\nDEFLECTED: Residual L2: {err_d:.6f}   Cosine: {cos_d:.6f}")

        _ = sweep_k(r_defl, U_defl, U_sel_defl, ks=[10, 20, 50, 100, 200],
                    restrict_ids=seed_ids_defl, nonneg=NONNEG, eps=EPS,
                    title=f"Sweep k (DEFLECTED, {NUM_PCS} PCs removed)")

    # --- Sign-consistent OMP (positive vs negative Δlogit) ---
    if RUN_SIGN_CONSISTENT:
        curated_ids = build_curated_refusal_ids(tokenizer) if USE_CURATED_LEXICON else None
        results, err_sc, cos_sc = sign_consistent_omp(
            r=r, U=U, U_sel=U_sel, W_out=W_out, k=K,
            allow_ids=allow_ids, topn=SC_TOPN, curated_ids=curated_ids
        )
        print("\n=== SIGN-CONSISTENT k-sparse over INPUT embeddings ===")
        for tid, c in results:
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            txt = tokenizer.decode([int(tid)], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            print(f"[{tid:5d}] coef={c:+ .4f}   token={tok!r:>16}   decode={txt!r}")
        print(f"\nSIGN-CONSISTENT: Residual L2: {err_sc:.6f}   Cosine: {cos_sc:.6f}")

if __name__ == "__main__":
    main()
