# sparse_refusal_fit_v2.py
import os
import json
import warnings
import math
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Your config: must define LLAMA_2_7B, REFUSAL_NV_PATH, LAYER_IDX, DEVICE
from config.names import *

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

# ---------------- tunables ----------------
DELTA = 10              # target sparsity for primary run
M_PRESELECT = 16384     # candidate atoms kept after correlation preselect (for OMP/HTP)
RIDGE_LAMBDA = 1e-6     # small Tikhonov for dense baseline Gram solve
DO_K_SWEEP = True
DO_LOGIT_HEURISTIC = True
TOP_TOKENS_TO_PRINT = 30
SAVE_DIR = REFUSAL_NV_PATH
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- load model safely ----------------
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_2_7B,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_state_dict=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---------------- load steering vector (residual space) ----------------
r_hat = torch.load(
    os.path.join(REFUSAL_NV_PATH, f"vec_layer_{LAYER_IDX}_Llama-2-7b-chat-hf.pt"),
    weights_only=True,
)
r = r_hat.to(dtype=torch.float32, device=DEVICE)  # [d]

# ---------------- meta-safe load of lm_head.weight ----------------
def load_lm_head_weight(repo_id: str) -> torch.Tensor:
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
    )
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_name = index["weight_map"]["lm_head.weight"]
    state = load_file(os.path.join(local_dir, shard_name), device="cpu")
    return state["lm_head.weight"].to(torch.float32)  # [V, d] CPU

W = load_lm_head_weight(LLAMA_2_7B)  # [V, d] CPU fp32
V, d = W.shape
assert r.numel() == d, f"r has dim {r.numel()} but model hidden dim is {d}"

# ---------------- helpers ----------------
@torch.inference_mode()
def explained_energy(eps: float, rnorm: float) -> float:
    return max(0.0, 1.0 - (eps / (rnorm + 1e-12)) ** 2)

@torch.inference_mode()
def debias_fit(W_rows_idx: torch.Tensor, r_dev: torch.Tensor):
    D_sup = W[W_rows_idx.cpu(), :].t().to(device=DEVICE, dtype=torch.float32)  # [d, s]
    x = torch.linalg.lstsq(D_sup, r_dev).solution
    eps = torch.linalg.vector_norm(D_sup @ x - r_dev).item()
    return x, eps

# ---------------- OMP/HTP dictionary (normalized) ----------------
with torch.inference_mode():
    col_norms = torch.linalg.vector_norm(W, dim=1) + 1e-12      # [V] CPU
    corr = (W @ r.cpu()) / col_norms                            # [V] CPU
    M = min(M_PRESELECT, V)
    topM = torch.topk(torch.abs(corr), k=M, largest=True).indices  # [M] CPU

D_sub = W[topM, :].t()                             # [d, M] CPU
Dn_sub = (D_sub / col_norms[topM])                 # [d, M] CPU
Dn_sub = Dn_sub.to(device=DEVICE, dtype=torch.float32)
r_dev = r
r_norm = torch.linalg.vector_norm(r_dev).item()

# ---------------- OMP on unit-norm columns ----------------
@torch.inference_mode()
def omp_unit_atoms(D_unit: torch.Tensor, r: torch.Tensor, k: int, eps_target: float = 0.0):
    d, m = D_unit.shape
    a_prime = torch.zeros(m, device=D_unit.device, dtype=torch.float32)
    residual = r.clone()
    support = []
    x = None
    for _ in range(k):
        c = torch.mv(D_unit.t(), residual)
        j = int(torch.argmax(torch.abs(c)).item())
        if j in support:
            break
        support.append(j)
        Ds = D_unit[:, support]
        x = torch.linalg.lstsq(Ds, r).solution
        residual = r - Ds @ x
        if eps_target > 0.0 and torch.linalg.vector_norm(residual).item() <= eps_target:
            break
    if x is not None:
        a_prime[support] = x
    eps = torch.linalg.vector_norm(residual).item()
    return a_prime, eps, support

# ---------------- HTP on unit-norm columns ----------------
@torch.inference_mode()
def htp_unit_atoms(D_unit: torch.Tensor, r: torch.Tensor, k: int, iters: int = 20):
    d, m = D_unit.shape
    Dt = D_unit.t()
    a = torch.zeros(m, device=D_unit.device, dtype=torch.float32)
    # quick Lipschitz estimate
    x = torch.randn(m, device=D_unit.device, dtype=torch.float32)
    x /= (torch.linalg.vector_norm(x) + 1e-12)
    for _ in range(3):
        x = Dt @ (D_unit @ x)
        x /= (torch.linalg.vector_norm(x) + 1e-12)
    L = float(torch.dot(x, Dt @ (D_unit @ x)))
    step = 1.0 / max(L, 1.0)
    for _ in range(iters):
        grad = Dt @ (D_unit @ a - r)
        z = a - step * grad
        idx = torch.topk(torch.abs(z), k=k).indices
        Ds = D_unit[:, idx]
        x_ls = torch.linalg.lstsq(Ds, r).solution
        a.zero_()
        a[idx] = x_ls
    return a

# ---------------- Dense baseline: solve a_dense, then prune+debias ----------------
@torch.inference_mode()
def dense_min_norm_solution(W_cpu: torch.Tensor, r_dev: torch.Tensor, ridge: float = 1e-6):
    # Solve x in (W^T W + λI) x = r ; then a = W x  (exact if λ=0 and full rank)
    # All on CPU to keep memory predictable
    G = (W_cpu.t() @ W_cpu).to(torch.float64)                     # [d, d]
    if ridge > 0:
        G = G + ridge * torch.eye(G.shape[0], dtype=G.dtype)
    x = torch.linalg.solve(G, r_dev.cpu().to(torch.float64))      # [d]
    a = (W_cpu @ x.to(torch.float32))                             # [V]
    return a.to(torch.float32)                                    # CPU

@torch.inference_mode()
def prune_and_debias(a_full_cpu: torch.Tensor, k: int, r_dev: torch.Tensor):
    # Pick top-k |a|, debias via LS on original atoms
    idx = torch.topk(torch.abs(a_full_cpu), k=min(k, a_full_cpu.numel())).indices
    x, eps = debias_fit(idx, r_dev)
    a_out = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    a_out[idx.to(device=DEVICE)] = x
    return a_out, idx, eps

# ---------------- Run OMP primary ----------------
a_sub_prime, _, sup_sub = omp_unit_atoms(Dn_sub, r_dev, k=DELTA, eps_target=0.0)
a_sub = a_sub_prime / col_norms[topM].to(a_sub_prime.device)
support_full = topM[torch.tensor(sup_sub, dtype=torch.long)]
x_debias, eps_abs = debias_fit(support_full, r_dev)
a_full = torch.zeros(V, device=DEVICE, dtype=torch.float32)
a_full[support_full.to(device=DEVICE)] = x_debias
eps_rel = eps_abs / (r_norm + 1e-12)

print(f"\n=== OMP result (delta={DELTA}) ===")
print(f"epsilon_abs = {eps_abs:.6e}, ||r|| = {r_norm:.6e}, epsilon_rel = {eps_rel:.6e}")
print(f"explained energy = {explained_energy(eps_abs, r_norm)*100:.2f}%")
print(f"nnz(a) = {int((a_full != 0).sum().item())}")

# Inspect tokens
with torch.inference_mode():
    nz_idx = torch.nonzero(a_full).squeeze(1).tolist()
    coeffs = a_full[nz_idx].detach().cpu()
    order = torch.argsort(torch.abs(coeffs), descending=True)
    idx_sorted = [nz_idx[i] for i in order.tolist()]
    coeffs_sorted = coeffs[order].tolist()

print("\nSelected atoms (idx, token, coefficient):")
for i, (idx_i, coef) in enumerate(zip(idx_sorted[:TOP_TOKENS_TO_PRINT], coeffs_sorted[:TOP_TOKENS_TO_PRINT]), 1):
    tok = tokenizer.convert_ids_to_tokens(int(idx_i))
    print(f"{i:2d}. {idx_i:6d}  {tok:20s}  {coef:+.6f}")

# Save OMP result
save_path = os.path.join(SAVE_DIR, f"sparse_code_layer_{LAYER_IDX}_delta_{DELTA}.pt")
torch.save(
    {
        "a": a_full.half().cpu(),
        "epsilon_abs": eps_abs,
        "epsilon_rel": eps_rel,
        "support": [int(i) for i in support_full.tolist()],
        "delta": int(DELTA),
        "M": int(M),
        "layer": int(LAYER_IDX),
        "method": "OMP+debias",
    },
    save_path,
)
print(f"\nSaved OMP sparse code to: {save_path}")

# ---------------- k-sweep: OMP and HTP ----------------
if DO_K_SWEEP:
    print("\n=== k-sweep (OMP + debias) ===")
    for k in [5, 10, 15, 20, 30, 40, 60, 80, 100]:
        a_prime_k, _, sup_k = omp_unit_atoms(Dn_sub, r_dev, k=k, eps_target=0.0)
        a_sub_k = a_prime_k / col_norms[topM].to(a_prime_k.device)
        sup_full_k = topM[torch.tensor(sup_k, dtype=torch.long)]
        x_k, eps_k = debias_fit(sup_full_k, r_dev)
        print(f"k={k:3d}  eps_abs={eps_k:9.4e}  eps_rel={eps_k/r_norm:7.4f}  explained={explained_energy(eps_k, r_norm)*100:6.2f}%")

    print("\n=== k-sweep (HTP + debias) ===")
    for k in [5, 10, 15, 20, 30, 40, 60, 80, 100]:
        a_prime_k = htp_unit_atoms(Dn_sub, r_dev, k=k, iters=20)
        idx_k = torch.topk(torch.abs(a_prime_k), k=k).indices
        sup_full_k = topM[idx_k.cpu()]
        x_k, eps_k = debias_fit(sup_full_k, r_dev)
        print(f"k={k:3d}  eps_abs={eps_k:9.4e}  eps_rel={eps_k/r_norm:7.4f}  explained={explained_energy(eps_k, r_norm)*100:6.2f}%")

# ---------------- Dense→Prune→Debias baseline ----------------
print("\n=== Dense→Prune→Debias baseline ===")
a_dense = dense_min_norm_solution(W, r_dev, ridge=RIDGE_LAMBDA)     # CPU
for k in [5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]:
    a_k, idx_k, eps_k = prune_and_debias(a_dense, k, r_dev)         # a_k on DEVICE
    print(f"k={k:3d}  eps_abs={eps_k:9.4e}  eps_rel={eps_k/r_norm:7.4f}  explained={explained_energy(eps_k, r_norm)*100:6.2f}%")

# Save the dense-pruned at DELTA as well
a_k, idx_k, eps_k = prune_and_debias(a_dense, DELTA, r_dev)
save_path2 = os.path.join(SAVE_DIR, f"sparse_code_denseprune_layer_{LAYER_IDX}_delta_{DELTA}.pt")
torch.save(
    {
        "a": a_k.half().cpu(),
        "epsilon_abs": eps_k,
        "epsilon_rel": eps_k / (r_norm + 1e-12),
        "support": [int(i) for i in idx_k.tolist()],
        "delta": int(DELTA),
        "layer": int(LAYER_IDX),
        "method": "DensePrune+debias",
        "ridge_lambda": float(RIDGE_LAMBDA),
    },
    save_path2,
)
print(f"Saved DensePrune sparse code to: {save_path2}")

# ---------------- optional: logit-space heuristic ----------------
if DO_LOGIT_HEURISTIC:
    try:
        with torch.inference_mode():
            r_post = model.model.norm(r_hat.unsqueeze(0)).squeeze(0).to(device=DEVICE, dtype=torch.float32)
            logit_dir = (W.to(torch.float32) @ r_post.cpu())  # [V] CPU
            k_logits = 20
            idx = torch.topk(torch.abs(logit_dir), k=k_logits).indices.tolist()
            print(f"\nTop-{k_logits} tokens by |logit delta| (heuristic):")
            for j in idx:
                tok = tokenizer.convert_ids_to_tokens(int(j))
                val = float(logit_dir[j])
                print(f"{j:6d}  {tok:20s}  {val:+.6f}")
    except Exception as e:
        print(f"\nLogit-space step skipped: {e}")

# ---------------- optional: bar plot for OMP result ----------------
try:
    with torch.inference_mode():
        nz_idx = torch.nonzero(a_full).squeeze(1).tolist()
        coeffs = a_full[nz_idx].detach().cpu()
        order = torch.argsort(torch.abs(coeffs), descending=True)
        idx_sorted = [nz_idx[i] for i in order.tolist()]
        coeffs_sorted = coeffs[order].tolist()
    top_show = min(len(idx_sorted), TOP_TOKENS_TO_PRINT)
    mags = [abs(c) for c in coeffs_sorted[:top_show]]
    labels = [tokenizer.convert_ids_to_tokens(i) for i in idx_sorted[:top_show]]
    plt.figure(figsize=(10, 4))
    plt.bar(range(top_show), mags)
    plt.xticks(range(top_show), labels, rotation=90)
    plt.title(f"OMP coefficients (|coef|) — delta={DELTA}, explained={explained_energy(eps_abs, r_norm)*100:.1f}%")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Plot skipped: {e}")
