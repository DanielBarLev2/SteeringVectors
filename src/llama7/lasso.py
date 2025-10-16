from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from transformers import AutoTokenizer
from config.names import *
import numpy as np
import os
import torch

# ---------- Config ----------
MODEL_NAME = LLAMA_2_7B
R_PATH = SV_PATH
K = 20
EPS = 1e-6
sparsity_lambdas = [1e-4, 2e-5, 1.25e-5, 1.2e-5, 1e-5, 9e-6, 5e-6, 2e-6, 1e-7, 1e-8]


# ---------- Helpers ----------

def plot_token_weights(sparsity, token_data, top_k=20, save_dir="lasso_token_bars"):
    os.makedirs(save_dir, exist_ok=True)
    if not token_data:
        print(f"[warn] Skipping lambda={sparsity:.1e}: No non-zero token weights.")
        return

    # Sort and slice top tokens
    token_data = sorted(token_data, key=lambda x: -abs(x[1]))[:top_k]
    tokens = [repr(t[3]) for t in token_data][::-1]
    weights = [t[1] for t in token_data][::-1]
    cosines = [t[2] for t in token_data][::-1]

    # Dynamic range for colormap
    cos_min = min(cosines)
    cos_max = max(cosines)
    range_padding = 0.05 * (cos_max - cos_min + EPS)
    norm = plt.Normalize(cos_min - range_padding, cos_max + range_padding)
    cmap = cm.coolwarm

    # Plot
    plt.figure(figsize=(12, max(6, int(top_k * 0.6))))
    colors = [cmap(norm(cos)) for cos in cosines]
    bars = plt.barh(tokens, weights, color=colors)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
    cbar.set_label('Cosine similarity with r')

    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Weight")
    plt.title(f"Token Weights — Top {top_k} Tokens")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/lambda_{sparsity:.1e}_top_tokens.png")
    plt.close()


# ---------- Main ----------
def main():
    torch.set_grad_enabled(False)

    # Load r (steering vector)
    r = torch.load(str(R_PATH), weights_only=True, map_location="cpu")
    r = r.cpu().numpy()
    r = r / np.linalg.norm(r)

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lm_head = torch.load("lm_head.pt")  # Load the output embedding matrix of the model
    D = normalize(lm_head, axis=1).T  # (H,V)

    results = []

    for spars_lambda in sparsity_lambdas:
        lasso = Lasso(alpha=spars_lambda, fit_intercept=False, max_iter=10000)
        lasso.fit(D, r)
        alpha = lasso.coef_

        r_hat = D @ alpha
        l1_dist = np.linalg.norm(r - r_hat, ord=1)
        l2_dist = np.linalg.norm(r - r_hat, ord=2)

        # Find nonzero components
        nonzero_indices = np.nonzero(alpha)[0]
        sparse_atoms = [(i, alpha[i]) for i in nonzero_indices]
        sparse_atoms.sort(key=lambda x: -abs(x[1]))

        token_data = []
        for i, val in sparse_atoms:
            token = tokenizer.decode([i], clean_up_tokenization_spaces=True).strip()
            cosine = np.dot(r, D[:, i]) / (np.linalg.norm(D[:, i]) + EPS)
            token_data.append((i, val, cosine, token))

        results.append({
            'lambda': spars_lambda,
            'alpha': alpha,
            'r_hat': r_hat,
            'l1_dist': l1_dist,
            'l2_dist': l2_dist,
            'sparse_atoms': sparse_atoms,
            'num_nonzero': len(sparse_atoms),
            'tokens': token_data
        })

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (10, 6)

    # Plot metrics
    lambdas = [r['lambda'] for r in results]
    l2s = [r['l2_dist'] for r in results]
    nonzeros = [r['num_nonzero'] for r in results]

    plt.figure()
    plt.plot(lambdas, l2s, marker='o', color='orange')
    plt.xscale('log')
    plt.xlabel('Lambda (Sparsity)')
    plt.ylabel('L2 Distance')
    plt.title(f"L2 Distance of r vs Dα")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"lasso_l2_distance.png")

    plt.figure()
    plt.plot(lambdas, nonzeros, marker='^', color='purple')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('# Non-zero Coefficients')
    plt.title(f"Sparsity vs Lambda")
    plt.grid(True)
    plt.savefig(f"lasso_sparsity.png")

    out_dir = f"lasso_tokens"
    os.makedirs(out_dir, exist_ok=True)

    for res in results:
        if res['tokens']:
            with open(f"{out_dir}/lambda_{res['lambda']:.1e}.txt", "w") as f:
                f.write("Index\tWeight\tCosine\tToken\n")
                for idx, val, cosine, tok in res['tokens']:
                    f.write(f"{idx}\t{val:+.5f}\t{cosine:+.5f}\t{repr(tok)}\n")
            plot_token_weights(res['lambda'], res['tokens'], save_dir=out_dir)


if __name__ == "__main__":
    main()
