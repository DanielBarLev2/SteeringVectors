# 🧭 Interpreting Steering Vectors
**Daniel Bar Lev & Gal Cohen**  
_Tel Aviv University_  
📄 [Read the full paper (PDF)](./Interpreting_Steering_Vectors.pdf)

---

## 🧩 Overview

**Steering vectors** are fixed directions injected into the residual stream of large language models (LLMs) to control their behavior — for example, to make a model *refuse harmful requests* or *answer truthfully*.  
While effective, the **semantics** of these vectors remain unclear.

This repository accompanies the paper **"Interpreting Steering Vectors" (2025)**, which investigates whether a steering vector — specifically the *refusal direction* from **Arditi et al. (2024)** — can be **interpreted in token space**.

We propose three complementary interpretability methods:
1. **Projection into the Vocabulary Space**
2. **Sparse Dictionary Learning**
3. **Controlled Logit Comparison**

Together, they reveal that the refusal vector aligns with a *small, interpretable set of refusal-related tokens*, such as `prohib`, `illegal`, `harm`, and `forbid`.

**Our results show that the refusal steering vector can be interpreted through a consistent lexical core composed of tokens such as `prohib`, `illegal`, `harm`, `forbidden`, `dangerous`, and `cannot`, which dominate both static projections and dynamic logit deltas.**


## 🧠 Key Idea

> Steering vectors are not opaque latent features — they correspond to interpretable lexical dimensions in the model’s output space.

By analyzing how these vectors project onto and interact with token embeddings, we can **trace their meaning directly in vocabulary space**, rather than through abstract latent dimensions.

---

## 🧪 Methodology

### 1. Projection into the Vocabulary Space

We normalize the steering vector \( r \) and project it through the decoding matrix \( W_{decode} \):


s = RMSNorm(r) @ W_{decode


This produces a score for each token.  
Top-scoring tokens consistently include **refusal-related words**, such as:

| Token | Score |
|--------|--------|
| harmful | +10.25 |
| prohibited | +8.65 |
| illegal | +5.71 |
| forbidden | +4.69 |

---

### 2. Sparse Dictionary Learning

We reconstruct \( r \) using a **small set of token embeddings** by solving a sparse regression:


min(alpha)|r - W_{decode} @ alpha|_2^2 + lambda|alpha|_0


High-weight tokens (for large λ) correspond to **refusal semantics**:

| Token | Weight | Cosine Similarity |
|--------|---------|------------------|
| prohib | +0.0437 | +0.0999 |
| harm | +0.0216 | +0.0696 |
| illegal | +0.0215 | +0.0748 |
| Sorry | +0.0195 | +0.0662 |

Sparse reconstructions reveal a compact, interpretable core of tokens that approximate the refusal direction.

---

### 3. Controlled Logit Comparison

We compare **token logits** between *steered* and *baseline* models under matched prefixes.  
For each prefix \( t \), we compute:

\[
\Delta_t = \ell_{steered}^t - \ell_{baseline}^t
\]

and average across positions and prompts.

This approach captures how \( r \) reshapes token probabilities in generation.

| **Harmless Prompts (α = +1)** | **Harmful Prompts (α = −1)** |
|-------------------------------|-------------------------------|
| prohib (+7.09) | great (+4.40) |
| illegal (+6.57) | pleasure (+4.40) |
| cannot (+6.14) | happy (+4.15) |
| neither (+6.03) | wonderful (+3.98) |

The same lexical polarity appears consistently:  
**Prohibition vs. Affirmation**.

---

## 📈 Results Summary

| Method | Core Finding | Signal Quality |
|---------|---------------|----------------|
| Projection | Reveals alignment with refusal tokens | Moderate (noisy due to subwords) |
| Sparse Coding | Produces compact, interpretable token sets | High |
| Logit Comparison | Demonstrates dynamic effect on generation | Very High |

Across all methods, the same lexical field dominates — indicating **a stable and interpretable semantic direction**.

---

