Perfect — let’s consolidate everything into a **full STAR specification**. Think of this like the “blueprint” before we draft the final LaTeX methodology section. I’ll structure it into **inputs → modules → outputs**, with notes on intent, shapes, and where STAR differs from prior models.

---

# ⭐ STAR (Shapelet-Temporal Transformer) — Full Specification

---

## **1. Input and Preprocessing**

* Input: multivariate time series

  $$
  X \in \mathbb{R}^{T \times V}
  $$

  where $T$ = length, $V$ = number of variables.
* Preprocess: z-normalize each variable independently (remove mean, scale variance).

---

## **2. Shapelet Bank (from FAST)**

* Use FAST’s kernel–shapelet induction process:

  * Apply base kernels of length 3, 6, 9 across series.
  * From kernel responses, induce **candidate shapelets** via distance distribution profile.
  * Select top-$K$ discriminative shapelets across variables.

**Output:**
$\mathcal{S} = \{S_1, \dots, S_K\},\; S_j \in \mathbb{R}^{\ell_j}$.
Shapelets are **generic** (not class-specific).

---

## **3. Shapelet Activation Trajectories**

For each shapelet $S_j$ and variable $v_j$:

* Slide $S_j$ across $X_{:, v_j}$.
* Compute similarity or negative distance:

  $$
  a_{j,t} = -D\!\big(X_{t:t+\ell_j-1, v_j}, S_j\big), \quad t = 1,\dots,T-\ell_j+1.
  $$

**Option A: Time-grid tokens (recommended)**

* Align activations to center index → for each time step $t$, form vector:

  $$
  z_t = [a_{1}(t), a_{2}(t), \dots, a_{K}(t)] \in \mathbb{R}^{K}.
  $$
* Stack across time:

  $$
  Z \in \mathbb{R}^{T \times K}.
  $$

**Output:** activation matrix, preserves *temporal evolution* of all shapelets across variables.

---

## **4. Embedding Projection**

* Linear projection (learnable):

  $$
  H_0 = Z W_e + b, \quad W_e \in \mathbb{R}^{K \times d}, \; b \in \mathbb{R}^d.
  $$
* $d$ = model dimension (e.g. 128–256).

**Purpose:** compress $K$-dim activation vector into dense representation, allow model to reweight shapelets.

---

## **5. Positional Encodings**

* Add temporal positional encodings to each timestep:

  $$
  H_0 \leftarrow H_0 + PE_{time}(t).
  $$
* PE can be sinusoidal (Vaswani et al.) or learned.
* Optional: add variable-ID embeddings if needed, but not required since shapelet bank already covers variables.

---

## **6. Transformer Encoder**

* Input: sequence of length $T$, each embedding $\in \mathbb{R}^d$.
* Standard Transformer encoder (L layers):

  * Multi-head self-attention: captures correlations of shapelet activations across **time** and **variables**.
  * Feedforward, residual connections, LayerNorm.

**Output:** contextualized sequence

$$
H_L \in \mathbb{R}^{T \times d}.
$$

---

## **7. Sequence Aggregation**

* Two options:

  * \[CLS] token: prepend to input sequence, take its final embedding $H_{cls} \in \mathbb{R}^d$.
  * Attention pooling: weighted average over timesteps.

---

## **8. Classification Head**

* Linear classifier + softmax:

  $$
  \hat{y} = \text{softmax}(W_{cls} H^* + b_{cls}).
  $$

where $H^*$ is global representation.

---

## **9. Flow Summary**

1. **Input**: $X \in \mathbb{R}^{T \times V}$.
2. **Shapelet discovery (FAST)**: select $K$ discriminative shapelets.
3. **Activation trajectories**: compute similarity signals across time.
4. **Time-grid tokens**: stack all shapelet activations per timestep → $Z \in \mathbb{R}^{T \times K}$.
5. **Projection**: project to model dim $d$.
6. **Positional encoding**: add temporal PEs.
7. **Transformer encoder**: contextualize across time and variables.
8. **Pooling**: aggregate sequence representation.
9. **Classification head**: predict class.

---

## **10. Shapes (concrete example)**

* Dataset: $T=100, V=8$.
* Shapelets: $K=60$.
* Model dim: $d=256$.

**Step 3:** activations $Z \in \mathbb{R}^{100 \times 60}$.
**Step 4:** projection $H_0 \in \mathbb{R}^{100 \times 256}$.
**Step 5–6:** Transformer encoder output $H_L \in \mathbb{R}^{100 \times 256}$.
**Step 7:** pool → $H^* \in \mathbb{R}^{256}$.
**Step 8:** classifier → logits → softmax.

---

## **11. Conceptual Differences vs ShapeFormer**

* STAR: **temporalized, class-agnostic** activations; unified temporal modeling.
* ShapeFormer: **static, class-specific** difference tokens + dual-branch design.
* STAR better suited for **turning-point sensitivity**, where timing of discriminative subsequences matters.

---

✅ This is the full STAR specification — **flows, compartments, steps, shapes, and intent**.

👉 Next step: would you like me to now **turn this into a polished LaTeX methodology section** (with equations + explanatory prose), similar to how we did for FAST?
