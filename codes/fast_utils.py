from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Literal

import numpy as np
from numba import njit, prange, set_num_threads


# ---------------------------
# Thread control
# ---------------------------

def set_numba_threads_count(n: int | None):
    if n is not None and n > 0:
        set_num_threads(n)

# ---------------------------
# UCR/UEA loader (sktime) + converters + z-norm
# ---------------------------

def _concat_len(v) -> int:
    a = np.asarray(v, dtype=np.float32).ravel()
    return int(a.size)

def _nested_concat_row(X_row) -> np.ndarray:
    """Concatenate all columns (variables) for one nested row into 1D."""
    segs = []
    for j in range(X_row.shape[0]):
        v = np.asarray(X_row.iloc[j], dtype=np.float32).ravel()
        segs.append(v)
    return np.concatenate(segs, axis=0) if len(segs) > 1 else segs[0]

def _nested_to_2d_with_pad(X_train_nested, X_test_nested) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert nested DataFrames to dense (n, Lmax) by concatenating columns per row
    and padding to a *global* Lmax computed over train ∪ test.
    """
    import pandas as pd
    if not isinstance(X_train_nested, pd.DataFrame):
        X_train_nested = pd.DataFrame(X_train_nested)
    if not isinstance(X_test_nested, pd.DataFrame):
        X_test_nested = pd.DataFrame(X_test_nested)

    def _total_len(df):
        Ls = []
        for i in range(len(df)):
            L = 0
            for j in range(df.shape[1]):
                L += _concat_len(df.iat[i, j])
            Ls.append(L)
        return Ls

    train_lengths = _total_len(X_train_nested)
    test_lengths = _total_len(X_test_nested)
    Lmax = int(max(max(train_lengths) if train_lengths else 0,
                   max(test_lengths) if test_lengths else 0))

    Xtr = np.zeros((len(X_train_nested), Lmax), dtype=np.float32)
    for i in range(len(X_train_nested)):
        rowv = _nested_concat_row(X_train_nested.iloc[i, :])
        Xtr[i, :rowv.size] = rowv

    Xte = np.zeros((len(X_test_nested), Lmax), dtype=np.float32)
    for i in range(len(X_test_nested)):
        rowv = _nested_concat_row(X_test_nested.iloc[i, :])
        Xte[i, :rowv.size] = rowv
    return Xtr, Xte

def _nested_to_vtv_with_pad(
    X_train_nested, X_test_nested
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert nested DataFrames to dense (n, T, V) by:
      - assuming each column is a variable,
      - padding each variable's series to a *global* T across train ∪ test,
      - stacking variables into the last axis V.
    """
    import pandas as pd
    if not isinstance(X_train_nested, pd.DataFrame):
        X_train_nested = pd.DataFrame(X_train_nested)
    if not isinstance(X_test_nested, pd.DataFrame):
        X_test_nested = pd.DataFrame(X_test_nested)

    V = int(X_train_nested.shape[1])

    def _col_max_len(df, j):
        L = 0
        for i in range(len(df)):
            L = max(L, _concat_len(df.iat[i, j]))
        return L

    per_var_T = []
    for j in range(V):
        Tj = max(_col_max_len(X_train_nested, j), _col_max_len(X_test_nested, j))
        per_var_T.append(int(Tj))
    T = int(max(per_var_T)) if per_var_T else 0

    Ntr = len(X_train_nested)
    Nte = len(X_test_nested)
    Xtr = np.zeros((Ntr, T, V), dtype=np.float32)
    Xte = np.zeros((Nte, T, V), dtype=np.float32)

    # left-aligned padding per variable
    for i in range(Ntr):
        for j in range(V):
            v = np.asarray(X_train_nested.iat[i, j], dtype=np.float32).ravel()
            L = min(v.size, T)
            if L > 0:
                Xtr[i, :L, j] = v[:L]
    for i in range(Nte):
        for j in range(V):
            v = np.asarray(X_test_nested.iat[i, j], dtype=np.float32).ravel()
            L = min(v.size, T)
            if L > 0:
                Xte[i, :L, j] = v[:L]
    return Xtr, Xte

def _z_norm(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-series, per-variable z-normalization.
    Accepts (N,T,V) or (T,V); returns same shape.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        Xb = X[None, ...]
        mu = Xb.mean(axis=1, keepdims=True)
        sd = Xb.std(axis=1, keepdims=True)
        return ((Xb - mu) / np.maximum(sd, eps))[0]
    elif X.ndim == 3:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.maximum(sd, eps)
    raise ValueError("Expected (T,V) or (N,T,V) array for z-norm.")

def load_ucr_uea_sktime(
    name: str,
    *,
    output: Literal["vtv", "flat"] = "vtv",
    z_norm: bool = True,
    merge_and_resplit: bool = False,
    random_state: int = 0,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust UCR/UEA loader:
      - fetch official splits via sktime,
      - convert to either (N,T,V) 'vtv' or (N,L) 'flat',
      - optional per-series, per-variable z-norm (for 'vtv'),
      - optional merge+resplit at original test fraction.
    Returns: X_train, y_train, X_test, y_test
    """
    from sktime.datasets import load_UCR_UEA_dataset
    from sklearn.model_selection import train_test_split

    X_train_nested, y_train = load_UCR_UEA_dataset(name, split="train", return_X_y=True)
    X_test_nested,  y_test  = load_UCR_UEA_dataset(name, split="test",  return_X_y=True)

    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    classes, inv_tr = np.unique(y_train, return_inverse=True)
    y_train = inv_tr.astype(np.int64)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = np.array([class_to_idx[v] for v in y_test], dtype=np.int64)

    if output == "vtv":
        X_train, X_test = _nested_to_vtv_with_pad(X_train_nested, X_test_nested)
        if z_norm:
            X_train = _z_norm(X_train)
            X_test  = _z_norm(X_test)
    elif output == "flat":
        X_train, X_test = _nested_to_2d_with_pad(X_train_nested, X_test_nested)
    else:
        raise ValueError("output must be 'vtv' or 'flat'.")

    if merge_and_resplit:
        n_tr, n_te = len(y_train), len(y_test)
        orig_test_frac = n_te / float(n_tr + n_te) if (n_tr + n_te) > 0 else 0.2
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        strat = y_all if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=orig_test_frac, random_state=random_state, stratify=strat
        )

    return (np.asarray(X_train, dtype=np.float32),
            np.asarray(y_train, dtype=np.int64),
            np.asarray(X_test,  dtype=np.float32),
            np.asarray(y_test,  dtype=np.int64))


# ---------------------------
# Train/Val split utility (for hyperparam tuning)
# ---------------------------

def split_train_for_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    val_frac: float = 0.2,
    random_state: int = 0,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the official train into train/val (stratified by default).
    Works for (N,T,V) or (N,L). Returns (X_tr, y_tr, X_val, y_val).
    """
    from sklearn.model_selection import train_test_split
    strat = y_train if stratify else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=random_state, stratify=strat
    )
    # keep dtype/layout
    return (np.asarray(X_tr, dtype=np.float32),
            np.asarray(y_tr, dtype=np.int64),
            np.asarray(X_val, dtype=np.float32),
            np.asarray(y_val, dtype=np.int64))

# ---------------------------
# Small helpers
# ---------------------------

def ensure_float32_arrays(arrs: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(a, dtype=np.float32) for a in arrs]

def diff1_batch(X_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for x in X_list:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            out.append(np.diff(x).astype(np.float32, copy=False))
        elif x.ndim == 2:
            out.append(np.diff(x, axis=1).astype(np.float32, copy=False))
        else:
            raise ValueError("diff1_batch expects 1D series or (n,L) matrices")
    return out


# ---------------------------
# Numba primitives
# ---------------------------

@njit(fastmath=True)
def _conv_valid_dilated1d(x: np.ndarray, w: np.ndarray, d: int) -> np.ndarray:
    L = x.shape[0]
    K = w.shape[0]
    span = (K - 1) * d + 1
    out_len = L - span + 1
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(out_len, dtype=np.float32)
    for t in range(out_len):
        s = 0.0
        idx = t
        for k in range(K):
            s += w[k] * x[idx]
            idx += d
        out[t] = s
    return out

@njit(fastmath=True)  # no parallel=True to avoid parfors conflicts when called inside prange regions
def _compute_C_blocks(x: np.ndarray, K: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build C alpha (sum of non-selected taps) and C gamma table (per-tap partial sums)
    for +2/-1 DC-balanced kernels with dilation d.
    """
    L = x.shape[0]
    span = (K - 1) * d + 1
    out_len = L - span + 1
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros((K, 0), dtype=np.float32)

    Ca = np.zeros(out_len, dtype=np.float32)
    Cg = np.zeros((K, out_len), dtype=np.float32)

    for t in range(out_len):
        acc = 0.0
        base = t
        for k in range(K):
            v = x[base]
            acc += v
            Cg[k, t] = 3.0 * v   # +3 when that tap is selected
            base += d
        Ca[t] = -acc            # background = -sum over span
    return Ca, Cg

@njit(parallel=True, fastmath=True)
def _collect_features_batch(
    X2d: np.ndarray,            # (n, L)
    shapelet: np.ndarray,       # (w,)
    thresholds: np.ndarray,     # (m,)
    include_max_feature: bool
) -> np.ndarray:
    """
    Stats per bias:
      0: PPV  = |{t: a[t] > thr}| / out_len
      1: LSPV = longest positive stretch length
      2: MPV  = mean( a[t] - thr | a[t] > thr )      <-- corrected to (v - thr)
      3: MIPV = mean index of positives normalized to [0,1]; -1 if none
      4: MAX  = max(a) if enabled (constant per bias)
    """
    n, L = X2d.shape
    w = shapelet.shape[0]
    m = thresholds.shape[0]
    stats_per_bias = 4 + (1 if include_max_feature else 0)
    if m == 0 or w == 0 or L < w:
        return np.zeros((n, m * stats_per_bias), dtype=np.float32)

    out_len = L - w + 1
    out = np.zeros((n, m * stats_per_bias), dtype=np.float32)

    for i in prange(n):
        x = X2d[i]
        # conv valid (d=1)
        a = np.empty(out_len, dtype=np.float32)
        for t in range(out_len):
            s = 0.0
            for k in range(w):
                s += shapelet[k] * x[t + k]
            a[t] = s

        maxC = 0.0
        if include_max_feature:
            mv = a[0]
            for t in range(1, out_len):
                if a[t] > mv:
                    mv = a[t]
            maxC = mv

        for j in range(m):
            thr = thresholds[j]
            cnt = 0
            sum_above = 0.0
            sum_idx = 0.0
            run = 0
            max_run = 0

            for t in range(out_len):
                v = a[t]
                if v > thr:
                    cnt += 1
                    sum_idx += t
                    sum_above += (v - thr)   # corrected MPV aggregator
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0

            base = j * stats_per_bias
            out[i, base + 0] = (cnt / out_len) if out_len > 0 else 0.0
            out[i, base + 1] = float(max_run)
            out[i, base + 2] = (sum_above / cnt) if cnt > 0 else 0.0
            out[i, base + 3] = (sum_idx / cnt) / (out_len - 1) if (cnt > 0 and out_len > 1) else -1.0
            if include_max_feature:
                out[i, base + 4] = maxC
    return out

@njit(parallel=True, fastmath=True)
def _argmax_table_single_dilation(
    X2d: np.ndarray,           # (n, L)
    K: int,
    indices: np.ndarray,       # (num_kernels, s)
    d: int
) -> np.ndarray:
    n, L = X2d.shape
    num_kernels = indices.shape[0]
    out = np.zeros((n, num_kernels), dtype=np.int32)
    for i in prange(n):
        x = X2d[i]
        Ca, Cg = _compute_C_blocks(x, K, d)
        out_len = Ca.shape[0]
        if out_len <= 0:
            for k in range(num_kernels):
                out[i, k] = -1
            continue
        for k in range(num_kernels):
            C = Ca.copy()
            s = indices.shape[1]
            for j in range(s):
                C += Cg[int(indices[k, j])]
            # argmax
            t_star = 0
            mv = C[0]
            for t in range(1, out_len):
                v = C[t]
                if v > mv:
                    mv = v
                    t_star = t
            out[i, k] = t_star
    return out


# ---------------------------
# Families & kernels
# ---------------------------

def _combinations(n: int, k: int) -> np.ndarray:
    idx = np.arange(k, dtype=np.int32)
    out = []
    while True:
        out.append(idx.copy())
        i = k - 1
        while i >= 0 and idx[i] == i + n - k:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j-1] + 1
    return np.asarray(out, dtype=np.int32)

_MR_INDICES = _combinations(9, 3)   # (84, 3)
_K6_INDICES = _combinations(6, 2)   # (15, 2)
_K3_INDICES = np.arange(3, dtype=np.int32).reshape(-1, 1)  # (3, 1)

@dataclass(frozen=True)
class FamilyKey:
    name: str   # "K9s3", "K6s2", "K3s1"
    K: int

@dataclass(frozen=True)
class ParentKey:
    family: FamilyKey
    dilation: int
    kernel_id: int  # row in indices table

@dataclass
class ParentInfo:
    family: FamilyKey
    dilation: int
    kernel_id: int
    thr_counts: int  # number of biases for this (fam,dil) per kernel
    best_eta2: float = 0.0

@dataclass
class ShapeletSpec:
    parent: ParentKey
    dilation: int
    values: np.ndarray     # (span,)
    thr_base: np.ndarray   # biases on base
    thr_diff: np.ndarray   # kept for compatibility; empty for STAR shapelets
    best_eta2: float


# ---------------------------
# Stats & quantiles
# ---------------------------

def lds_quantiles(m: int, offset: float = 0.0) -> np.ndarray:
    if m <= 0:
        return np.zeros(0, dtype=np.float64)
    q = (np.arange(m, dtype=np.float64) + 0.5) / float(m)
    if offset != 0.0:
        q = (q + offset) - np.floor(q + offset)
    return q

def anova_F_eta2_by_column(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, m = X.shape
    classes = np.unique(y)
    C = classes.size
    if n == 0 or m == 0 or C < 2:
        return np.zeros(m), np.zeros(m)

    mu = X.mean(axis=0)
    SS_tot = ((X - mu) ** 2).sum(axis=0)

    SS_between = np.zeros(m, dtype=np.float64)
    for c in classes:
        mask = (y == c)
        if not np.any(mask):
            continue
        mu_c = X[mask].mean(axis=0)
        SS_between += mask.sum() * (mu_c - mu) ** 2

    eta2 = SS_between / np.maximum(SS_tot, 1e-12)
    df_between = C - 1
    df_within = n - C
    MS_between = SS_between / max(df_between, 1)
    MS_within = (SS_tot - SS_between) / max(df_within, 1)
    F = MS_between / np.maximum(MS_within, 1e-12)
    return F, eta2.astype(np.float64)


# ---------------------------
# MultiROCKET-like dilation fitting
# ---------------------------

def _mr_fit_dilations(input_length: int, num_thresholds_per_kernel: int, max_dilations_per_kernel: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_thresholds_per_kernel <= 0 or input_length <= 1:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    true_max = min(num_thresholds_per_kernel, max_dilations_per_kernel)
    if true_max <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    max_exp = float(np.log2(max(1.0, (input_length - 1) / 8.0)))  # (L-1)/(9-1)
    raw = np.power(2.0, np.linspace(0.0, max_exp, true_max)).astype(np.int32)
    if raw.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    dils_list: List[int] = []
    counts_list: List[int] = []
    prev = None
    for v in raw:
        if prev is None or v != prev:
            dils_list.append(int(v))
            counts_list.append(1)
            prev = int(v)
        else:
            counts_list[-1] += 1

    counts = np.asarray(counts_list, dtype=np.float64)
    multiplier = num_thresholds_per_kernel / float(true_max)
    counts = np.floor(counts * multiplier + 1e-9).astype(np.int32)
    shortfall = int(num_thresholds_per_kernel - int(counts.sum()))
    i = 0
    n = counts.size
    while shortfall > 0 and n > 0:
        counts[i] += 1
        shortfall -= 1
        i = (i + 1) % n

    return np.asarray(dils_list, dtype=np.int32), counts


# ---------------------------
# Step-1: Build kernel features
# ---------------------------

def _make_kernel_values(K: int, select_idx: np.ndarray) -> np.ndarray:
    w = -np.ones(K, dtype=np.float32)
    for j in range(select_idx.shape[0]):
        w[int(select_idx[j])] = 2.0
    return w

def _dilated_kernel(w: np.ndarray, d: int) -> np.ndarray:
    if d == 1:
        return w.copy()
    K = w.shape[0]
    span = (K - 1) * d + 1
    wd = np.zeros(span, dtype=np.float32)
    pos = 0
    for k in range(K):
        wd[pos] = w[k]
        pos += d
    return wd

def _family_table(name: str) -> Tuple[FamilyKey, np.ndarray]:
    if name == "K9s3":
        return FamilyKey("K9s3", 9), _MR_INDICES
    elif name == "K6s2":
        return FamilyKey("K6s2", 6), _K6_INDICES
    elif name == "K3s1":
        return FamilyKey("K3s1", 3), _K3_INDICES
    else:
        raise ValueError("Unknown family")

def build_parent_argmax_map(
    X_train: np.ndarray,
    ranked_parents: List[ParentKey],
) -> Dict[ParentKey, np.ndarray]:
    """
    Cache t* (argmax positions) per ParentKey across all training series.
    Groups by (family, dilation) to reuse C-construction.
    """
    X2d = np.asarray(X_train, dtype=np.float32)
    groups: Dict[Tuple[str, int], List[int]] = {}
    fam_of: Dict[Tuple[str, int], FamilyKey] = {}
    for pk in ranked_parents:
        key = (pk.family.name, int(pk.dilation))
        if key not in groups:
            groups[key] = []
            fam_of[key] = pk.family
        groups[key].append(int(pk.kernel_id))

    idx_map = {"K9s3": _MR_INDICES, "K6s2": _K6_INDICES, "K3s1": _K3_INDICES}
    argmax_map: Dict[ParentKey, np.ndarray] = {}
    for key, kids in groups.items():
        fam = fam_of[key]
        d = int(key[1])
        indices = idx_map[fam.name]
        table = _argmax_table_single_dilation(X2d, fam.K, indices, d)  # (n, num_kernels)
        for kid in kids:
            pk = ParentKey(fam, d, int(kid))
            argmax_map[pk] = table[:, int(kid)].copy()
    return argmax_map


def build_kernel_features_step1(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    additional_kernel_families: bool,
    kernel_budget_total: int,
    include_max_feature: bool,         # 4 stats baseline; caller may inflate budget externally if True
    subsample_frac_for_thresholds: float,
    subsample_min: int,
    subsample_max: int,
    lds_offset: float,
) -> Tuple[np.ndarray, np.ndarray, List[str],
           Dict[ParentKey, ParentInfo], List[ParentKey], Dict[ParentKey, float], Dict[ParentKey, Tuple[int, int]]]:
    """
    Stage-1:
      - Allocate per-kernel threshold counts across dilations (log2 for K9, fixed for K6/K3)
      - Calibrate biases on a single training series
      - Emit stats on raw and first-difference channels
      - Rank parents by best eta^2 across emitted stats
    """
    Xtr = np.asarray(X_train, dtype=np.float32)
    Xte = np.asarray(X_test, dtype=np.float32)
    ytr = np.asarray(y_train, dtype=np.int64)

    n_tr, L = Xtr.shape
    stats_per_bias = 4 + (1 if include_max_feature else 0)

    fams: List[str] = ["K9s3"]
    if additional_kernel_families:
        fams = ["K9s3", "K6s2", "K3s1"]

    fam_kernel_counts = {"K9s3": _MR_INDICES.shape[0], "K6s2": _K6_INDICES.shape[0], "K3s1": _K3_INDICES.shape[0]}
    total_kernels = sum(fam_kernel_counts[f] for f in fams)

    total_thresholds_target = kernel_budget_total // max(1, (2 * stats_per_bias))  # split over RAW & DIFF
    thresholds_per_kernel = max(1, total_thresholds_target // max(1, total_kernels))

    Xtr_diff2d = np.diff(Xtr, axis=1).astype(np.float32, copy=False)
    Xte_diff2d = np.diff(Xte, axis=1).astype(np.float32, copy=False)

    fam_dils: Dict[str, np.ndarray] = {}
    fam_counts: Dict[str, np.ndarray] = {}

    dils9, cnts9 = _mr_fit_dilations(L, thresholds_per_kernel, max_dilations_per_kernel=32)
    fam_dils["K9s3"] = dils9
    fam_counts["K9s3"] = cnts9

    if "K6s2" in fams:
        cand = np.array([1, 2, 4], dtype=np.int32)
        cap = max(1, min(21, (L - 1) // 5))
        sel = cand[cand <= cap]
        if sel.size == 0:
            sel = np.array([1], dtype=np.int32)
        fam_dils["K6s2"] = sel
        per = max(1, thresholds_per_kernel // fam_dils["K6s2"].size)
        fam_counts["K6s2"] = np.full(fam_dils["K6s2"].size, per, dtype=np.int32)

    if "K3s1" in fams:
        cand = np.array([2, 6], dtype=np.int32)
        fam_dils["K3s1"] = cand[cand <= max(1, (L - 1) // 3)]
        if fam_dils["K3s1"].size == 0:
            fam_dils["K3s1"] = np.array([1], dtype=np.int32)
        per = max(1, thresholds_per_kernel // fam_dils["K3s1"].size)
        fam_counts["K3s1"] = np.full(fam_dils["K3s1"].size, per, dtype=np.int32)

    Xtr_blocks: List[np.ndarray] = []
    Xte_blocks: List[np.ndarray] = []
    parent_info: Dict[ParentKey, ParentInfo] = {}
    parent_best_eta2: Dict[ParentKey, float] = {}
    ranked_parents: List[ParentKey] = []
    feature_registry: Dict[ParentKey, Tuple[int, int]] = {}

    probe_idx = 0
    x_probe = Xtr[probe_idx]

    col_cursor_tr = 0
    col_cursor_te = 0
    for fam_name in fams:
        fam, indices = _family_table(fam_name)
        dils = fam_dils[fam_name]
        cnts = fam_counts[fam_name]

        for d_idx in range(dils.shape[0]):
            d = int(dils[d_idx])
            m_per_kernel = int(cnts[d_idx])
            if m_per_kernel <= 0:
                continue

            # thresholds from RAW probe
            thr_map: List[np.ndarray] = []
            for kid in range(indices.shape[0]):
                w0 = _make_kernel_values(fam.K, indices[kid])
                wd = _dilated_kernel(w0, d)
                a = _conv_valid_dilated1d(x_probe, wd, 1)
                if a.size == 0:
                    thr = np.zeros(0, dtype=np.float32)
                else:
                    q = lds_quantiles(m_per_kernel, offset=lds_offset).astype(np.float64)
                    thr = np.quantile(a.astype(np.float32), q).astype(np.float32)
                thr_map.append(thr)

            # thresholds from DIFF probe (separately)
            x_probe_diff = Xtr_diff2d[probe_idx]
            thr_map_diff: List[np.ndarray] = []
            for kid in range(indices.shape[0]):
                w0 = _make_kernel_values(fam.K, indices[kid])
                wd = _dilated_kernel(w0, d)
                a_diff = _conv_valid_dilated1d(x_probe_diff, wd, 1)
                if a_diff.size == 0:
                    thr_d = np.zeros(0, dtype=np.float32)
                else:
                    q = lds_quantiles(m_per_kernel, offset=lds_offset).astype(np.float64)
                    thr_d = np.quantile(a_diff.astype(np.float32), q).astype(np.float32)
                thr_map_diff.append(thr_d)

            for kid in range(indices.shape[0]):
                pk = ParentKey(fam, d, int(kid))
                thr_base = thr_map[kid]
                thr_diff = thr_map_diff[kid]
                w0 = _make_kernel_values(fam.K, indices[kid])
                wd = _dilated_kernel(w0, d)

                Xtr_block_raw = _collect_features_batch(Xtr, wd, thr_base, include_max_feature)
                Xte_block_raw = _collect_features_batch(Xte, wd, thr_base, include_max_feature)

                Xtr_block_diff = _collect_features_batch(Xtr_diff2d, wd, thr_diff, include_max_feature)
                Xte_block_diff = _collect_features_batch(Xte_diff2d, wd, thr_diff, include_max_feature)

                Xtr_block = np.hstack([Xtr_block_raw, Xtr_block_diff]) if (Xtr_block_raw.size and Xtr_block_diff.size) \
                            else (Xtr_block_raw if Xtr_block_diff.size == 0 else Xtr_block_diff)
                Xte_block = np.hstack([Xte_block_raw, Xte_block_diff]) if (Xte_block_raw.size and Xte_block_diff.size) \
                            else (Xte_block_raw if Xte_block_diff.size == 0 else Xte_block_diff)

                _, eta2 = anova_F_eta2_by_column(Xtr_block, ytr)
                best_eta = float(np.max(eta2)) if eta2.size else 0.0

                Xtr_blocks.append(Xtr_block)
                Xte_blocks.append(Xte_block)

                parent_info[pk] = ParentInfo(fam, d, int(kid), thr_counts=int(thr_base.shape[0]), best_eta2=best_eta)
                parent_best_eta2[pk] = best_eta
                ranked_parents.append(pk)

                col_start = col_cursor_tr
                col_end = col_start + Xtr_block.shape[1]
                feature_registry[pk] = (col_start, col_end)
                col_cursor_tr = col_end
                col_cursor_te += Xte_block.shape[1]

    Xtr_k = np.hstack(Xtr_blocks).astype(np.float32, copy=False) if Xtr_blocks else np.zeros((Xtr.shape[0], 0), dtype=np.float32)
    Xte_k = np.hstack(Xte_blocks).astype(np.float32, copy=False) if Xte_blocks else np.zeros((Xte.shape[0], 0), dtype=np.float32)

    ranked_parents.sort(key=lambda pk: parent_best_eta2[pk], reverse=True)

    names: List[str] = []
    return Xtr_k, Xte_k, names, parent_info, ranked_parents, parent_best_eta2, feature_registry


# ---------------------------
# Step-2: Harvest shapelets (feature-map induced; RAW-only in STAR)
# ---------------------------

def _quantiles_over_shapelet_acts(X_list: Sequence[np.ndarray], w: np.ndarray, m: int, q_offset: float) -> np.ndarray:
    if m <= 0:
        return np.zeros(0, dtype=np.float32)
    qs = lds_quantiles(m, offset=q_offset).astype(np.float64)
    pool: List[float] = []
    for x in X_list:
        a = _conv_valid_dilated1d(np.asarray(x, dtype=np.float32), w, 1)
        if a.size:
            pool.extend(a.tolist())
    if not pool:
        return np.zeros(m, dtype=np.float32)
    pool_arr = np.asarray(pool, dtype=np.float32)
    thr = np.quantile(pool_arr, qs).astype(np.float32)
    return thr

def harvest_shapelets(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    ranked_parents: List[ParentKey],
    parent_info: Dict[ParentKey, ParentInfo],
    parent_best_eta2: Dict[ParentKey, float],
    include_max_feature: bool,
    additional_kernel_families: bool,
    shapelet_budget_total: int = 10000,
    delta_eta2: float = 0.02,
    sigma_optimism: float = 0.25,
    shapelet_bias_mode: str = "reuse",  # "recalibrate_small" | "reuse"
    subsample_frac: float = 0.2,
    subsample_min: int = 10,
    subsample_max: int = 256,
    lds_offset: float = 0.0,
    eb_bias_source: str = "single_series",          # "single_series" | "subseries"
    keep_best_dilation_only: bool = True,
    argmax_map: Optional[Dict[ParentKey, np.ndarray]] = None,
) -> List[ShapeletSpec]:
    """
    Parallel-friendly harvesting (RAW channel only for STAR):
      - cached argmax per (parent, series)
      - EB on a subsample; full-train with SAME biases by default ("reuse")
      - budget in columns; cap total number of shapelets per dataset
    """
    Xtr = np.asarray(X_train, dtype=np.float32)
    ytr = np.asarray(y_train, dtype=np.int64)
    n_tr, L = Xtr.shape

    # shapelet count cap
    C = int(np.unique(ytr).size)
    cap_count = min(max(64, int(math.floor(0.5 * L * C))), 1024)

    stats_per_bias = 4 + (1 if include_max_feature else 0)
    budget_cols_remaining = int(shapelet_budget_total)

    # EB subsample
    sub_n = int(np.clip(math.ceil(subsample_frac * n_tr), subsample_min, subsample_max))
    idx_all = np.arange(n_tr, dtype=np.int32)
    step = max(1, n_tr // max(1, sub_n))
    subsample_idx = idx_all[::step][:sub_n]
    X_sub2d = Xtr[subsample_idx]
    y_sub = ytr[subsample_idx]

    kept: List[ShapeletSpec] = []
    best_eta_by_kernelid: Dict[Tuple[str, int], float] = {}

    for pk in ranked_parents:
        fam = pk.family
        d = int(pk.dilation)
        kid = int(pk.kernel_id)
        span = (fam.K - 1) * d + 1
        if span > L // 2:
            continue

        kern_key = (fam.name, kid)
        best_eta_seen = best_eta_by_kernelid.get(kern_key, -1.0)
        parent_eta = parent_best_eta2.get(pk, 0.0)
        if keep_best_dilation_only and parent_eta + 1e-9 < best_eta_seen:
            continue

        # argmax per series
        if argmax_map is None or pk not in argmax_map:
            indices = {"K9s3": _MR_INDICES, "K6s2": _K6_INDICES, "K3s1": _K3_INDICES}[fam.name]
            table = _argmax_table_single_dilation(Xtr, fam.K, indices, d)
            tstars = table[:, kid]
        else:
            tstars = argmax_map[pk]

        for i in range(n_tr):
            t_star = int(tstars[i])
            if t_star < 0 or t_star + span > L:
                continue
            w = Xtr[i, t_star:t_star + span].astype(np.float32).copy()
            w -= w.mean(dtype=np.float64)

            # EB thresholds on base
            m_allowed = max(2, parent_info[pk].thr_counts)
            if eb_bias_source == "single_series":
                a = _conv_valid_dilated1d(Xtr[i], w, 1)
                if a.size > 0:
                    q = lds_quantiles(m_allowed, offset=lds_offset).astype(np.float64)
                    thr_sub = np.quantile(a.astype(np.float32), q).astype(np.float32)
                else:
                    thr_sub = np.zeros(m_allowed, dtype=np.float32)
            else:
                thr_sub = _quantiles_over_shapelet_acts([row for row in X_sub2d], w, m_allowed, q_offset=lds_offset)

            # EB features on subsample
            Xs = _collect_features_batch(X_sub2d, w, thr_sub, include_max_feature)
            _, eta2_sub = anova_F_eta2_by_column(Xs, y_sub)
            eta2_sub_best = float(np.max(eta2_sub)) if eta2_sub.size else 0.0

            if eta2_sub_best + sigma_optimism < parent_eta + delta_eta2:
                continue

            # Full features with SAME biases (default "reuse")
            if shapelet_bias_mode == "reuse":
                thr_base = thr_sub
            else:
                thr_base = _quantiles_over_shapelet_acts([row for row in X_sub2d], w, m_allowed, q_offset=lds_offset)

            X_full_base = _collect_features_batch(Xtr, w, thr_base, include_max_feature)
            _, eta2_full = anova_F_eta2_by_column(X_full_base, ytr)
            eta2_full_best = float(np.max(eta2_full)) if eta2_full.size else 0.0

            cols_add = stats_per_bias * thr_base.shape[0]
            if cols_add <= 0 or budget_cols_remaining <= 0:
                continue
            if cols_add > budget_cols_remaining:
                take = budget_cols_remaining // stats_per_bias
                take = max(0, min(take, thr_base.shape[0]))
                if take == 0:
                    continue
                thr_base = thr_base[:take]
                cols_add = stats_per_bias * thr_base.shape[0]
                if cols_add <= 0:
                    continue

            kept.append(ShapeletSpec(
                parent=pk, dilation=d, values=w, thr_base=thr_base, thr_diff=np.zeros(0, dtype=np.float32),
                best_eta2=eta2_full_best
            ))
            budget_cols_remaining -= cols_add

            if eta2_full_best > best_eta_seen:
                best_eta_seen = eta2_full_best
                best_eta_by_kernelid[kern_key] = best_eta_seen

            if len(kept) > cap_count:
                worst_idx = int(np.argmin(np.asarray([s.best_eta2 for s in kept], dtype=np.float64)))
                del kept[worst_idx]

            if budget_cols_remaining <= 0:
                return kept

    return kept
