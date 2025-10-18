from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Import STAR-aligned FAST wrapper
from .fast_functions import fast as FastModel
from .fast_utils import ShapeletSpec  # compatibility if needed elsewhere


# -----------------------------
# Shapelet bank discovery (FAST)
# -----------------------------

def _instantiate_fast_for_star(
    *,
    include_max_feature: bool,
    additional_kernel_families: bool,
    kernel_budget_total: Optional[int],
    shapelet_budget_total: Optional[int],
    # EB and harvesting knobs
    delta_eta2: float = 0.02,
    sigma_optimism: float = 0.25,
    shapelet_bias_mode: str = "reuse",
    eb_bias_source: str = "single_series",
    shapelet_subsample_frac: float = 0.20,
    shapelet_subsample_min: int = 10,
    shapelet_subsample_max: int = 256,
    lds_offset: float = 0.0,
) -> FastModel:
    """
    Create a FAST instance with STAR-friendly defaults:
      - kernel_budget_total default 50k (K9, 4 stats) unless user overrides
      - shapelet_budget_total left as provided (or half by FAST default if None)
      - bias_mode='reuse' (your new default)
    """
    if kernel_budget_total is None:
        kernel_budget_total = 50_000

    Fm = FastModel(
        include_max_feature=include_max_feature,
        additional_kernel_families=additional_kernel_families,
        kernel_budget_total=kernel_budget_total,
        shapelet_budget_total=shapelet_budget_total,
        delta_eta2=delta_eta2,
        sigma_optimism=sigma_optimism,
        shapelet_bias_mode=shapelet_bias_mode,
        shapelet_subsample_frac=shapelet_subsample_frac,
        shapelet_subsample_min=shapelet_subsample_min,
        shapelet_subsample_max=shapelet_subsample_max,
        eb_bias_source=eb_bias_source,
        keep_best_dilation_only=True,
        lds_offset=lds_offset,
    )
    return Fm


@dataclass
class StarShapelet:
    """Compact holder for STAR bank entries."""
    values: np.ndarray       # (span,)
    length: int              # span
    var_idx: int             # which variable/channel this shapelet belongs to
    best_eta2: float         # from FAST
    parent_family: str = ""
    parent_dilation: int = 1


def discover_shapelet_bank_for_star(
    X_train_1var: np.ndarray,
    y_train: np.ndarray,
    *,
    var_idx: int,
    include_max_feature: bool = False,
    additional_kernel_families: bool = True,
    kernel_budget_total: Optional[int] = None,
    shapelet_budget_total: Optional[int] = None,
    top_k_per_var: Optional[int] = None,
    fast_kwargs: Optional[Dict[str, Any]] = None,
) -> List[StarShapelet]:
    """
    Run FAST on a *single variable* (univariate) training matrix X_train_1var=(n,L) to harvest a bank.
    Assumes FAST returns shapelets already sorted by discriminativeness. We just *slice* to top_k_per_var.
    """
    fast_kwargs = fast_kwargs or {}

    Fm = _instantiate_fast_for_star(
        include_max_feature=include_max_feature,
        additional_kernel_families=additional_kernel_families,
        kernel_budget_total=kernel_budget_total,   # defaulted to 50k if None
        shapelet_budget_total=shapelet_budget_total,
        **fast_kwargs,
    )

    # We only need harvesting; assemble with features="shapelets" to run Stage-2 and keep kept_shapelets_.
    _Xtr, _Xte, _ = Fm.assemble_features(X_train_1var, y_train, X_train_1var, features_used="shapelets")
    kept = Fm.kept_shapelets_  # list of ShapeletSpec, already eta^2-ranked

    # Optional top-k slice (no re-ranking needed)
    if top_k_per_var is not None and top_k_per_var > 0:
        kept = kept[: top_k_per_var]

    out: List[StarShapelet] = []
    for s in kept:
        span = s.values.shape[0]
        fam_name = getattr(s.parent.family, "name", "")
        out.append(
            StarShapelet(
                values=s.values.astype(np.float32, copy=False),
                length=int(span),
                var_idx=int(var_idx),
                best_eta2=float(getattr(s, "best_eta2", 0.0)),
                parent_family=fam_name,
                parent_dilation=int(getattr(s, "dilation", 1)),
            )
        )
    return out


def discover_multivar_shapelet_bank_for_star(
    X_train: np.ndarray,    # (n, T, V)
    y_train: np.ndarray,
    *,
    include_max_feature: bool = False,
    additional_kernel_families: bool = True,
    kernel_budget_total: Optional[int] = None,
    shapelet_budget_total: Optional[int] = None,
    top_k_per_var: Optional[int] = None,
    fast_kwargs: Optional[Dict[str, Any]] = None,
) -> List[StarShapelet]:
    """
    Run FAST independently per variable and concatenate top-k per variable.
    """
    X = np.asarray(X_train, dtype=np.float32)
    n, T, V = X.shape
    bank: List[StarShapelet] = []
    for v in range(V):
        Xv = X[:, :, v]  # (n, T)
        bank_v = discover_shapelet_bank_for_star(
            Xv, y_train,
            var_idx=v,
            include_max_feature=include_max_feature,
            additional_kernel_families=additional_kernel_families,
            kernel_budget_total=kernel_budget_total,
            shapelet_budget_total=shapelet_budget_total,
            top_k_per_var=top_k_per_var,
            fast_kwargs=fast_kwargs,
        )
        bank.extend(bank_v)
    return bank


# -----------------------------------
# Batched Activations (ZNCC) + Masking
# -----------------------------------

def _group_bank_by_var_len(
    bank: Sequence[StarShapelet],
    device: str,
    dtype: torch.dtype,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Group shapelets by (var_idx, length) and prebuild a conv weight tensor per group.
    Returns: dict[(v, L)] -> dict with:
      - 'idx': List[int] column indices in Z for this group's shapelets
      - 'W':   torch.Tensor (G, 1, L) conv weights (reversed, z-normed)
      - 'L':   int length
      - 'c':   int center offset = L // 2
    """
    groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for j, shp in enumerate(bank):
        v = int(shp.var_idx)
        L = int(shp.length)
        key = (v, L)
        if key not in groups:
            groups[key] = {"idx": [], "ws": []}
        groups[key]["idx"].append(j)
        s = torch.from_numpy(shp.values.astype(np.float32))
        s = (s - s.mean()).div(s.std(unbiased=False).clamp(min=1e-6))
        s = torch.flip(s, dims=[0]).view(1, 1, L)  # reverse for conv
        groups[key]["ws"].append(s)
    for key, g in groups.items():
        W = torch.cat(g["ws"], dim=0).to(device=device, dtype=dtype)  # (G,1,L)
        L = int(W.shape[-1]); c = L // 2
        groups[key] = {"idx": g["idx"], "W": W, "L": L, "c": c}
    return groups


def _rolling_sd_batched(xv: torch.Tensor, L: int, eps: float = 1e-8) -> torch.Tensor:
    """
    xv: (N,1,T) -> sd: (N,1,S) where S=T-L+1
    """
    ones = torch.ones((1, 1, L), device=xv.device, dtype=xv.dtype)
    s  = F.conv1d(xv, ones, stride=1, padding=0)            # sum
    s2 = F.conv1d(xv * xv, ones, stride=1, padding=0)       # sum of squares
    mu = s / float(L)
    var = (s2 / float(L)) - mu * mu
    return torch.sqrt(var.clamp(min=eps))


def build_activation_grid(
    X_batch: np.ndarray,                # (N,T,V) or (T,V)
    bank: Sequence[StarShapelet],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    z_norm_input: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched ZNCC for STAR with shapelet-specific masks.
    Returns:
      Z       : (N, T, K) activations (zeros where invalid)
      time_ms : (N, T)   time mask (True if *all* shapelets invalid at that t)
      ch_ms   : (N, T, K) channel-wise mask (True where that shapelet is invalid at that t)
    """
    Xb = np.asarray(X_batch, dtype=np.float32)
    if Xb.ndim == 2:
        Xb = Xb[None, ...]
    N, T, V = Xb.shape
    K = len(bank)

    Z = torch.zeros((N, T, K), device=device, dtype=dtype)
    if K == 0:
        m_time = torch.zeros((N, T), dtype=torch.bool, device=device)
        m_ch   = torch.ones((N, T, 0), dtype=torch.bool, device=device)
        return Z, m_time, m_ch

    x = torch.from_numpy(Xb).to(device=device, dtype=dtype)  # (N,T,V)
    if z_norm_input:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        x = (x - mu) / sd
    x = x.permute(0, 2, 1).contiguous()  # (N,V,T)

    # Per-shapelet (channel) mask: start as all True (invalid), then mark placed regions False
    m_ch = torch.ones((N, T, K), dtype=torch.bool, device=device)

    groups = _group_bank_by_var_len(bank, device=device, dtype=dtype)

    for (v, L), g in groups.items():
        if L > T:
            continue
        idx_cols = torch.tensor(g["idx"], device=device, dtype=torch.long)  # (G,)
        W = g["W"].to(dtype=torch.float32)              # (G,1,L) ensure fp32 compute
        xv = x[:, v:v+1, :].to(dtype=torch.float32)     # (N,1,T) ensure fp32 compute

        # --- numerically stable ZNCC in full float32 (disable autocast) ---
        with torch.amp.autocast("cuda", enabled=False):
            sd_x = _rolling_sd_batched(xv, L)                 # (N,1,S) fp32
            denom = (float(L) * sd_x).clamp(min=1e-6)         # clamp in fp32
            dots  = F.conv1d(xv, W, stride=1, padding=0)      # (N,G,S) fp32
            a     = dots / denom                               # (N,G,S) fp32

        # sanitize: remove any numerical junk just in case
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).to(dtype)
        S = a.shape[-1]
        t0 = L // 2

        # place into Z and unmask only the valid window for these channels
        Z[:, t0:t0+S, idx_cols] = a.permute(0, 2, 1).contiguous()  # (N,S,G)
        m_ch[:, t0:t0+S, idx_cols] = False

    # time mask = positions where *all* shapelets are invalid
    m_time = m_ch.all(dim=2)  # (N,T)
    return Z, m_time, m_ch


# -----------------------------------------
# Batched, mask-aware standardization (per-channel)
# -----------------------------------------

def standardize_activation_grid(
    Z: torch.Tensor,                 # (N,T,K) or (T,K)
    time_mask: torch.Tensor,         # (N,T) or (T,)
    ch_mask: Optional[torch.Tensor] = None,  # (N,T,K) True=invalid (preferred)
) -> torch.Tensor:
    """
    Mask-aware per-channel z-score over time.
    If ch_mask is provided (N,T,K), it is used for statistics.
    Otherwise, time_mask is broadcast to (N,T,1).
    Masked positions in output are set to 0.
    """
    if Z.ndim == 2 and time_mask.ndim == 1:
        Z = Z.unsqueeze(0)
        time_mask = time_mask.unsqueeze(0)

    assert Z.dim() == 3 and time_mask.dim() == 2 and Z.shape[:2] == time_mask.shape
    N, T, K = Z.shape

    if ch_mask is not None:
        assert ch_mask.shape == (N, T, K)
        valid = (~ch_mask).to(Z.dtype)  # (N,T,K)
    else:
        valid = (~time_mask).to(Z.dtype).unsqueeze(-1)  # (N,T,1) -> broadcast

    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (N,1,K)
    mu = (Z * valid).sum(dim=1, keepdim=True) / denom      # (N,1,K)
    Zc = Z - mu
    var = ((Zc * valid) * Zc).sum(dim=1, keepdim=True) / denom  # (N,1,K)
    sd = var.sqrt().clamp(min=1e-6)
    Zz = Zc / sd

    # sanitize & zero masked positions
    Zz = torch.nan_to_num(Zz, nan=0.0, posinf=0.0, neginf=0.0)
    Zz = Zz * valid
    return Zz if N > 1 else Zz[0]


# -----------------------------------------
# Generic encoder inputs (raw / first-diff)
# -----------------------------------------

def build_generic_inputs(
    X_batch: np.ndarray,                # (N,T,V) or (T,V)
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    z_norm: bool = True,
    include_diff: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build tensors for the generic encoders:
      - raw_std: per-sample, per-variable z-normed raw series (B,T,V)
      - diff_std: per-sample, per-variable z-normed first-differences, padded to T (B,T,V) if include_diff
    """
    Xb = np.asarray(X_batch, dtype=np.float32)
    if Xb.ndim == 2:
        Xb = Xb[None, ...]
    N, T, V = Xb.shape

    x = torch.from_numpy(Xb).to(device=device, dtype=dtype)  # (B,T,V)

    if z_norm:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        x_raw = (x - mu) / sd
    else:
        x_raw = x

    x_diff = None
    if include_diff:
        # first differences along time, pad at the front to keep length T
        d = torch.diff(x, dim=1)                               # (B,T-1,V)
        d = torch.cat([torch.zeros((N, 1, V), device=x.device, dtype=x.dtype), d], dim=1)
        if z_norm:
            mu_d = d.mean(dim=1, keepdim=True)
            sd_d = d.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
            x_diff = (d - mu_d) / sd_d
        else:
            x_diff = d

    return x_raw, x_diff

# =========================================================
#  Optimizer
# =========================================================

import math
from typing import Iterable, Union

class _RAdam(torch.optim.Optimizer):
    """
    Drop-in RAdam (for PyTorch builds that lack torch.optim.RAdam).
    Matches the canonical formulation (Liu et al., 2019).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                t = state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                beta2_t = beta2 ** t
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)

                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"] * group["lr"])

                if N_sma >= 5:
                    step_size = group["lr"] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma
                    ) / (1 - beta1 ** t)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    step_size = group["lr"] / (1 - beta1 ** t)
                    p.add_(exp_avg, alpha=-step_size)
        return loss


def make_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    name: str = "radam",
    lr: float = 1e-2,               
    weight_decay: float = 5e-4,     
    betas: Tuple[float, float] = (0.9, 0.999)
) -> torch.optim.Optimizer:
    """
    Create an optimizer with ShapeFormer-style defaults.
    Usage:
        opt = make_optimizer(model.parameters(), name="radam")
    """
    name = name.lower()
    if name == "radam":
        if hasattr(torch.optim, "RAdam"):
            return torch.optim.RAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            return _RAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")
