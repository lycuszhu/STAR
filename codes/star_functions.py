from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .star_utils import (
    StarShapelet,
    discover_multivar_shapelet_bank_for_star,
    build_activation_grid,
    standardize_activation_grid,
    build_generic_inputs,
)



# =========================================================
#  A) Shapelet encoder
# =========================================================

class ShapeletEncoder(nn.Module):
    """
    Tokenize time by projecting the K-channel shapelet-activation grid at each t
    from R^K -> R^d_model, add a learnable absolute PE, and run a norm-first
    TransformerEncoder. Pool via [CLS] or attention pooling.
    """
    def __init__(
        self,
        k_channels: int,
        *,
        d_model: int = 128,
        n_heads: int = 16,         
        n_layers: int = 1,         
        ff_mult: int = 4,
        dropout: float = 0.4,      
        pooling: str = "cls",      # "cls" | "attn"
        max_len: int = 4096,
    ):
        super().__init__()
        assert pooling in ("cls", "attn")
        self.pooling = pooling

        # Linear token projection K -> d
        self.in_proj = nn.Linear(k_channels, d_model, bias=True)
        self.ln_in   = nn.LayerNorm(d_model)

        # Learnable absolute positional embeddings
        self.pos = nn.Embedding(max_len + 1, d_model)  # +1 in case [CLS]
        self.dropout = nn.Dropout(dropout)

        # Norm-first TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(ff_mult * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.pool_ln = nn.LayerNorm(d_model)
            self.pool_scorer = nn.Linear(d_model, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos.weight, std=0.02)
        if self.pooling == "cls":
            nn.init.zeros_(self.cls_token)

    def forward(self, Zt: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Zt:        (B, T, K)   standardized shapelet activations
        time_mask: (B, T) bool True where time step is pad/invalid (no evidence)
        """
        B, T, K = Zt.shape
        x = self.in_proj(Zt)              # (B,T,d)
        x = self.ln_in(x)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)              # (B,1,d)
            x = torch.cat([cls, x], dim=1)                      # (B,1+T,d)
            pe = self.pos(torch.arange(T + 1, device=x.device)) # (1+T,d)
            key_padding_mask = None
            if time_mask is not None:
                pad = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([pad, time_mask], dim=1)  # (B,1+T)
        else:
            pe = self.pos(torch.arange(T, device=x.device))
            key_padding_mask = time_mask

        x = x + pe.unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.dropout(x)

        if self.pooling == "cls":
            return x[:, 0, :]  # (B,d)
        else:
            hs = self.pool_ln(x)
            logits = self.pool_scorer(hs).squeeze(-1)  # (B,T)
            if key_padding_mask is not None:
                logits = logits.masked_fill(key_padding_mask, float("-inf"))
            w = torch.softmax(logits, dim=1).unsqueeze(-1)
            return (w * x).sum(dim=1)  # (B,d)


# =========================================================
#  B) Generic encoder
#      Conv(temporal 1x8, depthwise) -> BN -> GELU
#      Conv(1x1, channel mix)        -> BN -> GELU
#      + abs PE -> Transformer -> avg pool
# =========================================================

class GenericEncoder(nn.Module):
    """
    ShapeFormer-style generic stream:
      - Block1: temporal depthwise Conv1d(kernel=8) + BN + GELU
      - Block2: pointwise Conv1d(1x1) to mix variables + BN + GELU
      - Learnable absolute PE
      - Norm-first TransformerEncoder
      - Temporal average pooling
    """
    def __init__(
        self,
        v_channels: int,
        *,
        d_model: int = 32,          
        n_heads: int = 16,          
        n_layers: int = 1,          
        ff_mult: int = 4,
        dropout: float = 0.4,       
        max_len: int = 4096,
        k_temporal: int = 8,        
    ):
        super().__init__()
        V = int(v_channels)
        self.v_channels = V
        self.dropout = nn.Dropout(dropout)

        # Block 1: temporal depthwise conv on each variable channel
        self.temporal = nn.Conv1d(
            in_channels=V, out_channels=V, kernel_size=k_temporal,
            stride=1, padding=k_temporal // 2, groups=V, bias=False
        )
        self.bn1 = nn.BatchNorm1d(V)
        self.act1 = nn.GELU()

        # Block 2: pointwise conv to mix variables -> d_model
        self.pointwise = nn.Conv1d(V, d_model, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.GELU()

        # Absolute PE before attention
        self.pos = nn.Embedding(max_len, d_model)

        # Norm-first TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(ff_mult * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos.weight, std=0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, T, V) z-normalized raw multivariate signal
        Returns: (B, d_model)
        """
        B, T, V = X.shape
        x = X.transpose(1, 2)           # (B,V,T)
        x = self.temporal(x)            # (B,V,T')  NOTE: with k=8, padding=4 => T' = T+1
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pointwise(x)           # (B,d,T')
        x = self.bn2(x)
        x = self.act2(x)
        x = x.transpose(1, 2)           # (B,T',d_model)

        Tx = x.size(1)                  # actual length after conv(s)
        # Safety: if positional table is too short, crop x (shouldn't happen with max_len bump)
        if Tx > self.pos.num_embeddings:
            x = x[:, : self.pos.num_embeddings, :]
            Tx = x.size(1)

        pos_emb = self.pos(torch.arange(Tx, device=X.device)).unsqueeze(0)  # (1,Tx,d)
        x = x + pos_emb
        x = self.encoder(x)             # (B,Tx,d)
        x = self.dropout(x)
        return x.mean(dim=1)            # GAP over time


# =========================================================
#  C) STAR Fusion model
# =========================================================

class StarFusionModel(nn.Module):
    def __init__(
        self,
        *,
        k_channels: int,
        v_channels: int,
        num_classes: int,
        # shapelet encoder
        shp_d_model: int = 128,
        shp_heads: int = 16,        
        shp_layers: int = 1,        
        shp_ff_mult: int = 4,
        shp_dropout: float = 0.4,   
        shp_pooling: str = "cls",
        # generic encoders
        enable_generic_raw: bool = True,
        enable_generic_diff: bool = False,
        gen_d_model: int = 32,      
        gen_heads: int = 16,        
        gen_layers: int = 1,        
        gen_ff_mult: int = 4,
        gen_dropout: float = 0.4,   
        # others
        max_len: int = 4096,
    ):
        super().__init__()
        self.enable_generic_raw = bool(enable_generic_raw)
        self.enable_generic_diff = bool(enable_generic_diff)

        self.shp_enc = ShapeletEncoder(
            k_channels=k_channels,
            d_model=shp_d_model, n_heads=shp_heads, n_layers=shp_layers,
            ff_mult=shp_ff_mult, dropout=shp_dropout, pooling=shp_pooling,
            max_len=max_len,
        )

        if self.enable_generic_raw:
            self.gen_raw = GenericEncoder(
                v_channels=v_channels,
                d_model=gen_d_model, n_heads=gen_heads, n_layers=gen_layers,
                ff_mult=gen_ff_mult, dropout=gen_dropout, max_len=max_len + 8,  # <-- was max_len
            )
        else:
            self.gen_raw = None

        if self.enable_generic_diff:
            self.gen_diff = GenericEncoder(
                v_channels=v_channels,
                d_model=gen_d_model, n_heads=gen_heads, n_layers=gen_layers,
                ff_mult=gen_ff_mult, dropout=gen_dropout, max_len=max_len + 8,  # <-- was max_len
            )
        else:
            self.gen_diff = None


        d_total = shp_d_model \
                  + (gen_d_model if self.enable_generic_raw else 0) \
                  + (gen_d_model if self.enable_generic_diff else 0)

        self.head = nn.Linear(d_total, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        *,
        shp_seq: torch.Tensor,             # (B,T,K) standardized activations
        shp_time_mask: Optional[torch.Tensor] = None,  # (B,T) bool
        X_raw: Optional[torch.Tensor] = None,          # (B,T,V)
        X_diff: Optional[torch.Tensor] = None,         # (B,T,V)
    ) -> torch.Tensor:
        h_shp = self.shp_enc(shp_seq, shp_time_mask)           # (B, d_shp)
        hs = [h_shp]
        if self.enable_generic_raw:
            assert X_raw is not None
            hs.append(self.gen_raw(X_raw))
        if self.enable_generic_diff:
            assert X_diff is not None
            hs.append(self.gen_diff(X_diff))
        h = torch.cat(hs, dim=-1)
        return self.head(h)  # logits (B,C)


# =========================================================
#  D) Factory from bundle
# =========================================================

@dataclass
class StarBundle:
    bank: List[StarShapelet]
    # shapelet tensors
    Z_train: torch.Tensor
    mask_train: torch.Tensor
    Z_val: Optional[torch.Tensor] = None
    mask_val: Optional[torch.Tensor] = None
    Z_test: Optional[torch.Tensor] = None
    mask_test: Optional[torch.Tensor] = None
    # generic tensors
    Xtr_raw_std: Optional[torch.Tensor] = None
    Xtr_diff_std: Optional[torch.Tensor] = None
    Xva_raw_std: Optional[torch.Tensor] = None
    Xva_diff_std: Optional[torch.Tensor] = None
    Xte_raw_std: Optional[torch.Tensor] = None
    Xte_diff_std: Optional[torch.Tensor] = None


def build_star_from_fast(
    X_train: np.ndarray,   # (N,T,V) -- DO NOT z-norm here for FAST
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    include_max_feature: bool = False,
    additional_kernel_families: bool = True,
    kernel_budget_total: Optional[int] = 50_000,
    shapelet_budget_total: Optional[int] = None,
    top_k_per_var: Optional[int] = None,
    fast_kwargs: Optional[Dict] = None,
    device: str = "cpu",
    # STAR activations (ZNCC) optional z-norm (usually False when coming from FAST path)
    z_norm_input_for_activations: bool = False,
    # Generic encoders config
    enable_generic_raw: bool = True,
    enable_generic_diff: bool = False,
    z_norm_generic_inputs: bool = True,
) -> StarBundle:
    """
    Returns a bundle with *all* splits if X_val is provided:
      - Train: Z_train/mask_train (+ generic raw/diff)
      - Val  : Z_val/mask_val (+ generic raw/diff)
      - Test : Z_test/mask_test (+ generic raw/diff)
    """
    bank = discover_multivar_shapelet_bank_for_star(
        X_train, y_train,
        include_max_feature=include_max_feature,
        additional_kernel_families=additional_kernel_families,
        kernel_budget_total=kernel_budget_total,
        shapelet_budget_total=shapelet_budget_total,
        top_k_per_var=top_k_per_var,
        fast_kwargs=fast_kwargs,
    )

    # numpy views
    Xtr = np.asarray(X_train, dtype=np.float32)
    Xte = np.asarray(X_test, dtype=np.float32)
    Xva = None if X_val is None else np.asarray(X_val, dtype=np.float32)

    # --- Train Z/masks
    Ztr, mtr, mch_tr = build_activation_grid(
        Xtr, bank, device=device, dtype=torch.float32, z_norm_input=z_norm_input_for_activations
    )
    Ztr = standardize_activation_grid(Ztr, mtr, ch_mask=mch_tr)

    # --- Val Z/masks (optional)
    Zva = mva = None
    if Xva is not None:
        Zva, mva, mch_va = build_activation_grid(
            Xva, bank, device=device, dtype=torch.float32, z_norm_input=z_norm_input_for_activations
        )
        Zva = standardize_activation_grid(Zva, mva, ch_mask=mch_va)

    # --- Test Z/masks
    Zte, mte, mch_te = build_activation_grid(
        Xte, bank, device=device, dtype=torch.float32, z_norm_input=z_norm_input_for_activations
    )
    Zte = standardize_activation_grid(Zte, mte, ch_mask=mch_te)

    # --- Generic inputs (optional)
    Xtr_raw_std = Xtr_diff_std = Xva_raw_std = Xva_diff_std = Xte_raw_std = Xte_diff_std = None
    if enable_generic_raw or enable_generic_diff:
        Xtr_raw_std, Xtr_diff_std = build_generic_inputs(
            Xtr, device=device, dtype=torch.float32, z_norm=z_norm_generic_inputs, include_diff=enable_generic_diff
        )
        if Xva is not None:
            Xva_raw_std, Xva_diff_std = build_generic_inputs(
                Xva, device=device, dtype=torch.float32, z_norm=z_norm_generic_inputs, include_diff=enable_generic_diff
            )
        Xte_raw_std, Xte_diff_std = build_generic_inputs(
            Xte, device=device, dtype=torch.float32, z_norm=z_norm_generic_inputs, include_diff=enable_generic_diff
        )

    return StarBundle(
        bank=bank,
        Z_train=Ztr, mask_train=mtr,
        Z_val=Zva,  mask_val=mva,
        Z_test=Zte, mask_test=mte,
        Xtr_raw_std=Xtr_raw_std, Xtr_diff_std=Xtr_diff_std,
        Xva_raw_std=Xva_raw_std, Xva_diff_std=Xva_diff_std,
        Xte_raw_std=Xte_raw_std, Xte_diff_std=Xte_diff_std,
    )

def make_star_model_from_bundle(
    bundle: StarBundle,
    *,
    num_classes: int,
    shp_d_model: int = 128,
    shp_heads: int = 16,
    shp_layers: int = 1,
    shp_ff_mult: int = 4,
    shp_dropout: float = 0.4,
    shp_pooling: str = "cls",
    enable_generic_raw: bool = True,
    enable_generic_diff: bool = False,
    gen_d_model: int = 32,
    gen_heads: int = 16,
    gen_layers: int = 1,
    gen_ff_mult: int = 4,
    gen_dropout: float = 0.4,
) -> StarFusionModel:
    _, T, K = bundle.Z_train.shape
    if bundle.Xtr_raw_std is not None:
        V = bundle.Xtr_raw_std.shape[-1]
    elif bundle.Xtr_diff_std is not None:
        V = bundle.Xtr_diff_std.shape[-1]
    else:
        V = 1

    return StarFusionModel(
        k_channels=K, v_channels=V, num_classes=num_classes,
        shp_d_model=shp_d_model, shp_heads=shp_heads, shp_layers=shp_layers,
        shp_ff_mult=shp_ff_mult, shp_dropout=shp_dropout, shp_pooling=shp_pooling,
        enable_generic_raw=enable_generic_raw,
        enable_generic_diff=enable_generic_diff,
        gen_d_model=gen_d_model, gen_heads=gen_heads, gen_layers=gen_layers,
        gen_ff_mult=gen_ff_mult, gen_dropout=gen_dropout,
        max_len=T,
    )
