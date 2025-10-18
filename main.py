# -----------------------
# How to run (ShapeFormer-aligned defaults):
#
# 1) Standard small/medium dataset:
#   python -m tests.run_bench \
#     --dataset BasicMotions \
#     --epochs 200 --batch_size 16 \
#     --d_model 128 --n_layers 1 --n_heads 16 --pooling cls --dropout 0.4 \
#     --gen_raw 1 --gen_diff 0 --gen_d_model 32 --gen_layers 1 --gen_heads 16 --gen_dropout 0.4 \
#     --optimizer radam --lr 1e-2 --weight_decay 5e-4 \
#     --include_max 0 --k369 1 --kernel_budget_total 50000 --top_k_per_var 3 \
#     --threads 20 --use_val 1 --val_frac 0.2 --early_stop_patience 5 --early_stop_min_delta 1e-4
#
# 2) Large-V dataset (control shapelets per variable):
#   python -m tests.run_bench \
#     --dataset YourLargeVDataset \
#     --epochs 200 --batch_size 16 \
#     --d_model 128 --n_layers 1 --n_heads 16 --dropout 0.4 \
#     --gen_raw 1 --gen_d_model 32 --gen_layers 1 --gen_heads 16 --gen_dropout 0.4 \
#     --optimizer radam --lr 1e-2 --weight_decay 5e-4 \
#     --top_k_per_var 3 --kernel_budget_total 50000 \
#     --threads 20 --use_val 1 --val_frac 0.2
# -----------------------

from __future__ import annotations
import argparse, time, random
from typing import Tuple, Dict, Any, List
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from codes.fast_utils import (
    load_ucr_uea_sktime,
    split_train_for_tuning,
    set_numba_threads_count,
)
from codes.star_functions import (
    build_star_from_fast,
    make_star_model_from_bundle,
)
from codes.star_utils import make_optimizer  # NEW: use RAdam/AdamW via helper

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loader_fusion(
    *,
    Z: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    X_raw_std: torch.Tensor | np.ndarray | None = None,
    X_diff_std: torch.Tensor | np.ndarray | None = None,
) -> DataLoader:
    def _to_t(x, dtype=None):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype if dtype is not None else torch.float32)
        return x

    Zt = _to_t(Z, torch.float32)
    Mt = _to_t(mask, torch.bool)
    yt = torch.tensor(y, dtype=torch.long)

    Xr = _to_t(X_raw_std, torch.float32)
    Xd = _to_t(X_diff_std, torch.float32)

    # Keep tuple shape fixed for collate; placeholders ignored when streams are disabled
    if Xr is None:
        Xr = torch.zeros((Zt.size(0), 1, 1), dtype=torch.float32)
    if Xd is None:
        Xd = torch.zeros((Zt.size(0), 1, 1), dtype=torch.float32)

    ds = TensorDataset(Zt, Mt, Xr, Xd, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def _summarize_bank_lengths(bank, prefix="[bank]"):
    if not bank:
        print(f"{prefix} K=0 (no shapelets)")
        return
    lens = np.array([int(s.length) for s in bank], dtype=np.int32)
    print(f"{prefix} K={lens.size} | len[min,max,mean]=({lens.min()},{lens.max()},{lens.mean():.1f})")

def _summarize_mask(name: str, mask: torch.Tensor | np.ndarray):
    m = torch.as_tensor(mask, dtype=torch.bool)
    valid_counts = (~m).sum(dim=1)
    print(f"[mask:{name}] valid_t_per_sample: min={int(valid_counts.min())} max={int(valid_counts.max())}")
    if int(valid_counts.min()) <= 0:
        bad = (valid_counts == 0).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"[mask:{name}] ERROR: samples with ALL timesteps masked (idx={bad})."
        )

def _check_finite(name, Z):
    Zt = Z if isinstance(Z, torch.Tensor) else torch.as_tensor(Z)
    bad = (~torch.isfinite(Zt)).sum().item()
    if bad > 0:
        print(f"[finite:{name}] non-finite entries: {bad}")

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    return_loss: bool = False,
    enable_generic_raw: bool = True,
    enable_generic_diff: bool = False,
) -> Tuple[float, float] | float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    all_y, all_pred = [], []
    for Z, mask, Xr, Xd, y in loader:
        Z = Z.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        Xr = Xr.to(device, non_blocking=True) if enable_generic_raw else None
        Xd = Xd.to(device, non_blocking=True) if enable_generic_diff else None

        # NEW API names
        logits = model(shp_seq=Z, shp_time_mask=mask, X_raw=Xr, X_diff=Xd)

        if return_loss:
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)

        pred = torch.argmax(logits, dim=1)
        all_y.extend(y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(all_y, all_pred)
    if return_loss:
        return (total_loss / max(1, total_n), acc)
    return acc

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    enable_generic_raw: bool = True,
    enable_generic_diff: bool = False,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    for Z, mask, Xr, Xd, y in loader:
        Z = Z.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        Xr = Xr.to(device, non_blocking=True) if enable_generic_raw else None
        Xd = Xd.to(device, non_blocking=True) if enable_generic_diff else None

        optimizer.zero_grad(set_to_none=True)
        # NEW API names
        logits = model(shp_seq=Z, shp_time_mask=mask, X_raw=Xr, X_diff=Xd)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)
    return total_loss / max(1, total_n)

# -----------------------
# CSV helper 
# -----------------------
def _append_result_csv(args, K: int, final_acc: float, best_acc: float, elapsed_s: float):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "tuning_results.csv"

    row = {
        "dataset": args.dataset,
        "group": "manual",
        "status": "ok",
        "elapsed_sec": f"{elapsed_s:.2f}",
        "include_max": str(args.include_max),
        "k369": str(args.k369),
        "pooling": args.pooling,
        "d_model": str(args.d_model),
        "n_layers": str(args.n_layers),
        "n_heads": str(args.n_heads),
        "dropout": f"{args.dropout}",
        "gen_raw": str(args.gen_raw),
        "gen_diff": str(args.gen_diff),
        "gen_d_model": str(args.gen_d_model),
        "gen_layers": str(args.gen_layers),
        "gen_heads": str(args.gen_heads),
        "gen_dropout": f"{args.gen_dropout}",
        "optimizer": args.optimizer,
        "lr": f"{args.lr}",
        "weight_decay": f"{args.weight_decay}",
        "batch_size": str(args.batch_size),
        "K": str(K),
        "final_test_acc": f"{final_acc:.6f}",
        "best_test_acc": f"{best_acc:.6f}",
        "best_epoch": "",
        "params_key": "",
        "log_path": "",
        "error": "",
    }

    exists = out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)

    # Shapelet encoder (STAR)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=16)      # ShapeFormer default
    ap.add_argument("--n_layers", type=int, default=1)      # ShapeFormer default
    ap.add_argument("--dropout", type=float, default=0.4)   # ShapeFormer best
    ap.add_argument("--pooling", type=str, choices=["cls", "attn"], default="cls")

    # Generic encoders (raw/diff)
    ap.add_argument("--gen_raw", type=int, default=1)
    ap.add_argument("--gen_diff", type=int, default=0)
    ap.add_argument("--gen_d_model", type=int, default=32)  # ShapeFormer best
    ap.add_argument("--gen_heads", type=int, default=16)    # ShapeFormer default
    ap.add_argument("--gen_layers", type=int, default=1)    # ShapeFormer default
    ap.add_argument("--gen_dropout", type=float, default=0.4)

    # Training
    ap.add_argument("--epochs", type=int, default=200)      # ShapeFormer
    ap.add_argument("--batch_size", type=int, default=16)   # ShapeFormer
    ap.add_argument("--optimizer", type=str, default="radam", choices=["radam", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-2)       # ShapeFormer
    ap.add_argument("--weight_decay", type=float, default=5e-4)  # ShapeFormer

    # Early stopping on val
    ap.add_argument("--use_val", type=int, default=1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    # FAST discovery knobs
    ap.add_argument("--include_max", type=int, default=0)
    ap.add_argument("--k369", type=int, default=1)
    ap.add_argument("--kernel_budget_total", type=int, default=None)
    ap.add_argument("--shapelet_budget_total", type=int, default=None)
    ap.add_argument("--top_k_per_var", type=int, default=None)

    # Data loading
    ap.add_argument("--z_norm", type=int, default=0)

    args = ap.parse_args()

    set_seed(42)
    if args.threads and args.threads > 0:
        set_numba_threads_count(args.threads)

    device = args.device
    print(f"[config] dataset={args.dataset} device={device} "
          f"| shp(d={args.d_model}, L={args.n_layers}, H={args.n_heads}, drop={args.dropout}, pool={args.pooling}) "
          f"| gen_raw={args.gen_raw} gen_diff={args.gen_diff} "
          f"| gen(d={args.gen_d_model}, L={args.gen_layers}, H={args.gen_heads}, drop={args.gen_dropout}) "
          f"| opt={args.optimizer} lr={args.lr} wd={args.weight_decay} bs={args.batch_size}")

    # 1) Load data
    X_train, y_train, X_test, y_test = load_ucr_uea_sktime(
        args.dataset, output="vtv", z_norm=bool(args.z_norm), merge_and_resplit=False
    )
    N, T, V = X_train.shape
    C = int(np.unique(y_train).size)
    print(f"[data] Train {X_train.shape} | Test {X_test.shape} | Classes={C}")

    # 2) Optional train/val split
    if args.use_val:
        X_tr, y_tr, X_val, y_val = split_train_for_tuning(
            X_train, y_train, val_frac=args.val_frac, random_state=42, stratify=True
        )
    else:
        X_tr, y_tr = X_train, y_train
        X_val, y_val = None, None

    # 3) Build STAR tensors (bank + activations + generic inputs)
    wall0 = time.time()
    t0 = time.time()
    bundle = build_star_from_fast(
        X_train=X_tr, y_train=y_tr, X_test=X_test, X_val=X_val,
        include_max_feature=bool(args.include_max),
        additional_kernel_families=bool(args.k369),
        kernel_budget_total=args.kernel_budget_total,
        shapelet_budget_total=args.shapelet_budget_total,
        top_k_per_var=args.top_k_per_var,
        device=device,
        # STAR activations z-norm (keep False for FAST path)
        z_norm_input_for_activations=False,
        # Generic encoders
        enable_generic_raw=bool(args.gen_raw),
        enable_generic_diff=bool(args.gen_diff),
        z_norm_generic_inputs=True,
    )
    asm_s = time.time() - t0

    Ztr, mtr = bundle.Z_train, bundle.mask_train
    if args.use_val:
        Zva, mva = bundle.Z_val, bundle.mask_val
    Zte, mte = bundle.Z_test, bundle.mask_test

    print(f"[bank] K={Ztr.shape[-1]} shapelets | assemble={asm_s:.2f}s")

    # 4) Make model (new defaults already ShapeFormer-aligned)
    model = make_star_model_from_bundle(
        bundle, num_classes=C,
        # shapelet encoder
        shp_d_model=args.d_model, shp_heads=args.n_heads, shp_layers=args.n_layers,
        shp_dropout=args.dropout, shp_pooling=args.pooling,
        # generic encoders
        enable_generic_raw=bool(args.gen_raw), enable_generic_diff=bool(args.gen_diff),
        gen_d_model=args.gen_d_model, gen_heads=args.gen_heads,
        gen_layers=args.gen_layers, gen_dropout=args.gen_dropout,
    ).to(device)

    # 5) DataLoaders (fusion)
    tr_loader = make_loader_fusion(
        Z=Ztr, mask=mtr, y=y_tr, batch_size=args.batch_size, shuffle=True,
        X_raw_std=bundle.Xtr_raw_std, X_diff_std=bundle.Xtr_diff_std,
    )
    if args.use_val:
        va_loader = make_loader_fusion(
            Z=Zva, mask=mva, y=y_val, batch_size=args.batch_size, shuffle=False,
            X_raw_std=bundle.Xva_raw_std, X_diff_std=bundle.Xva_diff_std,
        )
    te_loader = make_loader_fusion(
        Z=Zte, mask=mte, y=y_test, batch_size=args.batch_size, shuffle=False,
        X_raw_std=bundle.Xte_raw_std, X_diff_std=bundle.Xte_diff_std,
    )

    # 6) Optimizer (RAdam by default)
    optimizer = make_optimizer(
        model.parameters(),
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # 7) Train
    best_test_acc = -1.0
    for ep in range(1, args.epochs + 1):
        t1 = time.time()
        tr_loss = train_one_epoch(
            model, tr_loader, optimizer, device,
            enable_generic_raw=bool(args.gen_raw), enable_generic_diff=bool(args.gen_diff)
        )
        took = time.time() - t1

        if args.use_val:
            val_loss, val_acc = evaluate(
                model, va_loader, device, return_loss=True,
                enable_generic_raw=bool(args.gen_raw), enable_generic_diff=bool(args.gen_diff)
            )
            print(f"[epoch {ep:03d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.4f} | {took:.2f}s")
        else:
            te_acc = evaluate(
                model, te_loader, device,
                enable_generic_raw=bool(args.gen_raw), enable_generic_diff=bool(args.gen_diff)
            )
            best_test_acc = max(best_test_acc, te_acc)
            print(f"[epoch {ep:03d}] train_loss={tr_loss:.4f} | test_acc={te_acc:.4f} "
                  f"(best={best_test_acc:.4f}) | {took:.2f}s")

    # 8) Final evaluation
    if args.use_val:
        final_loss_last, final_acc_last = evaluate(
            model, te_loader, device, return_loss=True,
            enable_generic_raw=bool(args.gen_raw), enable_generic_diff=bool(args.gen_diff)
        )
        best_test_acc = final_acc_last
        print(f"[final] last_epoch test_loss={final_loss_last:.4f} | test_acc={final_acc_last:.4f}")
        final_test_acc = final_acc_last
    else:
        final_loss_last, final_acc_last = evaluate(
            model, te_loader, device, return_loss=True,
            enable_generic_raw=bool(args.gen_raw), enable_generic_diff=bool(args.gen_diff)
        )
        final_test_acc = final_acc_last
        if best_test_acc < 0:
            best_test_acc = final_acc_last
        print(f"[final] test_loss={final_loss_last:.4f} | test_acc={final_acc_last:.4f} "
              f"| best_test_acc={best_test_acc:.4f}")

    # 9) Append CSV
    elapsed_total = time.time() - wall0
    _append_result_csv(
        args=args, K=int(Ztr.shape[-1]),
        final_acc=float(final_test_acc),
        best_acc=float(best_test_acc),
        elapsed_s=float(elapsed_total),
    )

if __name__ == "__main__":
    main()
