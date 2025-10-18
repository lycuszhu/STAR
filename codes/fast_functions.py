from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import numpy as np

from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# NOTE: adjust these imports to your project layout if needed.
from .fast_utils import (
    # Step-1 kernels
    build_kernel_features_step1,
    build_parent_argmax_map,
    # Step-2 harvesting
    harvest_shapelets,
    # feature emission for shapelets (RAW channel only)
    _collect_features_batch,
    # helpers
    ensure_float32_arrays,
    # types (for meta/debug)
    ParentKey, ParentInfo, ShapeletSpec,
)


# ---------------------------
# Helper container for heads
# ---------------------------

@dataclass
class _Head:
    scaler: Optional[StandardScaler] = None
    clf: Optional[object] = None
    Xtr: Optional[np.ndarray] = None
    Xte: Optional[np.ndarray] = None
    names: Optional[List[str]] = None


# ---------------------------
# FAST wrapper used by STAR
# ---------------------------

class fast:
    """
    FAST (STAR-aligned)

    Stage-1: MultiROCKET-like kernel features (K=9; optional K=6/K=3), pooled stats on
             RAW + DIFF channels (budget split internally).
    Stage-2: Harvest feature-map-induced shapelets; pooled stats on RAW channel only.
    Heads:   kernels / shapelets / both (linear models).
    """

    def __init__(
        self,
        *,
        include_max_feature: bool = True,          # add 5th stat 'max' (applies to kernels and shapelets)
        additional_kernel_families: bool = True,    # include K=6/K=3 in addition to K=9
        kernel_budget_total: Optional[int] = None,  # columns; None => 50k (K9) or 60k (K369); +25% if include_max_feature=True
        shapelet_budget_total: Optional[int] = None,# columns; None => 0.5 * kernel budget
        # Step-2 early-abandoning gate
        delta_eta2: float = 0.02,
        sigma_optimism: float = 0.25,
        shapelet_bias_mode: Literal["recalibrate_small", "reuse"] = "reuse",
        shapelet_subsample_frac: float = 0.20,
        shapelet_subsample_min: int = 10,
        shapelet_subsample_max: int = 256,
        eb_bias_source: Literal["single_series", "subseries"] = "single_series",
        keep_best_dilation_only: bool = True,       # retained internal speed gate in harvester
        # LDS
        lds_offset: float = 0.0,
    ):
        self.include_max_feature = bool(include_max_feature)
        self.additional_kernel_families = bool(additional_kernel_families)

        # Kernel feature budget (columns) — base=50k; +20% for K369; +25% if include_max_feature
        if kernel_budget_total is None:
            base = 50_000
            if self.additional_kernel_families:
                base = int(round(base * 1.20))   # 60,000 for K369
            if self.include_max_feature:
                base = int(round(base * 1.25))   # +25% if also add 'max'
            self.kernel_budget_total = base
        else:
            self.kernel_budget_total = int(kernel_budget_total)

        # Shapelet feature budget (columns) — default: half of kernel budget
        if shapelet_budget_total is None:
            self.shapelet_budget_total = self.kernel_budget_total // 2
        else:
            self.shapelet_budget_total = int(shapelet_budget_total)

        # Vetting / EB
        self.delta_eta2 = float(delta_eta2)
        self.sigma_optimism = float(sigma_optimism)
        self.shapelet_bias_mode = shapelet_bias_mode
        self.shapelet_subsample_frac = float(shapelet_subsample_frac)
        self.shapelet_subsample_min = int(shapelet_subsample_min)
        self.shapelet_subsample_max = int(shapelet_subsample_max)
        self.eb_bias_source = eb_bias_source
        self.keep_best_dilation_only = bool(keep_best_dilation_only)

        self.lds_offset = float(lds_offset)

        # caches
        self._Xtr_kernels: Optional[np.ndarray] = None
        self._Xte_kernels: Optional[np.ndarray] = None
        self._Xtr_shapelets: Optional[np.ndarray] = None
        self._Xte_shapelets: Optional[np.ndarray] = None
        self._Xtr_both: Optional[np.ndarray] = None
        self._Xte_both: Optional[np.ndarray] = None

        self.kernel_feat_names_: List[str] = []
        self.shapelet_feat_names_: List[str] = []
        self.both_feat_names_: List[str] = []

        # meta from step-1
        self.parent_info_: Dict[ParentKey, ParentInfo] = {}
        self.ranked_parents_: List[ParentKey] = []
        self.parent_best_eta2_: Dict[ParentKey, float] = {}

        # kept shapelets
        self.kept_shapelets_: List[ShapeletSpec] = []

        # heads
        self.head_kernels = _Head()
        self.head_shapelets = _Head()
        self.head_both = _Head()

        # y cache
        self._y_seen_: Optional[np.ndarray] = None

    # ---------------------------
    # Feature assembly
    # ---------------------------
    def assemble_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        features_used: Literal["kernels", "shapelets", "both"] = "both",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build requested feature matrices. Always computes kernels; shapelets only if requested
        and shapelet_budget_total > 0.
        """
        Xtr32 = ensure_float32_arrays([X_train])[0]
        Xte32 = ensure_float32_arrays([X_test])[0]
        self._y_seen_ = np.asarray(y_train, dtype=np.int64)

        # ---- Step-1: kernel features (RAW + DIFF inside)
        (Xtr_k, Xte_k, kernel_names,
         parent_info, ranked_parents, parent_best_eta2, feature_registry) = build_kernel_features_step1(
            X_train=Xtr32,
            y_train=y_train,
            X_test=Xte32,
            additional_kernel_families=self.additional_kernel_families,
            kernel_budget_total=self.kernel_budget_total,
            include_max_feature=self.include_max_feature,
            subsample_frac_for_thresholds=0.0,  # single-series thresholds internally
            subsample_min=0,
            subsample_max=0,
            lds_offset=self.lds_offset,
        )

        self._Xtr_kernels = Xtr_k.astype(np.float32, copy=False)
        self._Xte_kernels = Xte_k.astype(np.float32, copy=False)
        self.kernel_feat_names_ = kernel_names

        self.parent_info_ = parent_info
        self.ranked_parents_ = ranked_parents
        self.parent_best_eta2_ = parent_best_eta2

        # Prepare empty shapelet matrices by default
        self._Xtr_shapelets = np.zeros((X_train.shape[0], 0), dtype=np.float32)
        self._Xte_shapelets = np.zeros((X_test.shape[0], 0), dtype=np.float32)
        self.shapelet_feat_names_ = []
        self.kept_shapelets_ = []

        # ---- Step-1.5: build argmax map once (for Stage-2 proposals)
        argmax_map = build_parent_argmax_map(Xtr32, self.ranked_parents_)

        # ---- Step-2: harvesting (only if budget > 0 and features_used asks for it)
        want_shapelets = (features_used in ("shapelets", "both")) and (self.shapelet_budget_total > 0)
        if want_shapelets:
            kept = harvest_shapelets(
                X_train=Xtr32,
                y_train=y_train,
                ranked_parents=self.ranked_parents_,
                parent_info=self.parent_info_,
                parent_best_eta2=self.parent_best_eta2_,
                include_max_feature=self.include_max_feature,
                additional_kernel_families=self.additional_kernel_families,
                shapelet_budget_total=self.shapelet_budget_total,
                delta_eta2=self.delta_eta2,
                sigma_optimism=self.sigma_optimism,
                shapelet_bias_mode=self.shapelet_bias_mode,   # default "reuse"
                subsample_frac=self.shapelet_subsample_frac,
                subsample_min=self.shapelet_subsample_min,
                subsample_max=self.shapelet_subsample_max,
                lds_offset=self.lds_offset,
                eb_bias_source=self.eb_bias_source,
                keep_best_dilation_only=self.keep_best_dilation_only,
                argmax_map=argmax_map,
            )
            self.kept_shapelets_ = kept

            # Emit features for kept shapelets (RAW channel only)
            blocks_tr: List[np.ndarray] = []
            blocks_te: List[np.ndarray] = []
            for shp in kept:
                Xb_tr = _collect_features_batch(Xtr32, shp.values, shp.thr_base, self.include_max_feature)
                Xb_te = _collect_features_batch(Xte32, shp.values, shp.thr_base, self.include_max_feature)
                blocks_tr.append(Xb_tr)
                blocks_te.append(Xb_te)

            if blocks_tr:
                self._Xtr_shapelets = np.hstack(blocks_tr).astype(np.float32, copy=False)
                self._Xte_shapelets = np.hstack(blocks_te).astype(np.float32, copy=False)
            else:
                self._Xtr_shapelets = np.zeros((X_train.shape[0], 0), dtype=np.float32)
                self._Xte_shapelets = np.zeros((X_test.shape[0], 0), dtype=np.float32)

        # ---- Combine heads
        self._Xtr_both = np.hstack(
            [a for a in (self._Xtr_kernels, self._Xtr_shapelets) if a is not None and a.shape[1] > 0]
        ).astype(np.float32, copy=False)
        self._Xte_both = np.hstack(
            [a for a in (self._Xte_kernels, self._Xte_shapelets) if a is not None and a.shape[1] > 0]
        ).astype(np.float32, copy=False)
        if self._Xtr_both.shape[0] != X_train.shape[0] or self._Xte_both.shape[0] != X_test.shape[0]:
            raise RuntimeError("Assembled feature shapes do not align with train/test sizes.")

        # names (unused)
        self.both_feat_names_ = []

        if features_used == "kernels":
            return self._Xtr_kernels, self._Xte_kernels, self.kernel_feat_names_
        elif features_used == "shapelets":
            return self._Xtr_shapelets, self._Xte_shapelets, self.shapelet_feat_names_
        else:
            return self._Xtr_both, self._Xte_both, self.both_feat_names_

    # ---------------------------
    # Fit & predict
    # ---------------------------
    def fit_classifier(
        self,
        *,
        clf_type: Literal["ridge", "logreg"] = "ridge",
        alpha_grid: Optional[np.ndarray] = None,
        max_iter_logreg: int = 200,
        C_logreg: float = 1.0,
    ):
        """
        Fit three heads: kernels / shapelets / both.
          - ridge: RidgeClassifierCV with logspace grid 1e-1..1e3 (9 pts) if not provided
          - logreg: lbfgs, multi_class=auto, C=1.0
        """
        if alpha_grid is None:
            alpha_grid = np.logspace(-1, 3, 9, base=10.0)

        def _mk_clf():
            if clf_type == "ridge":
                return RidgeClassifierCV(alphas=alpha_grid)
            else:
                return LogisticRegression(
                    penalty="l2", C=C_logreg, solver="lbfgs", max_iter=max_iter_logreg, n_jobs=None
                )

        # kernels
        if self._Xtr_kernels is not None and self._Xtr_kernels.shape[1] > 0:
            pipe = make_pipeline(StandardScaler(with_mean=False), _mk_clf())
            pipe.fit(self._Xtr_kernels, self._y_train_placeholder(y_required=True))
            self.head_kernels = _Head(
                pipe.named_steps["standardscaler"],
                pipe.named_steps[list(pipe.named_steps.keys())[-1]],
                self._Xtr_kernels,
                self._Xte_kernels,
                self.kernel_feat_names_,
            )
        else:
            self.head_kernels = _Head()

        # shapelets
        if self._Xtr_shapelets is not None and self._Xtr_shapelets.shape[1] > 0:
            pipe = make_pipeline(StandardScaler(with_mean=False), _mk_clf())
            pipe.fit(self._Xtr_shapelets, self._y_train_placeholder(y_required=True))
            self.head_shapelets = _Head(
                pipe.named_steps["standardscaler"],
                pipe.named_steps[list(pipe.named_steps.keys())[-1]],
                self._Xtr_shapelets,
                self._Xte_shapelets,
                self.shapelet_feat_names_,
            )
        else:
            self.head_shapelets = _Head()

        # both
        if self._Xtr_both is not None and self._Xtr_both.shape[1] > 0:
            pipe = make_pipeline(StandardScaler(with_mean=False), _mk_clf())
            pipe.fit(self._Xtr_both, self._y_train_placeholder(y_required=True))
            self.head_both = _Head(
                pipe.named_steps["standardscaler"],
                pipe.named_steps[list(pipe.named_steps.keys())[-1]],
                self._Xtr_both,
                self._Xte_both,
                self.both_feat_names_,
            )
        else:
            self.head_both = _Head()

    def predict(self, which: Literal["kernels", "shapelets", "both"] = "both") -> np.ndarray:
        if which == "kernels":
            head = self.head_kernels
        elif which == "shapelets":
            head = self.head_shapelets
        else:
            head = self.head_both

        if head.clf is None or head.Xte is None or head.scaler is None:
            raise RuntimeError(f"No model for '{which}'. Did you call fit_classifier() after assembling features?")

        Xte_scaled = head.scaler.transform(head.Xte)
        return head.clf.predict(Xte_scaled)

    # -------------
    # internals
    # -------------
    def _y_train_placeholder(self, y_required: bool = False) -> np.ndarray:
        if y_required and self._y_seen_ is None:
            raise RuntimeError("assemble_features must be called before fit_classifier (y_train not cached).")
        return self._y_seen_

    def set_y_for_fit(self, y_train: np.ndarray):
        self._y_seen_ = np.asarray(y_train, dtype=np.int64)
