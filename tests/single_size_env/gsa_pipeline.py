# gsa_pipeline.py
# Global Sensitivity Analysis pipeline with robust scaling, kernel builder,
# optional heavy metrics (Permutation, Sobol, PDP), and constant-column handling.

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

# Screening
from sklearn.feature_selection import mutual_info_regression
import dcor  # pip install dcor

# Models & utilities
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic,
    WhiteKernel, ConstantKernel as C
)

# Variance-based GSA
from SALib.sample import saltelli
from SALib.analyze import sobol

# Plotting
import matplotlib.pyplot as plt


# ============================== Utilities ===================================

def _default_param_names(d: int) -> List[str]:
    return [f"p{i}" for i in range(d)]


def _infer_bounds_from_data(
    X: np.ndarray,
    pad_frac: float = 0.02
) -> List[List[float]]:
    """Infer per-parameter [low, high] from data with a small padding."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span = np.where(span == 0.0, 1.0, span)  # avoid zero span
    lo2 = lo - pad_frac * span
    hi2 = hi + pad_frac * span
    return [[float(a), float(b)] for a, b in zip(lo2, hi2)]


def _drop_constant_columns(
    X: np.ndarray,
    param_names: List[str],
    atol: float = 0.0
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    """Drop columns with (max - min) ~= 0 (within atol)."""
    const_mask = np.isclose(X.max(axis=0) - X.min(axis=0), 0.0, atol=atol)
    dropped = np.where(const_mask)[0].tolist()
    kept = np.where(~const_mask)[0].tolist()
    Xr = X[:, kept] if len(kept) > 0 else X[:, :0]
    names_r = [param_names[i] for i in kept]
    return Xr, names_r, kept, dropped


def _make_scaler(
    scaler: Optional[str]
):
    """Return an initialized scaler object or None."""
    if scaler is None:
        return None
    scaler = scaler.lower()
    if scaler in ("minmax", "min_max", "min-max"):
        return MinMaxScaler(feature_range=(0, 1))
    if scaler in ("standard", "z", "zscore", "z-score"):
        return StandardScaler(with_mean=True, with_std=True)
    raise ValueError(f"Unknown scaler='{scaler}'. Use 'minmax', 'standard', or None.")


# ============================== Kernel builder ===============================

def build_kernel(
    kind: str = "rbf",
    ard: bool = True,
    d: int = 1,
    length_scale_init: float | np.ndarray = 1.0,
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e2),
    alpha_rq: float = 1.0,
    nu_matern: float = 1.5,
    constant_bounds: Tuple[float, float] = (1e-3, 1e3),
    noise_bounds: Tuple[float, float] = (1e-9, 1e-2),
    noise_init: float = 1e-6,
    constant_init: float = 1.0,
) -> Any:
    """
    Construct a GP kernel with ARD support where possible.

    kind: 'rbf' | 'matern32' | 'matern52' | 'rq'
    ard:  if True, use vector length-scales (per-dimension)
    d:    input dimensionality
    """
    # length_scale_init can be scalar or vector; make vector for ARD
    if ard:
        if np.isscalar(length_scale_init):
            ls = np.ones(d) * float(length_scale_init)
        else:
            ls = np.asarray(length_scale_init, dtype=float)
            if ls.shape != (d,):
                raise ValueError(f"length_scale_init must have shape ({d},) for ARD=True.")
    else:
        ls = float(length_scale_init)

    if kind.lower() in ("rbf", "sqexp", "squared_exponential"):
        base = RBF(length_scale=ls, length_scale_bounds=length_scale_bounds)
    elif kind.lower() in ("matern32", "matern_32", "matern-32"):
        base = Matern(length_scale=ls, length_scale_bounds=length_scale_bounds, nu=1.5)
    elif kind.lower() in ("matern52", "matern_52", "matern-52"):
        base = Matern(length_scale=ls, length_scale_bounds=length_scale_bounds, nu=2.5)
    elif kind.lower() in ("rq", "rationalquadratic", "rational_quadratic"):
        # sklearn's RationalQuadratic kernel only accepts scalar length-scales.
        # When ARD was requested we quietly fall back to a scalar using the
        # geometric mean so that optimisation still receives a sensible scale.
        if ard:
            ls_scalar = float(np.exp(np.mean(np.log(ls))))
        else:
            ls_scalar = float(ls)
        base = RationalQuadratic(length_scale=ls_scalar, alpha=alpha_rq,
                                 length_scale_bounds=length_scale_bounds)
    else:
        raise ValueError(f"Unknown kernel kind='{kind}'.")

    kernel = (
        C(constant_init, constant_bounds) * base
        + WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)
    )
    return kernel


def _extract_lengthscales(kernel, d: int) -> np.ndarray:
    """Traverse a fitted kernel to recover per-dimension length-scales."""

    ls_collection: List[np.ndarray] = []

    def visit(k):
        if hasattr(k, "k1") and hasattr(k, "k2"):
            visit(k.k1)
            visit(k.k2)
        if hasattr(k, "length_scale"):
            try:
                raw = np.asarray(k.length_scale, dtype=float)
            except Exception:
                return
            arr = np.atleast_1d(raw)
            if arr.shape == (d,):
                ls_collection.append(arr)
            elif arr.size == 1:
                ls_collection.append(np.repeat(float(arr.item()), d))

    visit(kernel)

    if not ls_collection:
        return np.full(d, np.nan)

    ls_stack = np.vstack(ls_collection)
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = ls_stack.shape[0] / np.nansum(1.0 / ls_stack, axis=0)
    return harmonic


# ============================== Metrics ======================================

def _screening_metrics(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
    enable_perm: bool = True
) -> Dict[str, Any]:
    """
    Compute MI, dCor and (optionally) permutation importance with a RF.
    If enable_perm=False, we still fit a light RF (for PDPs if later requested).
    """
    mi = mutual_info_regression(X, y, random_state=random_state)
    dcor_vals = np.array([dcor.distance_correlation(X[:, i], y) for i in range(X.shape[1])])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # RF is handy for PDPs and a robust baseline; keep n_estimators moderate
    rf = RandomForestRegressor(
        n_estimators=600 if enable_perm else 300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    ).fit(Xtr, ytr)

    if enable_perm:
        perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=random_state, n_jobs=-1)
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std
    else:
        perm_mean = np.zeros(X.shape[1])
        perm_std = np.zeros(X.shape[1])

    return {
        "MI": mi,
        "dCor": dcor_vals,
        "PermMean": perm_mean,
        "PermStd": perm_std,
        "rf_model": rf,
        "Xte": Xte, "yte": yte
    }


def _gp_surrogate_and_sobol(
    X: np.ndarray,
    y: np.ndarray,
    bounds_orig: List[List[float]],
    kernel_kind: str = "rbf",
    ard: bool = True,
    scaler: Optional[str] = "minmax",
    gp_random_state: int = 0,
    N_sobol: int = 4096,
    enable_sobol: bool = True,
    length_scale_init: float | np.ndarray = 1.0,
) -> Tuple[Optional[GaussianProcessRegressor], Optional[Dict[str, Any]], np.ndarray]:
    """
    Fit a GP on scaled X->y and (optionally) compute Sobol indices on the GP surrogate.

    IMPORTANT: Sobol sampling is done in ORIGINAL space using bounds_orig.
    The Saltelli sample Z_orig is then mapped through the scaler -> GP space
    before gp.predict. This fixes the StandardScaler mismatch issue.
    """
    d = X.shape[1]

    # Build scaler and scale X for GP
    scaler_obj = _make_scaler(scaler)
    if scaler_obj is not None:
        X_scaled = scaler_obj.fit_transform(X)
    else:
        X_scaled = X.copy()

    # Build kernel with ARD
    kernel = build_kernel(
        kind=kernel_kind, ard=ard, d=d,
        length_scale_init=length_scale_init
    )

    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=gp_random_state)
    gp.fit(X_scaled, y)

    # Extract ARD length-scales (smaller => more sensitive)
    ard_ls = _extract_lengthscales(gp.kernel_, d)

    # Optionally compute Sobol on the GP surrogate
    if enable_sobol:
        problem = {
            "num_vars": d,
            "names": [f"p{i}" for i in range(d)],
            "bounds": bounds_orig  # ORIGINAL space bounds
        }
        # 1) Sample in original space
        Z_orig = saltelli.sample(problem, N_sobol, calc_second_order=False)
        # 2) Transform into the scaled GP space
        if scaler_obj is not None:
            Z_scaled = scaler_obj.transform(Z_orig)
        else:
            Z_scaled = Z_orig
        # 3) Predict with GP
        y_hat = gp.predict(Z_scaled)
        # 4) Sobol indices
        Si = sobol.analyze(
            problem, y_hat, calc_second_order=False, conf_level=0.95,
            print_to_console=False, seed=gp_random_state
        )
    else:
        Si = None

    return gp, Si, ard_ls


def _aggregate_table(
    param_names: List[str],
    MI: np.ndarray, dCor: np.ndarray,
    PermMean: np.ndarray, PermStd: np.ndarray,
    ARD_LS: np.ndarray,
    Si: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Build the consolidated table. If Sobol is disabled (Si=None), S1/ST columns are zeros.
    """
    d = len(param_names)
    if Si is None:
        S1 = np.zeros(d); S1c = np.zeros(d)
        ST = np.zeros(d); STc = np.zeros(d)
    else:
        S1 = np.asarray(Si["S1"])
        S1c = np.asarray(Si["S1_conf"])
        ST = np.asarray(Si["ST"])
        STc = np.asarray(Si["ST_conf"])

    tbl = pd.DataFrame(
        {
            "MI": MI,
            "dCor": dCor,
            "PermMean": PermMean,
            "PermStd": PermStd,
            "ARD_LS": ARD_LS,
            "S1": S1, "S1_conf": S1c,
            "ST": ST, "ST_conf": STc,
        },
        index=param_names
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ls = 1.0 / tbl["ARD_LS"].to_numpy()
    tbl["1/ARD_LS"] = inv_ls

    # Aggregate ranking across available metrics; Sobol may be zeroed if disabled
    ranks = pd.DataFrame({
        "r_MI": tbl["MI"].rank(ascending=False),
        "r_dCor": tbl["dCor"].rank(ascending=False),
        "r_Perm": tbl["PermMean"].rank(ascending=False),
        "r_InvLS": tbl["1/ARD_LS"].rank(ascending=False),
        "r_ST": tbl["ST"].rank(ascending=False),
    }, index=tbl.index)
    tbl["AggRank"] = ranks.mean(axis=1)
    return tbl.sort_values("AggRank")


# ============================== Public API ===================================

def gsa_for_target(
    X: np.ndarray,
    y: np.ndarray,
    *,
    param_names: Optional[List[str]] = None,
    # Scaling & bounds
    scaler: Optional[str] = "minmax",          # 'minmax' | 'standard' | None
    bounds: Optional[List[List[float]]] = None, # ORIGINAL space bounds for Sobol; if None, inferred
    bounds_pad_frac: float = 0.02,
    # Kernel / GP
    kernel_kind: str = "rbf",  # 'rbf' | 'matern32' | 'matern52' | 'rq'
    ard: bool = True,
    length_scale_init: float | np.ndarray = 1.0,
    gp_random_state: int = 0,
    # Toggles for heavy steps
    enable_perm: bool = True,
    enable_gp: bool = True,
    enable_sobol: bool = True,
    # Sobol budget
    N_sobol: int = 4096,
    # Preprocessing
    drop_const_atol: float = 0.0,
    # PDPs
    make_pdp: bool = True,
    topk_pdp: int = 3,
    pdp_fig_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run GSA for a single scalar target y.

    Switch off heavy metrics with:
      enable_perm=False (skip permutation importance),
      enable_gp=False   (skip GP & ARD & Sobol entirely),
      enable_sobol=False (keep GP + ARD but skip Sobol).

    Scaling note:
      Sobol sampling ALWAYS occurs in ORIGINAL space and gets transformed
      into the GP scaled space before prediction. This fixes the
      StandardScaler mismatch bug.
    """
    n, d = X.shape
    if param_names is None:
        param_names = _default_param_names(d)

    # 1) Drop constant columns
    Xr, names_r, kept_idx, dropped_idx = _drop_constant_columns(X, param_names, atol=drop_const_atol)
    if Xr.shape[1] == 0:
        raise ValueError("All parameters are constant; nothing to analyze.")

    # 2) Screening (MI, dCor, optional Permutation)
    scr = _screening_metrics(Xr, y, random_state=gp_random_state, enable_perm=enable_perm)

    # 3) GP + Sobol (optional)
    if enable_gp:
        if bounds is None:
            bounds_use = _infer_bounds_from_data(Xr, pad_frac=bounds_pad_frac)
        else:
            if len(bounds) != len(param_names):
                raise ValueError(
                    "Length of 'bounds' must match number of input parameters before dropping constants."
                )
            bounds_use = [bounds[i] for i in kept_idx]
        gp, Si, ard_ls = _gp_surrogate_and_sobol(
            X=Xr, y=y, bounds_orig=bounds_use,
            kernel_kind=kernel_kind, ard=ard, scaler=scaler,
            gp_random_state=gp_random_state, N_sobol=N_sobol,
            enable_sobol=enable_sobol, length_scale_init=length_scale_init
        )
    else:
        gp, Si, ard_ls = None, None, np.full(Xr.shape[1], np.nan)

    # 4) Table
    table = _aggregate_table(
        param_names=names_r,
        MI=scr["MI"], dCor=scr["dCor"],
        PermMean=scr["PermMean"], PermStd=scr["PermStd"],
        ARD_LS=ard_ls, Si=Si
    )

    # 5) PDPs for top-k using the RF (if requested)
    pdp_paths = []
    if make_pdp:
        k = min(topk_pdp, Xr.shape[1])
        top_params = table.index[:k].tolist()
        name_to_idx = {nm: i for i, nm in enumerate(names_r)}
        feats_idx = [name_to_idx[nm] for nm in top_params]
        for nm, fi in zip(top_params, feats_idx):
            fig, ax = plt.subplots(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(
                scr["rf_model"], Xr, features=[fi], ax=ax, kind="average"
            )
            ax.set_title(f"PDP – {nm}")
            fig.tight_layout()
            if pdp_fig_prefix:
                path = f"{pdp_fig_prefix}_{nm}.png"
                fig.savefig(path, dpi=150)
                pdp_paths.append(path)
            plt.close(fig)

    extras = {
        "rf_model": scr["rf_model"],
        "gp_model": gp,
        "sobol_raw": Si,
        "kept_idx": kept_idx,
        "dropped_idx": dropped_idx,
        "kept_names": names_r,
        "pdp_saved_paths": pdp_paths,
        "scaler": scaler,
        "kernel_kind": kernel_kind,
        "ard": ard,
    }
    return table, extras


def gsa_pipeline(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feature_names: Optional[List[str]] = None,
    param_names: Optional[List[str]] = None,
    # Scaling & bounds
    scaler: Optional[str] = "minmax",
    bounds: Optional[List[List[float]]] = None,
    bounds_pad_frac: float = 0.02,
    # Kernel / GP
    kernel_kind: str = "rbf",
    ard: bool = True,
    length_scale_init: float | np.ndarray = 1.0,
    gp_random_state: int = 0,
    # Toggles
    enable_perm: bool = True,
    enable_gp: bool = True,
    enable_sobol: bool = True,
    make_pdp: bool = True,
    # Budgets / misc
    N_sobol: int = 4096,
    drop_const_atol: float = 0.0,
    topk_pdp: int = 3,
    pdp_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run GSA for each column in Y.

    Time-saving toggles:
      - enable_perm=False  -> skips permutation importance (heavy)
      - enable_sobol=False -> skips Sobol (heavy)
      - make_pdp=False     -> skips PDP plotting (moderate)
      - enable_gp=False    -> skips GP/ARD (and automatically Sobol)

    scaler: 'minmax'|'standard'|None
    kernel_kind: 'rbf'|'matern32'|'matern52'|'rq'
    """
    n, d = X.shape
    m = Y.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(m)]
    if param_names is None:
        param_names = _default_param_names(d)

    if not enable_gp:
        enable_sobol = False  # Sobol needs the GP surrogate

    results = {}
    for j in range(m):
        y = Y[:, j]
        prefix = f"{pdp_prefix}_{feature_names[j]}" if pdp_prefix else None

        table, extras = gsa_for_target(
            X, y,
            param_names=param_names,
            scaler=scaler,
            bounds=bounds,
            bounds_pad_frac=bounds_pad_frac,
            kernel_kind=kernel_kind,
            ard=ard,
            length_scale_init=length_scale_init,
            gp_random_state=gp_random_state,
            enable_perm=enable_perm,
            enable_gp=enable_gp,
            enable_sobol=enable_sobol,
            N_sobol=N_sobol,
            drop_const_atol=drop_const_atol,
            make_pdp=make_pdp,
            topk_pdp=topk_pdp,
            pdp_fig_prefix=prefix
        )
        results[feature_names[j]] = {"table": table, "models": extras}

    return {
        "results": results,
        "feature_names": feature_names,
        "param_names": param_names,
        "notes": (
            "Sobol indices are computed on a GP surrogate. "
            "Sampling occurs in ORIGINAL parameter space and is transformed "
            "through the chosen scaler before GP prediction. "
            "For correlated/constrained inputs, interpret Sobol cautiously and "
            "lean on MI/dCor/Permutation + PDPs."
        ),
    }


# ============================== Example usage =================================
if __name__ == "__main__":
    # Synthetic demo
    rng = np.random.default_rng(0)
    N, d = 1000, 7
    X = rng.random((N, d))
    X[:, 3] = 0.0  # constant column to demonstrate dropping

    # Two synthetic targets
    y1 = np.sin(6 * X[:, 0]) + 0.6 * X[:, 1] ** 2 + 0.25 * X[:, 2] + 0.05 * rng.normal(size=N)
    y2 = 0.8 * X[:, 2] - 0.2 * np.cos(4 * X[:, 4]) + 0.15 * X[:, 5] ** 3 + 0.05 * rng.normal(size=N)
    Y = np.column_stack([y1, y2])

    out = gsa_pipeline(
        X, Y,
        feature_names=["nu_peak", "width"],
        param_names=["nh","T","Chi","xh","xC","y","beta"],
        scaler="standard",            # try "minmax" or None as well
        kernel_kind="rq",             # try "rbf", "matern32", "matern52", "rq"
        ard=True,
        length_scale_init=1.0,        # or np.ones(d_kept) after dropping consts (handled internally)
        enable_perm=True,
        enable_gp=True,
        enable_sobol=True,
        make_pdp=True,
        N_sobol=2048,                 # increase for tighter Sobol CIs
        drop_const_atol=0.0,
        topk_pdp=3,
        pdp_prefix="pdp_demo",
        gp_random_state=0,
    )

    for feat, obj in out["results"].items():
        print(f"\n=== GSA for {feat} ===")
        print(obj["table"].round(4))
        if obj["models"]["pdp_saved_paths"]:
            print("Saved PDPs:", obj["models"]["pdp_saved_paths"])