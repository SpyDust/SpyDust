# Global Sensitivity Analysis (GSA) Pipeline — README

This guide explains how to use the **`gsa_pipeline.py`** script to perform **global sensitivity analysis** on a dataset of inputs (`X`) and outputs (`Y`), and how to interpret the output table (e.g., the one you shared). It also covers how partial dependence plots are produced and how to tune the Gaussian Process (GP) surrogate for more reliable Sobol’ indices.

---

## 1) What the pipeline does

Given:
- **Parameters** `X` with shape `(N_samples, N_params)`
- **Targets** `Y` with shape `(N_samples, m)` (e.g., `m=2` for `[peak_frequency, width]`)

The pipeline computes, **for each target column**:
1. **Model-free dependence metrics** (do not require a model):
   - **MI** — Mutual Information
   - **dCor** — Distance Correlation
2. **Model-based importance** using a **Random Forest**:
   - **Permutation Importance**: `PermMean`, `PermStd`
   - **Partial Dependence Plots (PDPs)** for the **top-k drivers**
3. **Gaussian Process (GP) surrogate** (with ARD kernel) and **Sobol’ indices**:
   - GP **ARD length-scales** (`ARD_LS`): smaller ⇒ more sensitive
   - **Sobol’ First-order** `S1` and **Total-order** `ST` indices (with 95% CIs)

The script also **drops constant columns** (e.g., `xh = 0`) before analysis.

---

## 2) Installation

```bash
pip install numpy pandas scikit-learn SALib dcor matplotlib
```

> Optional (for advanced GP tuning): `gpytorch` or `gpflow` — *not required* by this pipeline.

---

## 3) Quick start

1. Save the script as `gsa_pipeline.py` (use the version I provided).
2. In your Python session / notebook:

```python
from gsa_pipeline import gsa_pipeline

# Your arrays:
# X: (N_samples, N_params)
# Y: (N_samples, 2)  # [peak_frequency, width]

param_names = ["nh","T","Chi","xh","xC","y","a","beta"]  # adjust to your order

out = gsa_pipeline(
    X, Y,
    feature_names=["nu_peak", "width"],
    param_names=param_names,
    N_sobol=4096,          # Saltelli base size (increase for tighter CIs)
    random_state=0,
    drop_const_atol=0.0,   # treat exact-constant columns as constant
    make_pdp=True,
    topk_pdp=3,
    pdp_prefix="pdp"       # saves files like pdp_nu_peak_nh.png
)

table_peak  = out["results"]["nu_peak"]["table"]
table_width = out["results"]["width"]["table"]

print(table_peak.round(4))
print(table_width.round(4))
```

3. The PDP images (top-k drivers per target) will be saved under names like:
   - `pdp_nu_peak_<param>.png`
   - `pdp_width_<param>.png`

---

## 4) Understanding the output table

Each row corresponds to a parameter; each column is a metric:

| Column     | Meaning |
|------------|---------|
| **MI**     | **Mutual Information** between parameter and target. Measures any dependency (linear or nonlinear). Higher ⇒ stronger dependence. Scale is non-negative and unitless. |
| **dCor**   | **Distance Correlation** (0–1). Zero ⇔ independent. Captures nonlinear and non-monotonic relations. |
| **PermMean** | **Permutation importance** mean (from Random Forest). Drop in test score when this feature is permuted. Larger magnitude ⇒ more important. Sign may vary depending on score definition, but magnitude carries importance. |
| **PermStd**  | Standard deviation (stability) of permutation importance across repeats. |
| **ARD_LS** | **GP ARD length-scale** per parameter (smaller ⇒ function varies faster along that axis ⇒ more important). If `NaN`, ARD wasn’t available (e.g., degenerate fit). |
| **1/ARD_LS** | Convenience inverse of ARD_LS so “bigger ⇒ stronger” (undefined if ARD_LS is NaN). |
| **S1**     | **Sobol’ first-order index**: fraction of variance explained by the parameter **alone** (holding others random). |
| **S1_conf** | 95% confidence interval half-width for `S1`. |
| **ST**     | **Sobol’ total-order index**: variance from the parameter **and all its interactions**. |
| **ST_conf** | 95% CI half-width for `ST`. |
| **AggRank** | Average rank across MI, dCor, PermMean, 1/ARD_LS, and ST. Lower ⇒ more consistently “important” across methods. |

### Interpreting typical patterns
- **High ST, high S1**: strong **main effect** (parameter matters on its own).
- **High ST, low S1**: mostly **interactions** (parameter matters via coupling).
- **ST ≈ 0 (CI includes 0)** and near-zero MI/Permutation: candidate to **fix** (keep constant).
- **ARD_LS small** and high ST: GP agrees that the function changes rapidly along this axis.

> **Caveat:** Sobol’ assumes **independent inputs** across specified bounds. In this pipeline, Sobol’ is computed on a **GP surrogate in [0,1]^d** after dropping constants; this is a pragmatic approximation. If your true inputs are correlated or constrained, trust **MI / dCor / Permutation** and PDPs more, or transform to an independent latent space first (e.g., via copulas).

---

## 5) Partial Dependence Plots (PDPs)

For the top-k drivers (by aggregate rank), we plot **1D PDPs** using the Random Forest:
- A PDP shows the **average effect** on the predicted target as one parameter varies over its range, marginalizing over other parameters.
- **Flat PDP** ⇒ weak marginal effect; **steep/curved PDP** ⇒ strong, possibly nonlinear effect.

Tips:
- For **interactions**, inspect pairwise PDPs (extend the code by passing `features=[(i,j)]` to `PartialDependenceDisplay`).
- PDPs assume the data distribution is representative; out-of-support regions can be misleading.

---

## 6) GP surrogate tuning tips

If `ARD_LS` is `NaN`, or Sobol’ CIs are large, or RF and GP disagree strongly, try:
- **Scale inputs to [0,1]** (the script already does this internally for GP).
- Increase **`N_sobol`** (e.g., 8192 or 16384) for tighter Sobol’ CIs (cost is on the GP, not the simulator).
- Increase GP optimizer restarts:  
  ```python
  GaussianProcessRegressor(..., n_restarts_optimizer=5)
  ```
- Try a **Matern** kernel (often more robust for rough functions):  
  ```python
  from sklearn.gaussian_process.kernels import Matern
  kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d), nu=1.5,
                                         length_scale_bounds=(1e-2, 1e2)) \
           + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))
  ```
- If the target is **noisy**, keep the **WhiteKernel** and consider larger noise bounds.
- If GP fit is still poor (check validation R²), prefer **tree/boosting models** (RF/XGBoost) for ranking and PDPs.

---

## 7) Dropping constant columns

Any parameter with zero range (e.g., `xh=0` for all samples) is automatically **dropped** before analysis. The returned table and PDPs only refer to the **kept** parameters. The indices of kept/dropped columns are stored in the `extras` dict for each feature (`out["results"][feat]["models"]`).

---

## 8) Dealing with correlated or constrained inputs

- **Sobol’ indices** assume independent inputs. If your parameters are correlated (e.g., physical constraints), Sobol’ values become distribution-dependent.
- Prefer **MI / dCor / Permutation** and **PDPs** in that case; or transform to an independent space (e.g., copula; Dirichlet for simplex constraints).

---

## 9) Multi-output targets

The pipeline runs the full analysis per target column in `Y`. Sensitivities can differ across targets (e.g., `nu_peak` vs `width`). Inspect both tables.

---

## 10) Troubleshooting

- **Huge or negative `PermMean` values:** permutation importance measures drop in a scoring metric; magnitude carries importance. Negative means the permutation occasionally improved the score due to noise/variance; use RF `oob_score` or cross-validation for robustness.
- **`ARD_LS = NaN`:** ARD extraction failed (e.g., kernel component name mismatch); GP may still predict fine. Try the Matern kernel and/or `n_restarts_optimizer`.
- **Sobol CIs ~ 1.0:** increase `N_sobol`, check GP fit quality, or simplify the target (e.g., log-transform).
- **PDP looks noisy:** increase RF trees (`n_estimators`), or smooth PDP curves by increasing resolution.

---

## 11) Reproducibility

Set `random_state` in the pipeline call; it controls train/test split, RF, permutation, and GP seed.

```python
out = gsa_pipeline(X, Y, random_state=0, ...)
```

---

## 12) Outputs & where to find them

- **Tables**: available in-memory, e.g. `out["results"]["nu_peak"]["table"]` (a `pandas.DataFrame`). Save to CSV with `table.to_csv("gsa_nu_peak.csv")` if desired.
- **PDPs**: saved as PNG files when `pdp_prefix` is provided, e.g., `pdp_nu_peak_<param>.png`.

---

## 13) License & citation

This pipeline is a thin integration of widely used libraries: `scikit-learn`, `SALib`, and `dcor`. Cite those projects if you use them in publications. This script itself can be used freely in your research codebase.

---

**Questions or tweaks?**  
Common extensions include pairwise PDPs, SHAP values for tree models, CSV/Parquet export, and GP alternatives (sparse GP, GPyTorch). Happy to add those if you need them.
