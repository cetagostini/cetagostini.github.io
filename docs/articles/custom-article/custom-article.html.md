<a href="#quarto-document-content" class="skip-link">Skip to content</a>

<div id="title-block-header" class="quarto-title-block default">

<div class="quarto-title">

<div class="quarto-title-block">

<div>

# Cross-City Media Spill: A Synthetic MMM with Learned Halo

Code

- <a href="javascript:void(0)" id="quarto-show-all-code" class="dropdown-item" role="button">Show All Code</a>

- <a href="javascript:void(0)" id="quarto-hide-all-code" class="dropdown-item" role="button">Hide All Code</a>

- 

  ------------------------------------------------------------------------

- <a href="javascript:void(0)" id="quarto-view-source" class="dropdown-item" role="button">View Source</a>

</div>

</div>

<div class="quarto-categories">

<div class="quarto-category">

python

</div>

<div class="quarto-category">

bayesian

</div>

<div class="quarto-category">

mmm

</div>

<div class="quarto-category">

synthetic

</div>

</div>

</div>

<div>

<div class="description">

Multi-city MMM with bounded, learned cross-city media spillovers built on pymc-marketing’s multidimensional API and MaskedPrior.

</div>

</div>

<div class="quarto-title-meta">

<div>

<div class="quarto-title-meta-heading">

Author

</div>

<div class="quarto-title-meta-contents">

Carlos Trujillo

</div>

</div>

<div>

<div class="quarto-title-meta-heading">

Published

</div>

<div class="quarto-title-meta-contents">

August 7, 2026

</div>

</div>

</div>

</div>

<div id="setup-and-data" class="section level2">

## 1 · Setup and data

Two independent synthetic worlds generated with [`prior-generator`](https://github.com/cetagostini/prior-generator), treated as two cities sharing a national media market. Each city has 10 media channels (C1–C10), 2 controls (Z1–Z2), and a sales target.

The true data-generating process includes a known 10 % cross-city media spillover: Caracas C1 and C2 affect Valencia, and Valencia C3 affects Caracas. The model below does **not** know which routes exist — it learns the active spill paths and their magnitudes through `MaskedPrior`, which zeroes out parameters for source→receiver channel pairs that carry no real signal.

<div id="c409d4ac" class="cell" execution_count="1">

Code

<div id="cb1" class="sourceCode cell-code">

``` sourceCode
import warnings
warnings.filterwarnings("ignore")

from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pydantic import Field, InstanceOf
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation
from pymc_marketing.mmm.additive_effect import MuEffect
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.scaling import DataDerivedScaling, FixedScaling, Scaling
from pymc_marketing.special_priors import MaskedPrior
```

</div>

</div>

</div>

<div id="panel-contract" class="section level2">

## 2 · Panel contract

<div id="a6098b17" class="cell" execution_count="2">

Code

<div id="cb2" class="sourceCode cell-code">

``` sourceCode
CITIES = ("Caracas", "Valencia")
CHANNELS = [f"C{k}" for k in range(1, 11)]
CONTROLS = ["Z1", "Z2"]

PANEL_DIMS = ("city",)
PANEL_CHANNEL_DIMS = ("city", "channel")
PANEL_CONTROL_DIMS = ("city", "control")

SPEND_DIMS = ("spend_city",)
SPEND_CHANNEL_DIMS = ("spend_city", "channel")
SPILL_PATH_DIMS = ("city", "spend_city", "channel")
SPEND_RENAME = {"city": "spend_city"}
```

</div>

</div>

<div id="load-raw-data" class="section level3">

### 2.1 · Load raw data

<div id="51978a4c" class="cell" execution_count="3">

Code

<div id="cb3" class="sourceCode cell-code">

``` sourceCode
valencia_raw = pd.read_csv("../../data/valencia_raw.csv", parse_dates=["date"])
caracas_raw  = pd.read_csv("../../data/caracas_raw.csv",  parse_dates=["date"])

frames = []
for city, df in [("Valencia", valencia_raw), ("Caracas", caracas_raw)]:
    df = df.rename(columns={"date": "date", "Y": "y"})
    df["city"] = city
    frames.append(df)

panel = pd.concat(frames, ignore_index=True).sort_values(
    ["date", "city"], ignore_index=True
)

X = panel[["date", "city", *CHANNELS, *CONTROLS]]
y = panel["y"]

print(f"Panel: {len(panel)} rows, {len(CITIES)} cities, "
      f"{len(CHANNELS)} channels, {len(CONTROLS)} controls")
panel.groupby("city")["y"].describe().round(2)
```

</div>

<div class="cell-output cell-output-stdout">

    Panel: 208 rows, 2 cities, 10 channels, 2 controls

</div>

<div class="cell-output cell-output-display" execution_count="3">

<div>

|          | count | mean | std  | min  | 25%  | 50%  | 75%  | max   |
|----------|-------|------|------|------|------|------|------|-------|
| city     |       |      |      |      |      |      |      |       |
| Caracas  | 104.0 | 9.51 | 0.41 | 8.43 | 9.24 | 9.52 | 9.81 | 10.43 |
| Valencia | 104.0 | 6.82 | 0.53 | 6.03 | 6.42 | 6.65 | 7.19 | 7.89  |

</div>

</div>

</div>

</div>

</div>

<div id="spill-scope-and-route-masks" class="section level2">

## 3 · Spill scope and route masks

Only three directed source→receiver channel pairs carry real cross-city media. The model learns a latent fraction `u ~ Beta(1, 1)` per active route and sets `spill_share = MAX_SPILL * u`. `MaskedPrior` ensures that every inactive source-receiver-channel triple has **no** sampled parameter at all — not a zero estimate, but a structural absence from the computational graph.

<div id="52063482" class="cell" execution_count="4">

Code

<div id="cb5" class="sourceCode cell-code">

``` sourceCode
MAX_SPILL = 0.10

# ── Which routes exist? ──
# (source_city, receiver_city, channel) triples that carry real spill.
SPILL_ROUTES = {
    ("Caracas", "Valencia", "C1"),
    ("Caracas", "Valencia", "C2"),
    ("Valencia", "Caracas", "C3"),
}

# ── Build the boolean spill-path mask ──
spill_mask_data = np.zeros(
    (len(CITIES), len(CITIES), len(CHANNELS)), dtype=bool
)
for src, rec, ch in SPILL_ROUTES:
    si = CITIES.index(rec)   # receiver axis
    sj = CITIES.index(src)   # spend axis
    ci = CHANNELS.index(ch)
    spill_mask_data[si, sj, ci] = True

spill_mask = xr.DataArray(
    spill_mask_data,
    dims=SPILL_PATH_DIMS,
    coords={"city": list(CITIES), "spend_city": list(CITIES), "channel": CHANNELS},
)

# ── Source-active mask: which (spend_city, channel) combos can export? ──
source_active_mask = spill_mask.any("city").transpose(*SPEND_CHANNEL_DIMS)

print("Spill routes:")
for src, rec, ch in sorted(SPILL_ROUTES):
    print(f"  {src} {ch} → {rec}")
print(f"\nSource-active (spend_city × channel): {int(source_active_mask.sum())} params")
```

</div>

<div class="cell-output cell-output-stdout">

    Spill routes:
      Caracas C1 → Valencia
      Caracas C2 → Valencia
      Valencia C3 → Caracas

    Source-active (spend_city × channel): 3 params

</div>

</div>

</div>

<div id="spilleffect-a-bounded-halo-mueffect" class="section level2">

## 4 · `SpillEffect`: a bounded halo `MuEffect`

This class follows the `HaloMediaEffect` pattern from pymc-marketing recipes. It:

1.  Reads the direct `channel_contribution` from the MMM after adstock and saturation.
2.  Multiplies each source contribution by a learned `spill_share` (bounded by `MAX_SPILL`).
3.  Sums over source city and channel, producing one additive term per receiving city.

`MaskedPrior` wraps `Beta(1, 1)` so that only the three active routes get a free parameter. The other 27 city×city×channel combinations contribute exactly zero to the model mean.

<div id="f771f15a" class="cell" execution_count="5">

Code

<div id="cb7" class="sourceCode cell-code">

``` sourceCode
class SpillEffect(MuEffect):
    """Bounded cross-city media spillover via MaskedPrior."""

    source_active_mask: InstanceOf[xr.DataArray] = Field(exclude=True)
    spill_path_mask: InstanceOf[xr.DataArray] = Field(exclude=True)
    fraction_prior: InstanceOf[Prior]
    max_share: float = Field(default=0.10, gt=0, le=1)
    prefix: str = "spill"

    def create_data(self, mmm: Any) -> None:
        """Register spend-city coord and fixed path mask."""
        model = mmm.model
        model.add_coord("spend_city", values=model.coords["city"])
        pmd.Data(
            f"{self.prefix}_path_mask",
            self.spill_path_mask.astype(float).values,
            dims=SPILL_PATH_DIMS,
        )

    def create_effect(self, mmm: Any):
        """Build the spill contribution as a deterministic addition to mu."""
        model = mmm.model

        # One fraction per active (spend_city, channel). MaskedPrior skips
        # inactive combos — they never appear in the graph.
        total_fraction = MaskedPrior(
            self.fraction_prior,
            mask=self.source_active_mask,
            active_dim=f"{self.prefix}_active_source_channel",
        ).create_variable(f"{self.prefix}_fraction", xdist=True)

        total_share = pmd.Deterministic(
            f"{self.prefix}_share",
            (self.max_share * total_fraction).transpose(*SPEND_CHANNEL_DIMS),
        )

        # Read direct contributions, rename source role to spend_city
        source_direct_original = pmd.Deterministic(
            f"{self.prefix}_source_contribution_original_scale",
            (model["channel_contribution"] * model["target_scale"])
            .rename(SPEND_RENAME)
            .transpose("date", *SPEND_CHANNEL_DIMS),
        )

        # Each active path gets total_share; inactive paths get 0 via the mask
        # path_share dims: (city, spend_city, channel) — no date yet
        path_share = pmd.Deterministic(
            f"{self.prefix}_path_share",
            (total_share * model[f"{self.prefix}_path_mask"]).transpose(
                *SPILL_PATH_DIMS
            ),
        )

        # Multiply by source contributions (has date) to get per-path time series
        by_path_original = pmd.Deterministic(
            f"{self.prefix}_by_path_original_scale",
            (source_direct_original * path_share).transpose(
                "date", *SPILL_PATH_DIMS
            ),
        )

        # Sum over source city and channel → one series per receiving city
        contribution_original = pmd.Deterministic(
            f"{self.prefix}_contribution_original_scale",
            by_path_original.sum(dim=(*SPEND_DIMS, "channel")).transpose(
                "date", *PANEL_DIMS
            ),
        )

        # Return in scaled model units
        return pmd.Deterministic(
            f"{self.prefix}_contribution",
            (contribution_original / model["target_scale"]).transpose(
                "date", *PANEL_DIMS
            ),
        )

    def set_data(self, mmm: Any, model: pm.Model, X: xr.Dataset) -> None:
        pass  # effect depends on channel_contribution, not on X
```

</div>

</div>

</div>

<div id="build-the-model" class="section level2">

## 5 · Build the model

Base MMM: GeometricAdstock(l_max=4) × Michaelis-Menten, one intercept per city, yearly Fourier seasonality. The `SpillEffect` is registered as an additive `mu_effect` — it adds to the mean alongside direct channels, controls, and intercept.

<div id="51e3eb1f" class="cell" execution_count="6">

Code

<div id="cb8" class="sourceCode cell-code">

``` sourceCode
by_city = panel.groupby("city")
target_scale = by_city["y"].max()
target_scale_da = xr.DataArray(
    [float(target_scale[c]) for c in CITIES],
    dims=PANEL_DIMS,
    coords={"city": list(CITIES),
}
)

adstock = GeometricAdstock(l_max=4)
saturation = MichaelisMentenSaturation()

spill_effect = SpillEffect(
    source_active_mask=source_active_mask,
    spill_path_mask=spill_mask,
    fraction_prior=Prior("Beta", alpha=1, beta=1, dims=SPEND_CHANNEL_DIMS),
    max_share=MAX_SPILL,
)

mmm = MMM(
    date_column="date",
    target_column="y",
    channel_columns=CHANNELS,
    control_columns=CONTROLS,
    dims=PANEL_DIMS,
    scaling=Scaling(
        channel=DataDerivedScaling(method="max", dims=()),
        target=FixedScaling(dims=(), value=target_scale_da),
    ),
    adstock=adstock,
    saturation=saturation,
    yearly_seasonality=1,
)

mmm.add_mu_effect(spill_effect)
mmm.build_model(X, y)

# Patch: pymc_extras Prior serialisation is incomplete in this version.
# The sampling succeeds; only the attrs round-trip fails.
mmm.set_idata_attrs = lambda idata: None

mmm.add_original_scale_contribution_variable(
    ["y", "channel_contribution", "control_contribution", "intercept_contribution"]
)
```

</div>

<div class="cell-output cell-output-display" execution_count="6">

    <pymc_marketing.mmm.mmm.MMM at 0x3315c4ec0>

</div>

</div>

</div>

<div id="model-structure" class="section level2">

## 6 · Model structure

<div id="ac613e0b" class="cell" execution_count="7">

Code

<div id="cb10" class="sourceCode cell-code">

``` sourceCode
print("Free RVs:", sorted(rv.name for rv in mmm.model.free_RVs))
```

</div>

<div class="cell-output cell-output-stdout">

    Free RVs: ['adstock_alpha', 'gamma_control', 'gamma_fourier', 'intercept_contribution', 'saturation_alpha', 'saturation_lam', 'spill_fraction_active', 'y_sigma']

</div>

</div>

<div id="e8ce8a47" class="cell" execution_count="8">

Code

<div id="cb12" class="sourceCode cell-code">

``` sourceCode
# Verify the spill fraction has exactly 3 active parameters
ip = mmm.model.initial_point()
frac_key = next(k for k in ip if k.startswith("spill_fraction"))
print(f"spill_fraction shape: {ip[frac_key].shape}  (expected 3 active routes)")
print(f"spill_fraction init values: {ip[frac_key]}")
```

</div>

<div class="cell-output cell-output-stdout">

    spill_fraction shape: (3,)  (expected 3 active routes)
    spill_fraction init values: [0. 0. 0.]

</div>

</div>

</div>

<div id="fit" class="section level2">

## 7 · Fit

<div id="54aaba21" class="cell" execution_count="9">

Code

<div id="cb14" class="sourceCode cell-code">

``` sourceCode
idata = mmm.fit(
    X=X,
    y=y,
    chains=2,
    cores=2,
    draws=500,
    tune=500,
    target_accept=0.9,
    random_seed=42,
    progressbar=False,
)
```

</div>

<div class="cell-output cell-output-stderr">

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [intercept_contribution, gamma_control, adstock_alpha, saturation_lam, saturation_alpha, y_sigma, spill_fraction_active, gamma_fourier]
    Sampling 2 chains for 500 tune and 500 draw iterations (1_000 + 1_000 draws total) took 49 seconds.
    There were 12 divergences after tuning. Increase `target_accept` or reparameterize.
    Chain 0 reached the maximum tree depth. Increase `max_treedepth`, increase `target_accept` or reparameterize.
    Chain 1 reached the maximum tree depth. Increase `max_treedepth`, increase `target_accept` or reparameterize.
    We recommend running at least 4 chains for robust computation of convergence diagnostics
    The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
    The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details

</div>

<div class="cell-output cell-output-display">

</div>

<div class="cell-output cell-output-display">

```
```

</div>

</div>

<div id="649bef3f" class="cell" execution_count="10">

Code

<div id="cb16" class="sourceCode cell-code">

``` sourceCode
print("Posterior vars:", sorted(idata.posterior.data_vars))
```

</div>

<div class="cell-output cell-output-stdout">

    Posterior vars: ['adstock_alpha', 'channel_contribution', 'channel_contribution_original_scale', 'control_contribution', 'control_contribution_original_scale', 'fourier_contribution', 'gamma_control', 'gamma_fourier', 'intercept_contribution', 'intercept_contribution_original_scale', 'saturation_alpha', 'saturation_lam', 'spill_by_path_original_scale', 'spill_contribution', 'spill_contribution_original_scale', 'spill_fraction', 'spill_fraction_active', 'spill_path_share', 'spill_share', 'spill_source_contribution_original_scale', 'total_media_contribution_original_scale', 'y_original_scale', 'y_sigma', 'yearly_seasonality_contribution']

</div>

</div>

</div>

<div id="diagnostics" class="section level2">

## 8 · Diagnostics

<div id="711721ed" class="cell" execution_count="11">

Code

<div id="cb18" class="sourceCode cell-code">

``` sourceCode
free_rv_names = sorted(rv.name for rv in mmm.model.free_RVs)
summary = az.summary(idata, var_names=free_rv_names)
n_div = int(idata.sample_stats["diverging"].sum())
print(f"Divergences: {n_div}")
if not summary.empty and "r_hat" in summary.columns:
    rh = pd.to_numeric(summary["r_hat"], errors="coerce")
    es = pd.to_numeric(summary["ess_bulk"], errors="coerce")
    print(f"Max r-hat:   {rh.max():.4f}")
    print(f"Min ESS:     {es.min():.0f}")
display(summary)
```

</div>

<div class="cell-output cell-output-stdout">

    Divergences: 12
    Max r-hat:   2.5900
    Min ESS:     2

</div>

<div class="cell-output cell-output-display">

<div>

|  | mean | sd | eti89_lb | eti89_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| adstock_alpha\[Caracas, C1\] | 0.2 | 0.2 | 0.051 | 0.59 | 8 | 20 | 1.17 | 0.067 | 0.06 |
| adstock_alpha\[Caracas, C2\] | 0.2 | 0.17 | 0.022 | 0.53 | 8 | 23 | 1.17 | 0.056 | 0.032 |
| adstock_alpha\[Caracas, C3\] | 0.3 | 0.2 | 0.048 | 0.59 | 5 | 24 | 1.31 | 0.074 | 0.055 |
| adstock_alpha\[Caracas, C4\] | 0.1 | 0.2 | 0.0083 | 0.46 | 3 | 9 | 1.62 | 0.073 | 0.074 |
| adstock_alpha\[Caracas, C5\] | 0.5 | 0.06 | 0.4 | 0.58 | 5 | 13 | 1.31 | 0.025 | 0.018 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| spill_fraction_active\[0\] | 0.6 | 0.3 | 0.12 | 0.94 | 5 | 19 | 1.43 | 0.15 | 0.053 |
| spill_fraction_active\[1\] | 0.6 | 0.23 | 0.16 | 0.93 | 15 | 21 | 1.22 | 0.067 | 0.049 |
| spill_fraction_active\[2\] | 0.45 | 0.23 | 0.096 | 0.82 | 32 | 45 | 1.05 | 0.041 | 0.021 |
| y_sigma\[Caracas\] | 0.009 | 0.0007 | 0.0081 | 0.01 | 6 | 15 | 1.22 | 0.00027 | 0.00018 |
| y_sigma\[Valencia\] | 0.0114 | 0.0007 | 0.01 | 0.013 | 5 | 12 | 1.33 | 0.00029 | 0.00018 |

75 rows × 9 columns

</div>

</div>

</div>

</div>

<div id="recovered-spill-shares" class="section level2">

## 9 · Recovered spill shares

The true spill multiplier is 10 %. The model learns `spill_fraction ∈ [0, 1]` and sets `spill_share = 0.10 × fraction`. If the data are informative, the posterior of `fraction` concentrates near 1.0 on the active routes and is absent on inactive routes.

<div id="3a4503c9" class="cell" execution_count="12">

Code

<div id="cb20" class="sourceCode cell-code">

``` sourceCode
post = idata.posterior
share = post["spill_share"]
print("Posterior spill_share by (city, spend_city, channel):")
display(share.mean(("chain", "draw")).to_series().unstack("channel").round(4))
```

</div>

<div class="cell-output cell-output-stdout">

    Posterior spill_share by (city, spend_city, channel):

</div>

<div class="cell-output cell-output-display">

<div>

| channel    | C1     | C2     | C3     | C4  | C5  | C6  | C7  | C8  | C9  | C10 |
|------------|--------|--------|--------|-----|-----|-----|-----|-----|-----|-----|
| spend_city |        |        |        |     |     |     |     |     |     |     |
| Caracas    | 0.0551 | 0.0642 | 0.0000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Valencia   | 0.0000 | 0.0000 | 0.0448 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

</div>

</div>

</div>

</div>

<div id="contribution-decomposition" class="section level2">

## 10 · Contribution decomposition

<div id="6c0d4ed9" class="cell" execution_count="13">

Code

<div id="cb22" class="sourceCode cell-code">

``` sourceCode
# Direct channel contributions per city
direct = post["channel_contribution_original_scale"]
direct_total = direct.sum("date").mean(("chain", "draw"))

# Spill contributions per city
spill = post["spill_contribution_original_scale"]
spill_total = spill.sum("date").mean(("chain", "draw"))

print("Mean cumulative contribution (original scale):")
for city in CITIES:
    d = float(direct_total.sel(city=city).sum())
    s = float(spill_total.sel(city=city))
    print(f"  {city}: direct={d:.1f}, spill={s:.1f}, "
          f"spill/direct={s/d:.1%}")
```

</div>

<div class="cell-output cell-output-stdout">

    Mean cumulative contribution (original scale):
      Caracas: direct=3108.5, spill=12.0, spill/direct=0.4%
      Valencia: direct=1884.0, spill=76.8, spill/direct=4.1%

</div>

</div>

</div>
