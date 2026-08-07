<a href="#quarto-document-content" class="skip-link">Skip to content</a>

<div id="title-block-header" class="quarto-title-block default">

<div class="quarto-title">

<div class="quarto-title-block">

<div>

# Cross-City Media Spill: A Synthetic MMM Dataset

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

Building a multi-geo MMM dataset with known cross-city media spillovers from prior-generator synthetic worlds.

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

<div id="setup" class="section level2">

## Setup

We load two independent synthetic worlds generated with [`prior-generator`](https://github.com/cetagostini/prior-generator) and treat them as two cities: **Valencia** and **Caracas**. Each has 10 media channels, 2 controls, and a sales target — plus the true per-channel contributions needed to engineer a controlled spillover.

<div id="1c905c0a" class="cell" execution_count="1">

Code

<div id="cb1" class="sourceCode cell-code">

``` sourceCode
import pandas as pd
import numpy as np
```

</div>

</div>

</div>

<div id="load-the-raw-worlds" class="section level2">

## 1 · Load the raw worlds

<div id="a50b5e88" class="cell" execution_count="2">

Code

<div id="cb2" class="sourceCode cell-code">

``` sourceCode
valencia_raw = pd.read_csv('../../data/valencia_raw.csv', parse_dates=['date'])
caracas_raw = pd.read_csv('../../data/caracas_raw.csv', parse_dates=['date'])

valencia_contrib = pd.read_csv('../../data/valencia_contributions.csv', parse_dates=['date'])
caracas_contrib = pd.read_csv('../../data/caracas_contributions.csv', parse_dates=['date'])

print(f'Valencia: {valencia_raw.shape[0]} weeks, {valencia_raw.shape[1]} cols')
print(f'Caracas:  {caracas_raw.shape[0]} weeks, {caracas_raw.shape[1]} cols')
valencia_raw.head(3)
```

</div>

<div class="cell-output cell-output-stdout">

    Valencia: 104 weeks, 15 cols
    Caracas:  104 weeks, 15 cols

</div>

<div class="cell-output cell-output-display" execution_count="2">

<div>

|  | date | city | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | Z1 | Z2 | Y |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 2025-01-06 | Valencia | 1.360327 | 0.850118 | 1.344349 | 5.125074 | 0.596736 | 1.243093 | 1.959474 | 4.335315 | 2.842975 | 1.994068 | 0.399335 | -0.100018 | 7.776969 |
| 1 | 2025-01-13 | Valencia | 1.445426 | 0.441765 | 1.420274 | 4.367961 | 3.824297 | 1.092566 | 1.791034 | 4.114020 | 0.957943 | 3.118692 | 0.321324 | -0.095795 | 7.782279 |
| 2 | 2025-01-20 | Valencia | 1.541429 | 0.615913 | 1.745829 | 5.057381 | 0.892164 | 1.409998 | 1.928290 | 1.939284 | 2.198756 | 1.808926 | 0.258592 | -0.091016 | 7.741244 |

</div>

</div>

</div>

The contribution columns tell us, for each week, how much each channel actually moved the target:

<div id="e6a75bfa" class="cell" execution_count="3">

Code

<div id="cb4" class="sourceCode cell-code">

``` sourceCode
contrib_cols = [c for c in valencia_contrib.columns if c.startswith('contrib_')]
valencia_contrib[contrib_cols].describe().round(2)
```

</div>

<div class="cell-output cell-output-display" execution_count="3">

<div>

|  | contrib_C1 | contrib_C2 | contrib_C3 | contrib_C4 | contrib_C5 | contrib_C6 | contrib_C7 | contrib_C8 | contrib_C9 | contrib_C10 | contrib_baseline |
|----|----|----|----|----|----|----|----|----|----|----|----|
| count | 104.0 | 104.0 | 104.0 | 104.00 | 104.0 | 104.0 | 104.00 | 104.0 | 104.0 | 104.00 | 104.00 |
| mean | 0.0 | 0.0 | 0.0 | 1.01 | 0.0 | 0.0 | 0.37 | 0.0 | 0.0 | 0.37 | 5.56 |
| std | 0.0 | 0.0 | 0.0 | 0.18 | 0.0 | 0.0 | 0.09 | 0.0 | 0.0 | 0.06 | 0.12 |
| min | 0.0 | 0.0 | 0.0 | 0.53 | 0.0 | 0.0 | 0.21 | 0.0 | 0.0 | 0.21 | 5.33 |
| 25% | 0.0 | 0.0 | 0.0 | 0.89 | 0.0 | 0.0 | 0.28 | 0.0 | 0.0 | 0.33 | 5.47 |
| 50% | 0.0 | 0.0 | 0.0 | 0.98 | 0.0 | 0.0 | 0.38 | 0.0 | 0.0 | 0.37 | 5.52 |
| 75% | 0.0 | 0.0 | 0.0 | 1.15 | 0.0 | 0.0 | 0.45 | 0.0 | 0.0 | 0.40 | 5.65 |
| max | 0.0 | 0.0 | 0.0 | 1.32 | 0.0 | 0.0 | 0.55 | 0.0 | 0.0 | 0.51 | 5.91 |

</div>

</div>

</div>

</div>

<div id="cross-city-spillover-construction" class="section level2">

## 2 · Cross-city spillover construction

We inject a **10 % media spillover** between cities:

- **Valencia** receives 10 % of Caracas’s C1 and C2 impact.
- **Caracas** receives 10 % of Valencia’s C3 impact.

This is a *controlled* perturbation — the true contribution magnitudes are known, so any downstream model can be scored against the injected signal.

<div id="7c044252" class="cell" execution_count="4">

Code

<div id="cb5" class="sourceCode cell-code">

``` sourceCode
MULTIPLIER = 0.10

# Spill INTO Valencia from Caracas C1 + C2
spill_valencia = MULTIPLIER * (
    caracas_contrib['contrib_C1'].values + caracas_contrib['contrib_C2'].values
)

# Spill INTO Caracas from Valencia C3
spill_caracas = MULTIPLIER * valencia_contrib['contrib_C3'].values

print(f'Spill → Valencia: mean {spill_valencia.mean():.4f}, sd {spill_valencia.std():.4f}')
print(f'Spill → Caracas:  mean {spill_caracas.mean():.4f}, sd {spill_caracas.std():.4f}')
```

</div>

<div class="cell-output cell-output-stdout">

    Spill → Valencia: mean 0.0000, sd 0.0000
    Spill → Caracas:  mean 0.0000, sd 0.0000

</div>

</div>

</div>

<div id="apply-the-spill-to-the-targets" class="section level2">

## 3 · Apply the spill to the targets

<div id="6f3a70d1" class="cell" execution_count="5">

Code

<div id="cb7" class="sourceCode cell-code">

``` sourceCode
valencia_mod = valencia_raw.copy()
valencia_mod['Y'] = valencia_mod['Y'] + spill_valencia

caracas_mod = caracas_raw.copy()
caracas_mod['Y'] = caracas_mod['Y'] + spill_caracas
```

</div>

</div>

</div>

<div id="combine-into-a-single-modelling-dataset" class="section level2">

## 4 · Combine into a single modelling dataset

<div id="e677d78d" class="cell" execution_count="6">

Code

<div id="cb8" class="sourceCode cell-code">

``` sourceCode
mmm_data_raw = pd.concat([valencia_mod, caracas_mod], ignore_index=True)
print(f'Combined raw: {mmm_data_raw.shape}')
mmm_data_raw.head(3)
```

</div>

<div class="cell-output cell-output-stdout">

    Combined raw: (208, 15)

</div>

<div class="cell-output cell-output-display" execution_count="6">

<div>

|  | date | city | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | Z1 | Z2 | Y |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 2025-01-06 | Valencia | 1.360327 | 0.850118 | 1.344349 | 5.125074 | 0.596736 | 1.243093 | 1.959474 | 4.335315 | 2.842975 | 1.994068 | 0.399335 | -0.100018 | 7.776969 |
| 1 | 2025-01-13 | Valencia | 1.445426 | 0.441765 | 1.420274 | 4.367961 | 3.824297 | 1.092566 | 1.791034 | 4.114020 | 0.957943 | 3.118692 | 0.321324 | -0.095795 | 7.782279 |
| 2 | 2025-01-20 | Valencia | 1.541429 | 0.615913 | 1.745829 | 5.057381 | 0.892164 | 1.409998 | 1.928290 | 1.939284 | 2.198756 | 1.808926 | 0.258592 | -0.091016 | 7.741244 |

</div>

</div>

</div>

</div>

<div id="build-the-contributions-dataset-truth-ledger" class="section level2">

## 5 · Build the contributions dataset (truth ledger)

<div id="e018e53e" class="cell" execution_count="7">

Code

<div id="cb10" class="sourceCode cell-code">

``` sourceCode
# Valencia truth: own contributions + spill received from Caracas
valencia_truth = valencia_contrib.copy()
valencia_truth['contrib_spill_from_caracas_C1'] = MULTIPLIER * caracas_contrib['contrib_C1'].values
valencia_truth['contrib_spill_from_caracas_C2'] = MULTIPLIER * caracas_contrib['contrib_C2'].values
valencia_truth['contrib_spill_total'] = (
    valencia_truth['contrib_spill_from_caracas_C1'] +
    valencia_truth['contrib_spill_from_caracas_C2']
)

# Caracas truth: own contributions + spill received from Valencia
caracas_truth = caracas_contrib.copy()
caracas_truth['contrib_spill_from_valencia_C3'] = spill_caracas
caracas_truth['contrib_spill_total'] = caracas_truth['contrib_spill_from_valencia_C3']

mmm_data_contributions = pd.concat([valencia_truth, caracas_truth], ignore_index=True)
print(f'Combined contributions: {mmm_data_contributions.shape}')
mmm_data_contributions.head(3)
```

</div>

<div class="cell-output cell-output-stdout">

    Combined contributions: (208, 17)

</div>

<div class="cell-output cell-output-display" execution_count="7">

<div>

|  | date | city | contrib_C1 | contrib_C2 | contrib_C3 | contrib_C4 | contrib_C5 | contrib_C6 | contrib_C7 | contrib_C8 | contrib_C9 | contrib_C10 | contrib_baseline | contrib_spill_from_caracas_C1 | contrib_spill_from_caracas_C2 | contrib_spill_total | contrib_spill_from_valencia_C3 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 2025-01-06 | Valencia | 0.0 | 0.0 | 0.0 | 1.234716 | 0.0 | 0.0 | 0.371157 | 0.0 | 0.0 | 0.331912 | 5.724500 | 0.0 | 0.0 | 0.0 | NaN |
| 1 | 2025-01-13 | Valencia | 0.0 | 0.0 | 0.0 | 1.242731 | 0.0 | 0.0 | 0.366281 | 0.0 | 0.0 | 0.343340 | 5.741899 | 0.0 | 0.0 | 0.0 | NaN |
| 2 | 2025-01-20 | Valencia | 0.0 | 0.0 | 0.0 | 1.275569 | 0.0 | 0.0 | 0.371623 | 0.0 | 0.0 | 0.292247 | 5.734829 | 0.0 | 0.0 | 0.0 | NaN |

</div>

</div>

</div>

</div>

<div id="verification" class="section level2">

## 6 · Verification

Sanity: the modified target should equal own contributions + baseline + spill.

<div id="461ec980" class="cell" execution_count="8">

Code

<div id="cb12" class="sourceCode cell-code">

``` sourceCode
own_cols = [c for c in mmm_data_contributions.columns if c.startswith('contrib_C')]

for city in ['Valencia', 'Caracas']:
    raw_rows = mmm_data_raw[mmm_data_raw['city'] == city].reset_index(drop=True)
    contrib_rows = mmm_data_contributions[mmm_data_contributions['city'] == city].reset_index(drop=True)

    own_sum = contrib_rows[own_cols].sum(axis=1)
    baseline = contrib_rows['contrib_baseline']
    spill = contrib_rows['contrib_spill_total']
    reconstructed = own_sum + baseline + spill

    diff = (raw_rows['Y'] - reconstructed).abs()
    print(f'{city}: max |Y − reconstructed| = {diff.max():.2e}')
```

</div>

<div class="cell-output cell-output-stdout">

    Valencia: max |Y − reconstructed| = 8.67e-01
    Caracas: max |Y − reconstructed| = 7.11e-01

</div>

</div>

</div>

<div id="save" class="section level2">

## 7 · Save

<div id="fc9b78fe" class="cell" execution_count="9">

Code

<div id="cb14" class="sourceCode cell-code">

``` sourceCode
mmm_data_raw.to_csv('../../data/mmm_data_raw.csv', index=False)
mmm_data_contributions.to_csv('../../data/mmm_data_contributions.csv', index=False)
print('Saved: data/mmm_data_raw.csv, data/mmm_data_contributions.csv')
```

</div>

<div class="cell-output cell-output-stdout">

    Saved: data/mmm_data_raw.csv, data/mmm_data_contributions.csv

</div>

</div>

</div>
