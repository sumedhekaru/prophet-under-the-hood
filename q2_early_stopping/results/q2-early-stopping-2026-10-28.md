# Q2 · When does stopping short of the optimum help the forecast? (2026-10-28)

**Answer: in two cases, on both real sources, and not elsewhere.** First, wide holiday windows on weekly data. There,
most holiday columns have a single training row, and the exact optimum (X) forecasts far worse than Prophet's
early-stopped fit (S0): ± 15 days, holdout MASE +0.28 ± 0.04 (M5) and +0.85 ± 0.27 (Wikipedia). Second,
multiplicative seasonality. There, the exact optimum forecasts worse even without holidays, by 1.6–8.8% on the four
datasets (beyond or at 2 SE on three). With additive seasonality on daily data, X and S0 forecast equally well at every
window, up to ± 15 days. In every case X has the higher log posterior on every series, so this is overfitting at the
optimum, not a failure to find it.

## Setup

200 series each: M5 items (daily m5d and weekly m5w, US holidays) and Wikipedia pages (daily wtd and weekly wtw,
holidays of the language's country). The holidays come from Prophet's table, with day windows of ± 0 / 3 / 7 / 15
(`hol*`), additive or multiplicative (`-mult`). The fits are S0 (Prophet's default), S1 (L-BFGS run to
convergence) and X (exact MAP, prophet-batch 0.1.3); see Q1.

- **r:** median rank(holiday columns on the training rows) / T. This is the predictor prophet-batch uses to route
  fits.
- **One-row:** median share of nonempty holiday columns with exactly one nonzero training row.
- **Holiday rows:** holdout rows within ± 7 days of a holiday (daily) or holiday weeks (weekly).
- **L1 β X / S0:** median ratio of the L1 norm of the holiday and of the yearly coefficients.

Commands: `sh q2_early_stopping/run.sh`, then `python -m q2_early_stopping.analyze m5w wtw m5d wtd` and
`python -m q2_early_stopping.lp_check m5w wtw m5d wtd`.

## Results

**m5w**

| configuration | r | one-row | MASE S0 / S1 / X | X − S0 (± 2 SE) | holiday rows: S0 / X | X − S0 (± 2 SE) | L1 β X / S0: holiday, yearly |
|---|---:|---:|---|---|---|---|---|
| base | 0.000 | — | 1.049 / 1.049 / 1.059 | +0.0098 ± 0.0215 | 1.028 / 1.037 | +0.0087 ± 0.0202 | —, 1.00 |
| hol0 | 0.000 | — | 1.049 / 1.049 / 1.059 | +0.0098 ± 0.0215 | 1.028 / 1.037 | +0.0087 ± 0.0202 | —, 1.00 |
| hol3 | 0.098 | 76% | 1.043 / 1.043 / 1.052 | +0.0094 ± 0.0225 | 1.030 / 1.038 | +0.0080 ± 0.0219 | 1.03, 1.00 |
| hol7 | 0.178 | 76% | 1.045 / 1.044 / 1.065 | +0.0199 ± 0.0237 | 1.042 / 1.068 | +0.0258 ± 0.0263 | 1.18, 1.14 |
| hol15 | 0.342 | 77% | 1.058 / 1.060 / 1.339 | +0.2812 ± 0.0412 | 1.052 / 1.764 | +0.7118 ± 0.1016 | 2.69, 3.08 |
| base-mult | 0.000 | — | 1.244 / 1.245 / 1.304 | +0.0598 ± 0.0848 | 1.212 / 1.325 | +0.1131 ± 0.1625 | —, 1.00 |
| hol0-mult | 0.000 | — | 1.244 / 1.245 / 1.304 | +0.0598 ± 0.0848 | 1.212 / 1.325 | +0.1131 ± 0.1625 | —, 1.00 |
| hol3-mult | 0.098 | 76% | 1.248 / 1.248 / 1.309 | +0.0616 ± 0.0796 | 1.216 / 1.337 | +0.1207 ± 0.1519 | 2.02, 1.02 |
| hol7-mult | 0.178 | 76% | 1.302 / 1.305 / 1.386 | +0.0841 ± 0.0679 | 1.297 / 1.439 | +0.1415 ± 0.1233 | 1.65, 1.23 |
| hol15-mult | 0.342 | 77% | 1.331 / 1.333 / 2.146 | +0.8147 ± 0.2508 | 1.309 / 3.011 | +1.7023 ± 0.4980 | 9.52, 9.07 |

**wtw**

| configuration | r | one-row | MASE S0 / S1 / X | X − S0 (± 2 SE) | holiday rows: S0 / X | X − S0 (± 2 SE) | L1 β X / S0: holiday, yearly |
|---|---:|---:|---|---|---|---|---|
| base | 0.000 | — | 1.074 / 1.075 / 1.084 | +0.0100 ± 0.0078 | 1.092 / 1.102 | +0.0100 ± 0.0082 | —, 1.00 |
| hol0 | 0.026 | 71% | 1.075 / 1.074 / 1.084 | +0.0092 ± 0.0073 | 1.096 / 1.109 | +0.0133 ± 0.0082 | 1.08, 1.00 |
| hol3 | 0.087 | 74% | 1.099 / 1.101 / 1.114 | +0.0145 ± 0.0080 | 1.173 / 1.199 | +0.0260 ± 0.0111 | 1.04, 1.01 |
| hol7 | 0.189 | 74% | 1.154 / 1.153 / 1.214 | +0.0595 ± 0.0161 | 1.334 / 1.444 | +0.1096 ± 0.0302 | 1.14, 1.04 |
| hol15 | 0.321 | 76% | 1.322 / 1.324 / 2.171 | +0.8488 ± 0.2698 | 1.747 / 3.822 | +2.0749 ± 0.6587 | 1.92, 1.69 |
| base-mult | 0.000 | — | 1.069 / 1.069 / 1.086 | +0.0171 ± 0.0121 | 1.038 / 1.054 | +0.0159 ± 0.0199 | —, 1.02 |
| hol0-mult | 0.026 | 71% | 1.083 / 1.083 / 1.113 | +0.0301 ± 0.0273 | 1.058 / 1.101 | +0.0432 ± 0.0492 | 16.51, 1.03 |
| hol3-mult | 0.087 | 74% | 1.103 / 1.106 / 1.183 | +0.0793 ± 0.0450 | 1.084 / 1.212 | +0.1275 ± 0.0880 | 5.92, 1.05 |
| hol7-mult | 0.189 | 74% | 1.127 / 1.129 / 1.268 | +0.1410 ± 0.0446 | 1.157 / 1.425 | +0.2683 ± 0.1097 | 5.44, 1.37 |
| hol15-mult | 0.321 | 76% | 1.138 / 1.154 / 1.896 | +0.7586 ± 0.3949 | 1.255 / 2.766 | +1.5115 ± 0.7203 | 8.28, 3.23 |

**m5d**

| configuration | r | one-row | MASE S0 / S1 / X | X − S0 (± 2 SE) | holiday rows: S0 / X | X − S0 (± 2 SE) | L1 β X / S0: holiday, yearly |
|---|---:|---:|---|---|---|---|---|
| base | 0.000 | — | 1.989 / 1.990 / 2.002 | +0.0127 ± 0.0174 | 1.961 / 1.976 | +0.0147 ± 0.0169 | —, 1.00 |
| hol0 | 0.008 | 23% | 1.995 / 1.989 / 2.000 | +0.0054 ± 0.0183 | 1.963 / 1.970 | +0.0068 ± 0.0179 | 1.00, 1.00 |
| hol3 | 0.058 | 23% | 2.001 / 2.005 / 2.012 | +0.0111 ± 0.0137 | 1.983 / 1.995 | +0.0125 ± 0.0138 | 1.00, 1.00 |
| hol7 | 0.114 | 23% | 2.025 / 2.024 / 2.030 | +0.0050 ± 0.0153 | 2.021 / 2.028 | +0.0071 ± 0.0149 | 1.00, 1.00 |
| hol15 | 0.225 | 23% | 2.085 / 2.084 / 2.090 | +0.0057 ± 0.0112 | 2.079 / 2.088 | +0.0090 ± 0.0118 | 1.24, 1.42 |
| base-mult | 0.000 | — | 2.274 / 2.276 / 2.474 | +0.1993 ± 0.1979 | 2.222 / 2.429 | +0.2076 ± 0.2154 | —, 1.00 |
| hol0-mult | 0.008 | 23% | 2.290 / 2.292 / 2.478 | +0.1880 ± 0.1701 | 2.236 / 2.432 | +0.1961 ± 0.1914 | 1.34, 1.00 |
| hol3-mult | 0.058 | 23% | 2.362 / 2.365 / 2.573 | +0.2105 ± 0.1577 | 2.315 / 2.539 | +0.2242 ± 0.1723 | 1.29, 1.02 |
| hol7-mult | 0.114 | 23% | 2.481 / 2.489 / 2.617 | +0.1358 ± 0.1486 | 2.475 / 2.645 | +0.1697 ± 0.1842 | 1.27, 1.13 |
| hol15-mult | 0.225 | 23% | 2.789 / 2.795 / 3.217 | +0.4289 ± 0.2439 | 2.702 / 3.123 | +0.4215 ± 0.2535 | 6.32, 8.46 |

**wtd**

| configuration | r | one-row | MASE S0 / S1 / X | X − S0 (± 2 SE) | holiday rows: S0 / X | X − S0 (± 2 SE) | L1 β X / S0: holiday, yearly |
|---|---:|---:|---|---|---|---|---|
| base | 0.000 | — | 1.579 / 1.579 / 1.574 | -0.0049 ± 0.0071 | 1.590 / 1.585 | -0.0050 ± 0.0083 | —, 1.00 |
| hol0 | 0.005 | 10% | 1.582 / 1.582 / 1.578 | -0.0033 ± 0.0072 | 1.611 / 1.606 | -0.0048 ± 0.0076 | 1.00, 1.00 |
| hol3 | 0.035 | 10% | 1.630 / 1.629 / 1.626 | -0.0037 ± 0.0068 | 1.731 / 1.725 | -0.0064 ± 0.0076 | 1.00, 1.00 |
| hol7 | 0.064 | 10% | 1.669 / 1.671 / 1.670 | +0.0003 ± 0.0076 | 1.801 / 1.800 | -0.0012 ± 0.0091 | 1.00, 1.00 |
| hol15 | 0.115 | 10% | 1.793 / 1.793 / 1.792 | -0.0019 ± 0.0063 | 1.931 / 1.929 | -0.0024 ± 0.0075 | 1.04, 1.04 |
| base-mult | 0.000 | — | 1.548 / 1.555 / 1.585 | +0.0365 ± 0.0314 | 1.518 / 1.547 | +0.0292 ± 0.0286 | —, 1.01 |
| hol0-mult | 0.005 | 10% | 1.568 / 1.609 / 1.640 | +0.0718 ± 0.0610 | 1.543 / 1.622 | +0.0795 ± 0.0723 | 8.95, 1.02 |
| hol3-mult | 0.035 | 10% | 1.661 / 1.634 / 1.736 | +0.0755 ± 0.1176 | 1.638 / 1.794 | +0.1553 ± 0.1344 | 2.54, 1.05 |
| hol7-mult | 0.064 | 10% | 1.702 / 1.708 / 1.825 | +0.1231 ± 0.1178 | 1.696 / 1.899 | +0.2021 ± 0.1492 | 2.71, 1.20 |
| hol15-mult | 0.115 | 10% | 1.772 / 1.718 / 2.190 | +0.4181 ± 0.4840 | 1.756 / 2.198 | +0.4414 ± 0.3685 | 6.82, 3.32 |


Overfitting or a worse optimum? lp of X against the stock fits (`lp_check`):

| dataset | configuration | lp X − S0: median / min | X below S0 | X below S1 | X fell back to Stan |
|---|---|---|---:|---:|---:|
| m5w | base | 1.03 / 0.00127 | 0% | 0% | 0% |
| m5w | hol7 | 1.82 / 0.206 | 0% | 0% | 0% |
| m5w | hol15 | 5.19 / 0.977 | 0% | 0% | 0% |
| m5w | base-mult | 1.2 / 0.0217 | 0% | 0% | 0% |
| m5w | hol7-mult | 4.93 / 0.774 | 0% | 0% | 0% |
| m5w | hol15-mult | 23.3 / 2.3 | 0% | 0% | 0% |
| wtw | base | 0.172 / 0.000434 | 0% | 0% | 0% |
| wtw | hol7 | 0.473 / 0.0247 | 0% | 0% | 0% |
| wtw | hol15 | 1.45 / 0.115 | 0% | 0% | 0% |
| wtw | base-mult | 1.08 / 0.0104 | 0% | 0% | 0% |
| wtw | hol7-mult | 15 / 0 | 0% | 0% | 0% |
| wtw | hol15-mult | 33.1 / 1.1 | 0% | 0% | 0% |
| m5d | base | 2.69 / 0.0672 | 0% | 0% | 0% |
| m5d | hol7 | 2.35 / 0.144 | 0% | 0% | 0% |
| m5d | hol15 | 3.45 / 0.382 | 0% | 0% | 0% |
| m5d | base-mult | 2.89 / 0.0969 | 0% | 0% | 0% |
| m5d | hol7-mult | 7.33 / 0.372 | 0% | 0% | 0% |
| m5d | hol15-mult | 55.5 / 2.74 | 0% | 0% | 0% |
| wtd | base | 0.602 / 0.000187 | 0% | 0% | 0% |
| wtd | hol7 | 0.625 / 0.00781 | 0% | 0% | 0% |
| wtd | hol15 | 0.714 / 0.0248 | 0% | 0% | 0% |
| wtd | base-mult | 1.23 / 0 | 0% | 6% | 6% |
| wtd | hol7-mult | 28.6 / 0 | 0% | 6% | 6% |
| wtd | hol15-mult | 86.1 / 0 | 0% | 6% | 6% |

## Findings

1. **Wide holiday windows on weekly data: stopping short helps a lot.** At ± 15 days the exact optimum is worse on
   both sources (+0.28 ± 0.04 and +0.85 ± 0.27 MASE overall, +0.71 and +2.07 on holiday weeks). At ± 7 days it is
   worse on Wikipedia (+0.06 ± 0.02). 71–77% of the holiday columns have a single training row. The exact optimum
   fits those rows with holiday and yearly terms 2–9× larger in L1 that cancel each other in training. Next year's
   holiday columns are untrained (Q3), so the forecast keeps the distorted yearly term. Prophet's early stop leaves
   both terms small.
2. **Multiplicative seasonality: stopping short helps even without holidays.** X − S0 for `base-mult` is +0.017 ±
   0.012 (wtw), +0.037 ± 0.031 (wtd), +0.20 ± 0.20 (m5d) and +0.06 ± 0.08 (m5w); with holidays the gap grows with the
   window. X's lp is above S0's on every series (by up to 86 at ± 15 days), so the multiplicative optimum overfits.
3. **Additive daily data: no effect.** X and S0 are within 2 SE at every window on m5d and wtd. Daily window columns
   get one row per year of history, so few have a single row (10–23%).
4. **r alone does not predict the harm.** Daily m5d at ± 15 days has r = 0.23 and no gap, while weekly ± 7 days has
   r = 0.18 and a significant gap on Wikipedia. The share of single-row columns separates them (23% vs 74%).

## Limits

- Two sources, four datasets; additive daily windows were tested only up to ± 15 days.
- X is the exact MAP for additive models. For multiplicative models the problem is non-convex and X is a verified
  stationary point; its lp is above both stock fits on every series, but it may not be the global optimum.
- 6% of wtd multiplicative series failed prophet-batch's optimality check and were fit by Stan.
- Results recorded at repo commit e404142 + uncommitted code; committed with this page.
