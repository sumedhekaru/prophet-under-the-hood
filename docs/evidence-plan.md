# Evidence plan

What each question needs before it can be stated as a finding, and how it is tested. Every run uses stock Prophet
1.5.0. prophet-batch is used only where the exact optimum is needed, and every claim must also hold with the
stock-only reference (S1 below). A question counts as answered only with evidence from at least two independent real
data sources.

## Data

| id | source | series used | frequency, length | dates | used for |
|---|---|---|---|---|---|
| M5d | M5 (Walmart unit sales), items summed over stores | 200 sampled of 3,049 | daily, 1,941 days | real (US) | Q1, Q2 |
| M5w | the same items summed to Walmart weeks (Sat–Fri) | 200 | weekly, 277 weeks | real (US) | Q1–Q3 |
| WTd | Wikipedia web traffic, extended (Monash, Zenodo 7371038): all-agents pages of the 7 largest language projects, mean ≥ 100 views a day, country holidays from the language | 200 sampled | daily, 2,557 days | real | Q1, Q2 |
| WTw | the same pages summed to weeks (Mon–Sun) | 200 | weekly, 365 weeks | real | Q1–Q3 |
| M4w, M4d, TM | M4 Weekly, M4 Daily, Tourism Monthly | 200 each | weekly / daily / monthly | not meaningful | Q1 only |
| short | M5w and WTw cut to their first 104 weeks (T ≥ 100: L-BFGS) and 80 weeks (Newton) | 200 each | weekly | real | Q1–Q3 |
| syn | synthetic series with known components | 100 per setting | weekly / daily | real calendar | controls |

Samples are seeded, and the sampled ids are committed with the results. The holdout is the last 52 weeks (weekly) or
365 days (daily), so every holiday appears once in it. Series that are all zero in training are dropped, and the
count is reported.

## Fits

- **S0 stock default:** `Prophet(...).fit(df)`, which is what users run. Prophet uses Newton below 100 rows and
  L-BFGS from 100.
- **S1 stock, run to convergence:** `fit(df, algorithm='LBFGS', iter=100000, tol_obj=1e-12, tol_rel_obj=1,
  tol_grad=1e-12, tol_rel_grad=1, tol_param=1e-12)`. This is the stock-only reference optimum.
- **S2 stock Newton:** `fit(df, algorithm='Newton')`.
- **X exact MAP:** prophet-batch `fit_many`, with its routing of wide-holiday series to Stan switched off.

The log posterior of every fit is evaluated by one function written from Prophet's model (priors k, m ~ N(0, 5),
δ ~ Laplace(0, τ), σ ~ half-N(0, 0.5), β ~ N(0, σ_j), likelihood y ~ N(ŷ, σ) on the scaled data), so fits from
different optimizers are compared on the same scale. It is checked against Stan's `lp__` up to a constant.

## Q1 · Does Prophet's default fit reach the optimum of its own model?

- **Measure:** per series, Δlp = lp(S1) − lp(S0), and the same for X; the in-sample and forecast change from S0 to
  S1, as a share of the series' mean |y|. Report median, p90 and max per dataset, split by optimizer (Newton below 100
  rows, L-BFGS from 100).
- **Checks:** S1 and X agree (Δlp within 1e-6 relative); S2 vs S1 for series under 100 rows.
- **Answered when:** the Δlp and forecast-change distributions are known on ≥ 2 real sources. The claim "the default
  fit stops short" stands only if a material share of L-BFGS series (≥ 10%) changes forecasts by ≥ 1% when run to
  convergence.

## Q2 · When does stopping short help the forecast?

- **Configurations:** base model; + country holidays (window 0); + holidays with windows ± 3 / ± 7 / ± 15 days;
  additive and multiplicative; on M5d, M5w, WTd, WTw.
- **Measure:** holdout MASE (seasonal-naive scale; m = 7 daily, 52 weekly) of S0 vs S1 (and X), paired per series,
  as Δ ± 2 SE; also MASE on holiday weeks or days only. Per configuration, r = rank(holiday columns) / T.
- **Mechanism:** on representative series, the yearly and holiday components of S0 vs S1, the L1 norms of the
  holiday and yearly coefficients, and how many holiday columns have exactly one training row.
- **Answered when:** ΔMASE is known across configurations on ≥ 2 sources, together with where it turns significant.
  The claim "stopping short helps" stands only if S0 is significantly better (beyond 2 SE) in some configurations on
  ≥ 2 sources, and the mechanism is shown on real series.

## Q3 · Do holiday windows given in days carry into the forecast on weekly data?

- **Configurations:** on M5w and WTw, holidays given as (a) Prophet's day windows ± 0 / ± 3 / ± 7 / ± 15 days;
  (b) week-aligned windows built in plain pandas, the holiday's week only and ± 1 week; (c) no holidays. S0 fits only:
  this is about Prophet as users run it.
- **Measure:** (i) for each holiday occurrence in the holdout year, which holiday columns it activates and whether
  those had any training row; (ii) the forecast's holiday component on the holdout's holiday weeks; (iii) MASE on
  holiday weeks and overall, paired, vs no holidays.
- **Answered when:** (i)–(iii) are known on ≥ 2 sources. The claim stands only if fixed-date holidays with day
  windows give near-zero forecast effect (most holdout columns untrained), and week-aligned windows improve
  holiday-week MASE beyond 2 SE.

## Layout

- `common/`: downloads (URL + sha256), loaders, the log posterior, fitting helpers, metrics.
- `q1_optimum/`, `q2_early_stopping/`, `q3_weekly_holidays/`: one script per run, with `results/*.md`. Each page
  records the machine, package versions and repo commit. Raw outputs are not committed.

## Not in scope

- Whether Prophet is a good model compared with others. That is covered elsewhere, and these questions are about
  Prophet on its own terms.
- MCMC fits, logistic growth.
