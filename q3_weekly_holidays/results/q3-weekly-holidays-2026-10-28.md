# Q3 · Do holiday windows given in days carry into the forecast on weekly data? (2026-10-28)

**Answer: partly, and for the major holidays mostly not. Fixing it does not improve accuracy on these data.**
Prophet's holiday windows are counted in days. On weekly data a fixed-date holiday falls on a different weekday every
year, so it activates different window columns each year. For the major holidays, 54–67% of the holdout's holiday
weeks activate only columns that had no training row, and the forecast keeps 0–26% of the holiday effect the fit
learned. Snapping holidays to their week (week-aligned windows) removes this: every holdout holiday week uses trained
columns, and the forecast keeps the full effect. Holidays do not improve holdout MASE over no holidays on either
source, and week-aligned windows beat day windows beyond 2 SE in one case only (Wikipedia, ± 1 week vs ± 15 days).

## Setup

Stock Prophet, default fit (S0), 200 series each of M5 items (weekly, Saturday weeks, US holidays) and Wikipedia
pages (weekly, Monday weeks, holidays of the language's country: en → US, ja → JP, de → DE, fr → FR, ru → RU). The
holidays come from Prophet's own table (`prophet.make_holidays.make_holidays_df`).

- **Configurations:**
  - `base`: no holidays.
  - `hol0` / `hol3` / `hol7` / `hol15`: day windows of ± 0 / 3 / 7 / 15 days.
  - `wk0` / `wk1`: each occurrence moved to the start of its week, with a window of ± 0 / 1 week
    (`common.data.holidays_by_week`, plain pandas).
  - `-major`: only New Year, Christmas, Easter where it is a public holiday, and US Thanksgiving.
- **Untrained:** the share of holdout holiday weeks whose active holiday columns all had no training row, so their
  coefficients stay at the prior mean of 0.
- **Effect:** the mean |holiday component| on holiday weeks, as a share of the series' mean |y|, in the holdout
  against in training.
- **Holiday weeks:** weeks that contain a (major) holiday date. They are the same weeks for every configuration.

Commands: `sh q3_weekly_holidays/run.sh`, then `python -m q3_weekly_holidays.analyze m5w wtw`.

## Results
**m5w** (200 series, 11.0 holiday weeks of 52 in the holdout on average)

Major holidays only: 3.0 holiday weeks of 52; "-major" rows are scored on those weeks, the others on all holiday weeks.

| configuration | untrained holdout holiday weeks | effect, holdout / training | MASE all | vs base (± 2 SE) | MASE holiday weeks | vs base (± 2 SE) |
|---|---:|---|---:|---|---:|---|
| base | — | — | 1.049 | — | 1.028 | — |
| hol0 | 100% | 0.0% / 0.0% | 1.049 | +0.0000 ± 0.0000 | 1.028 | +0.0000 ± 0.0000 |
| hol3 | 40% | 8.4% / 19.5% | 1.043 | -0.0064 ± 0.0094 | 1.030 | +0.0020 ± 0.0128 |
| hol7 | 40% | 13.4% / 31.9% | 1.045 | -0.0047 ± 0.0125 | 1.042 | +0.0139 ± 0.0205 |
| hol15 | 32% | 22.4% / 44.7% | 1.058 | +0.0082 ± 0.0243 | 1.052 | +0.0241 ± 0.0262 |
| wk0 | 9% | 18.1% / 20.7% | 1.051 | +0.0021 ± 0.0054 | 1.040 | +0.0115 ± 0.0110 |
| wk1 | 4% | 24.9% / 27.0% | 1.061 | +0.0119 ± 0.0123 | 1.048 | +0.0200 ± 0.0157 |
| hol0-major | — | 0.0% / 0.0% | 1.049 | +0.0000 ± 0.0000 | 1.104 | +0.0000 ± 0.0000 |
| hol3-major | 67% | 0.0% / 17.6% | 1.047 | -0.0019 ± 0.0070 | 1.105 | +0.0013 ± 0.0205 |
| hol7-major | 60% | 9.9% / 38.1% | 1.050 | +0.0007 ± 0.0087 | 1.150 | +0.0461 ± 0.0458 |
| wk0-major | 0% | 27.7% / 27.7% | 1.049 | +0.0001 ± 0.0040 | 1.123 | +0.0187 ± 0.0281 |
| wk1-major | 0% | 33.8% / 33.8% | 1.046 | -0.0038 ± 0.0078 | 1.116 | +0.0122 ± 0.0271 |

Week-aligned vs day windows, paired (holiday-week MASE; negative = week-aligned better):

| week-aligned | day windows | Δ holiday weeks (± 2 SE) | Δ all weeks (± 2 SE) |
|---|---|---|---|
| wk0 | hol3 | +0.0095 ± 0.0131 | +0.0085 ± 0.0088 |
| wk1 | hol7 | +0.0060 ± 0.0206 | +0.0165 ± 0.0113 |
| wk1 | hol15 | -0.0042 ± 0.0260 | +0.0037 ± 0.0216 |
| wk0-major | hol3-major | +0.0173 ± 0.0382 | +0.0020 ± 0.0070 |
| wk1-major | hol7-major | -0.0339 ± 0.0507 | -0.0045 ± 0.0089 |

**wtw** (200 series, 10.5 holiday weeks of 52 in the holdout on average)

Major holidays only: 3.4 holiday weeks of 52; "-major" rows are scored on those weeks, the others on all holiday weeks.

| configuration | untrained holdout holiday weeks | effect, holdout / training | MASE all | vs base (± 2 SE) | MASE holiday weeks | vs base (± 2 SE) |
|---|---:|---|---:|---|---:|---|
| base | — | — | 1.074 | — | 1.092 | — |
| hol0 | 17% | 11.7% / 23.0% | 1.075 | +0.0006 ± 0.0101 | 1.096 | +0.0039 ± 0.0284 |
| hol3 | 46% | 15.8% / 43.7% | 1.099 | +0.0250 ± 0.0177 | 1.173 | +0.0804 ± 0.0461 |
| hol7 | 40% | 37.7% / 86.0% | 1.154 | +0.0801 ± 0.0462 | 1.334 | +0.2418 ± 0.1170 |
| hol15 | 39% | 73.4% / 114.8% | 1.322 | +0.2479 ± 0.0687 | 1.747 | +0.6551 ± 0.1957 |
| wk0 | 0% | 58.9% / 49.6% | 1.098 | +0.0240 ± 0.0237 | 1.192 | +0.0998 ± 0.0843 |
| wk1 | 0% | 122.1% / 112.1% | 1.150 | +0.0762 ± 0.0403 | 1.269 | +0.1769 ± 0.0961 |
| hol0-major | 0% | 4.3% / 13.1% | 1.083 | +0.0083 ± 0.0060 | 1.258 | +0.0413 ± 0.0374 |
| hol3-major | 67% | 8.2% / 47.1% | 1.088 | +0.0139 ± 0.0127 | 1.350 | +0.1334 ± 0.0905 |
| hol7-major | 54% | 14.2% / 85.4% | 1.118 | +0.0439 ± 0.0383 | 1.596 | +0.3801 ± 0.2624 |
| wk0-major | 0% | 54.2% / 55.6% | 1.070 | -0.0043 ± 0.0184 | 1.159 | -0.0568 ± 0.1450 |
| wk1-major | 0% | 119.9% / 121.2% | 1.080 | +0.0062 ± 0.0213 | 1.175 | -0.0413 ± 0.1402 |

Week-aligned vs day windows, paired (holiday-week MASE; negative = week-aligned better):

| week-aligned | day windows | Δ holiday weeks (± 2 SE) | Δ all weeks (± 2 SE) |
|---|---|---|---|
| wk0 | hol3 | +0.0194 ± 0.1012 | -0.0010 ± 0.0303 |
| wk1 | hol7 | -0.0649 ± 0.1425 | -0.0039 ± 0.0436 |
| wk1 | hol15 | -0.4782 ± 0.2006 | -0.1717 ± 0.0728 |
| wk0-major | hol3-major | -0.1902 ± 0.1792 | -0.0183 ± 0.0214 |
| wk1-major | hol7-major | -0.4215 ± 0.3037 | -0.0377 ± 0.0341 |


## Findings

1. **Day windows lose most of a major holiday's effect in the forecast.** With the major holidays and ± 3 or ± 7
   days, 54–67% of holdout holiday weeks are untrained, and the forecast's holiday effect is 0–26% of the one learned
   in training (M5: 0.0% / 17.6% and 9.9% / 38.1%; Wikipedia: 8.2% / 47.1% and 14.2% / 85.4%). With all of a
   country's holidays the loss is smaller (32–46% untrained) because some holidays are fixed to a weekday
   (Thanksgiving, Memorial Day).
2. **Zero-width day windows can switch holidays off entirely.** On M5's Saturday weeks, `hol0` puts a holiday into
   the model only when it falls on a Saturday: 100% of holdout holiday weeks are untrained, and the fit is identical
   to `base`.
3. **Week-aligned windows carry the full effect into the forecast** (0–9% untrained; holdout effect ≈ training
   effect).
4. **None of this improves accuracy here.** No holiday configuration beats `base` beyond 2 SE on either source. On
   Wikipedia, adding a country's full holiday list makes forecasts worse, the more so the wider the window (± 15 days:
   +0.25 ± 0.07 MASE). Week-aligned windows beat day windows beyond 2 SE only against ± 15 days on Wikipedia (−0.48
   ± 0.20 on holiday weeks, −0.17 ± 0.07 overall).

## Limits

- Two sources, both of which respond weakly to holidays at the weekly level (retail unit sales of single items,
  page views). Data with strong holiday peaks (store sales, travel) may show the accuracy gain that is absent here.
- Country holidays assigned from the page language (en → US) are a rough proxy for where the readers are.
- About 3 major-holiday weeks per holdout year: the holiday-week MASE carries large standard errors.
- Results recorded at repo commit e404142 + uncommitted code; committed with this page.
