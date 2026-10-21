# USDA analysis — feasibility assessment (issue #13)

**Question.** Can we build a look-ahead-safe sample that joins USDA corn-report
*surprises* (actual − pre-report market expectation) to ZC price moves, as the
README's fundamental-overlay idea proposes?

**Short answer.** The **actuals and precise release timestamps are readily
obtainable**; the **pre-report market expectation is the binding constraint**
(no free, programmatic historical source). Recommendation: keep USDA as an
*optional* component, and if pursued, scope it to a small, manually-collected
sample rather than a full historical regression — investing the remaining time
in the trading-data pipeline (#11/#12) instead.

## What each column needs, and whether it's available

| Field | Source | Feasible? |
|-------|--------|-----------|
| `actual` (value at release) | USDA NASS **QuickStats** API (free key) for Crop Production & Grain Stocks; NASS also publishes a **release-time JSON** with the same data | **Yes** |
| `release_time` (precise) | NASS **release calendar**; major reports drop at **12:00 p.m. ET** (Grain Stocks at month-end; Jan combined release 12:05 p.m. ET) | **Yes** |
| at-release vs later **revisions** | Capture from the release-time JSON / original publication, not a later QuickStats refresh | Yes, with care |
| `expected` (pre-report consensus) | Trade-press analyst surveys (Reuters/Bloomberg/Dow Jones, brokerage notes) | **No clean/free source — the gap** |
| ZC price around the event | Databento trades we already pull (`data_store` / `bars`) | **Yes** |

## The expectations gap

A genuine "surprise" needs the market's expectation *as of just before the
release*. Those pre-report consensus figures are reported in paywalled trade
press and brokerage notes, not in any free API, and historical archives are not
cleanly machine-readable. Options considered:

- **Manual transcription** of a small sample (~8–15 recent Crop Production /
  Grain Stocks releases) from public recaps — workable for an illustrative
  study, not for a long regression.
- **Proxy expectations** (e.g. pre-release futures/options-implied moves) —
  measures anticipated *volatility*, not the expected *number*, so it does not
  cleanly isolate the surprise. Not recommended as a substitute.

## No look-ahead

`expected` must be timestamped strictly before `release_time`; `actual` must be
the value published *at* release, excluding later revisions; and the price
reaction must use only prices at/after the release. `src/finm37000/usda.py`
enforces the first two (`build_surprise_sample`) and the third
(`price_reaction` / `join_reactions`).

## What this PR delivers

- `usda.py`: `compute_surprise`, a look-ahead-safe `build_surprise_sample`, a
  windowed `price_reaction`, and `join_reactions` producing the small validated
  joined sample the acceptance criteria call for.
- `tests/test_usda.py`: value checks and the look-ahead / naive-timestamp /
  unknown-report / window-coverage rejections, all on synthetic data.
- `examples/usda_feasibility.py`: a runnable synthetic end-to-end demo.

The machinery is ready if the team decides to collect a small expectations
sample; the recommendation is that doing so is optional and lower priority than
#11/#12.

## Sources

- NASS QuickStats API & developer docs: https://quickstats.nass.usda.gov/api , https://www.nass.usda.gov/developer/index.php
- Release-time data file note (JSON at release): https://www.nass.usda.gov/developer/index.php
- August 2026 Crop Production release (12:00 p.m. ET, Aug 12, 2026): https://www.nass.usda.gov/Newsroom/2026/08-12-2026B.php
- Grain Stocks survey & schedule: https://www.nass.usda.gov/Surveys/Guide_to_NASS_Surveys/Off-Farm_Grain_Stocks/index.php
- CBOT corn (ZC) price units for the reaction side: cents/bushel, 5,000 bu — https://www.cmegroup.com/rulebook/CBOT/I/10.pdf
