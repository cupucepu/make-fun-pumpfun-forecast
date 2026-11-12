# Pump.fun top-5 forecast — make.fun pumpfun-top5-forecast-01

Forecasts the 5 largest pump.fun coins by market cap at each of seven
answer reads, as probabilities per place plus per-place market-cap bins.

## Approach

Not persistence ("the last chart holds"). Every coin's log10 market cap
(in SOL) is treated as a noisy, drifting process and the seven answer
reads are Monte-Carlo simulated (300 draws × 7 horizons, one fixed seed):

- per coin: last log-cap, recency-weighted hourly drift (shrunk and
  capped), per-coin volatility, venue/age adjustments, mild mean
  reversion toward the coin's own median log-cap;
- per read: log-cap ~ Normal(mu(horizon), sigma(horizon)), sigma growing
  with the square root of the horizon (horizons read from `task.json`);
- place probabilities = simulated place frequencies, averaged over the
  reads, scaled to leave an unnamed share at each place for coins born
  after the history cut (which no forecast can name);
- size probabilities = per-place log-cap trend fits (recency-weighted
  linear trend of each place's own cap across the marks, extrapolated
  to each read, Normal sd widening with sqrt horizon), binned and
  averaged over the reads.

Stdlib only, deterministic (fixed seed, sorted candidates), no network.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes `out/forecast.json` with exactly three keys: `probabilities`
(`{mint, place, p}`), `top5` (5 distinct mints), `sizes`
(`{place, bin, p}`).
