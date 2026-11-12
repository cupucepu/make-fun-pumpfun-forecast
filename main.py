#!/usr/bin/env python3
"""Pump.fun top-5 forecast (make.fun pumpfun-top5-forecast-01).

Approach (deliberately NOT "the last chart holds"):
  pump.fun leaders churn fast and new coins are born constantly, so a
  persistence forecast leaves most of the score on the table.  This program
  treats every coin's log10 market cap (in SOL) as a noisy, drifting
  process and Monte-Carlo simulates the seven answer reads:
    * per coin: last log-cap, recency-weighted hourly drift (shrunk),
      per-coin volatility, venue/age adjustments, mild mean-reversion
      toward the coin's own median log-cap;
    * per answer read (horizons read from task.json, never hard-coded):
      log-cap ~ Normal(mu(horizon), sigma(horizon)), sigma growing with
      the square root of the horizon;
    * place probabilities = simulated place frequencies, averaged over
      the seven reads, then scaled down so each place leaves an unnamed
      share for coins born after the cut (which nobody can name);
    * size probabilities = per-place log-cap trend fits (recency-
      weighted linear trend of each place's own cap across the marks,
      extrapolated to each read, Normal sd widening with sqrt horizon),
      binned and averaged over the reads.
  Everything is deterministic: one fixed RNG seed, sorted candidate
  order, no wall clock, no network.  Stdlib only.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 987654321
N_SIMS = 300
# Unnamed share left at each place (newcomers born after the cut + coins
# outside the candidate set).  Place 1 is the most likely to be taken by
# a newcomer, so it keeps the largest unnamed share.
PLACE_TARGET = {1: 0.65, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.75}
SIZE_TARGET = 0.90
MIN_P = 0.002
# Drift (log10 cap per hour) is shrunk and capped: short histories and
# wild 10-minute jumps must not dominate a 7-13h forecast.
DRIFT_SHRINK_N = 8.0
DRIFT_CAP = 0.25
# Momentum persists for about this long (hours); beyond it the drift
# contribution saturates.
DRIFT_TAU_H = 2.5
# Mean reversion toward the coin's own median log-cap, half-life ~12h.
REVERT_HALF_LIFE_H = 12.0
REVERT_STRENGTH = 0.30
VOL_FLOOR_H = 0.08
VOL_CAP_H = 0.40
SIGMA0 = 0.10
# Size model: place-level log-cap trend (per hour, clipped) + sd that
# widens with the square root of the horizon.
SIZE_SD0 = 0.30
SIZE_SD_H = 0.06
SIZE_SLOPE_CAP = 0.60
SIZE_SLOPE_FLOOR = -0.10
MAX_CANDIDATES = 150
MAX_GROWERS = 30


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(mc_lamports):
    """log10 of market cap in SOL, as the scorer takes it."""
    return math.log10(int(mc_lamports)) - 9.0


def _norm_cdf(x, mean, sd):
    """P(X <= x) for X ~ Normal(mean, sd); infinities handled."""
    if x == float("inf"):
        return 1.0
    if x == -float("inf"):
        return 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def bin_of(x, edges):
    """Size bin 0..81 for a log10 cap in SOL (edges 2.00..6.00 step 0.05)."""
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges)
    # binary search over 81 edges
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo  # edges[lo-1] <= x < edges[lo]; bin = lo


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")
    return task, marks


def build_series(marks):
    """Per-mint time series: list of (hours_since_first, logcap, venue,
    complete, created_at).  Marks are 10 minutes apart on the grid."""
    t0 = parse_ts(marks[0]["at"])
    series = {}
    for m in marks:
        t = (parse_ts(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            mint = r["mint"]
            pts = series.get(mint)
            if pts is None:
                pts = series[mint] = {
                    "pts": [],
                    "venue": r.get("venue", "curve"),
                    "complete": bool(r.get("complete")),
                    "created_at": r.get("created_at"),
                    "name": r.get("name", ""),
                }
            pts["pts"].append((t, logcap_sol(r["market_cap_lamports"])))
            pts["venue"] = r.get("venue", pts["venue"])
            pts["complete"] = bool(r.get("complete"))
    return series, t0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def coin_stats_raw(pts, last_t):
    """Raw drift / robust 10-min volatility / median for one coin."""
    series = pts["pts"]
    last = series[-1][1]
    num = 0.0
    den = 0.0
    changes = []
    for (t1, v1), (t2, v2) in zip(series, series[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        dv = v2 - v1
        changes.append(dv)
        w = 0.5 ** ((last_t - t2) / 1.0)  # half-life 1h recency weight
        num += w * dv / dt
        den += w
    drift = num / den if den > 0 else 0.0
    n = len(changes)
    drift *= n / (n + DRIFT_SHRINK_N) if n else 0.0
    drift = max(-DRIFT_CAP, min(DRIFT_CAP, drift))
    if changes:
        med = _median(changes)
        mad = _median([abs(c - med) for c in changes])
        vol10 = 1.4826 * mad
    else:
        vol10 = 0.0
    vals = sorted(v for _, v in series)
    median = vals[len(vals) // 2]
    return last, drift, vol10, median, n


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = load_history(input_dir)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    reads = [parse_ts(r) for r in task["reads"]]
    last_mark_t = parse_ts(marks[-1]["at"])
    horizons = [max(0.0, (r - last_mark_t).total_seconds() / 3600.0)
                for r in reads]
    # A forecast is scored at every read; if a read somehow precedes the
    # last mark, give it the smallest positive horizon.
    horizons = [h if h > 0 else 0.25 for h in horizons]

    series, _t0 = build_series(marks)
    cut_t = last_mark_t

    # ---- per-coin model (two passes: raw stats, then volatility shrink) ----
    raw = []
    for mint, info in series.items():
        last, drift, vol10, median, n = coin_stats_raw(
            info, (cut_t - parse_ts(marks[0]["at"])).total_seconds() / 3600.0)
        raw.append((mint, info, last, drift, vol10, median, n))
    g_vol10 = _median([r[4] for r in raw if r[6] > 0]) if any(r[6] > 0 for r in raw) else 0.02
    coins = []
    for mint, info, last, drift, vol10, median, n in raw:
        # shrink the noisy per-coin robust vol toward the global value
        vol10 = (n * vol10 + 4.0 * g_vol10) / (n + 4.0) if n else g_vol10
        vol_h = max(VOL_FLOOR_H, min(VOL_CAP_H, vol10 * math.sqrt(6.0)))
        # age at the cut, in hours
        try:
            age_h = max(0.0, (cut_t - parse_ts(info["created_at"])).total_seconds() / 3600.0)
        except Exception:
            age_h = 1.0
        # venue adjustment to the drift: curve coins that never graduated
        # mostly fade; graduated (pool) coins persist; migrating coins are
        # about to open their pool.
        venue = info["venue"]
        if venue == "curve" and not info["complete"]:
            drift += -0.03
        elif venue == "migrating":
            drift += 0.02
        coins.append({
            "mint": mint, "last": last, "drift": drift, "vol": vol_h,
            "median": median, "age": age_h,
        })
    if not coins:
        raise ValueError("no coins in history")

    # Candidate set: top by predicted mid-horizon log-cap, plus the
    # fastest growers not already in it (breakout insurance).
    h_mid = sorted(horizons)[len(horizons) // 2]

    def mu_at(c, h):
        drift_part = c["drift"] * DRIFT_TAU_H * (1.0 - math.exp(-h / DRIFT_TAU_H))
        rev = REVERT_STRENGTH * (c["median"] - c["last"]) * (
            1.0 - math.exp(-h * math.log(2) / REVERT_HALF_LIFE_H))
        return c["last"] + drift_part + rev

    def sig_at(c, h):
        return math.sqrt(SIGMA0 * SIGMA0 + c["vol"] * c["vol"] * h)

    scored = sorted(coins, key=lambda c: (-mu_at(c, h_mid), c["mint"]))
    cand = scored[:MAX_CANDIDATES]
    cand_mints = {c["mint"] for c in cand}
    growers = sorted(
        (c for c in coins if c["mint"] not in cand_mints),
        key=lambda c: (-c["drift"], c["mint"]),
    )[:MAX_GROWERS]
    cand = cand + growers
    cand.sort(key=lambda c: (-mu_at(c, h_mid), c["mint"]))

    # ---- Monte-Carlo over the seven reads (order only) ----
    rng = random.Random(SEED)
    n_reads = len(horizons)
    place_counts = {k: {} for k in range(1, places + 1)}
    place_top5 = {}
    total_draws = N_SIMS * n_reads
    for h in horizons:
        mus = [mu_at(c, h) for c in cand]
        sigs = [sig_at(c, h) for c in cand]
        for _ in range(N_SIMS):
            draws = [rng.gauss(mus[i], sigs[i]) for i in range(len(cand))]
            order = sorted(range(len(cand)), key=lambda i: (-draws[i], cand[i]["mint"]))
            for k in range(1, places + 1):
                i = order[k - 1]
                mint = cand[i]["mint"]
                place_counts[k][mint] = place_counts[k].get(mint, 0) + 1
                place_top5[mint] = place_top5.get(mint, 0) + 1

    # ---- probabilities: frequencies scaled to leave an unnamed share ----
    probabilities = []
    for k in range(1, places + 1):
        target = PLACE_TARGET.get(k, 0.85)
        rows = sorted(place_counts[k].items(), key=lambda kv: (-kv[1], kv[0]))
        raw = [(m, cnt / total_draws) for m, cnt in rows if cnt / total_draws >= MIN_P]
        s = sum(p for _, p in raw)
        if s <= 0:
            continue
        scale = target / s
        for m, p in raw:
            q = math.floor(p * scale * 1_000_000) / 1_000_000
            if q > 0:
                probabilities.append({"mint": m, "place": k, "p": q})
    # top5: most frequent top-5 holders, ordered by mean simulated rank
    # proxy (total top-5 frequency, tie-broken by mid-horizon mu).
    mu_mid = {c["mint"]: mu_at(c, h_mid) for c in cand}
    top5 = sorted(place_top5.items(), key=lambda kv: (-kv[1], -mu_mid.get(kv[0], -99.0), kv[0]))
    top5 = [m for m, _ in top5[:places]]
    # fill from the candidate list if fewer than `places` (should not happen)
    for c in cand:
        if len(top5) >= places:
            break
        if c["mint"] not in top5:
            top5.append(c["mint"])

    # ---- sizes: place-level log-cap trend, extrapolated per read ----
    # The level of each place's cap itself trends up as new coins
    # graduate with big caps; coin-level simulations miss that
    # systematic inflow, so sizes come from a recency-weighted linear
    # fit of each place's own log-cap across the marks.
    t_first = parse_ts(marks[0]["at"])
    mark_times = [(parse_ts(m["at"]) - t_first).total_seconds() / 3600.0
                  for m in marks]
    last_t = mark_times[-1]
    sizes = []
    for k in range(1, places + 1):
        pts = []
        for ti, m in zip(mark_times, marks):
            if len(m["rows"]) >= k:
                pts.append((ti, logcap_sol(m["rows"][k - 1]["market_cap_lamports"])))
        if not pts:
            continue
        # recency-weighted linear fit (half-life 1h), slope clipped
        sw = sx = sy = sxx = sxy = 0.0
        for ti, v in pts:
            w = 0.5 ** ((last_t - ti) / 1.0)
            sw += w
            sx += w * ti
            sy += w * v
            sxx += w * ti * ti
            sxy += w * ti * v
        den = sw * sxx - sx * sx
        if den > 1e-12:
            slope = (sw * sxy - sx * sy) / den
        else:
            slope = 0.0
        slope = max(SIZE_SLOPE_FLOOR, min(SIZE_SLOPE_CAP, slope))
        level = (sy - slope * sx) / sw + slope * last_t  # fitted value at cut
        # average the Normal(center(h), sd(h)) bin mass over the reads
        bin_mass = {}
        for h in horizons:
            center = level + slope * h
            sd = SIZE_SD0 + SIZE_SD_H * math.sqrt(h)
            for b in range(len(edges) + 1):
                lo = -float("inf") if b == 0 else edges[b - 1]
                hi = float("inf") if b == len(edges) else edges[b]
                p = _norm_cdf(hi, center, sd) - _norm_cdf(lo, center, sd)
                bin_mass[b] = bin_mass.get(b, 0.0) + p / n_reads
        raw = [(b, p) for b, p in sorted(bin_mass.items()) if p >= MIN_P]
        s = sum(p for _, p in raw)
        if s <= 0:
            continue
        scale = SIZE_TARGET / s
        for b, p in raw:
            q = math.floor(p * scale * 1_000_000) / 1_000_000
            if q > 0:
                sizes.append({"place": k, "bin": b, "p": q})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task["output"]), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
