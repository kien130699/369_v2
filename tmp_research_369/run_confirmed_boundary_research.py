from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

DATA = Path("XAUUSD_M1_2025_2026.csv")
OUT = Path("tmp_research_369/results")
OUT.mkdir(parents=True, exist_ok=True)
SPREAD = 0.30
PRICE_SCALE = 1000
STEP_T = 10000
UPPER_T = 6600
LOWER_T = 4400
LOWER_TARGET_OFFSET_T = 100
SLS = np.array([4.4, 6.6, 9.0, 13.0, 15.0], dtype=np.float64)
CONFIGS = [(float(sl), mode) for sl in SLS for mode in ("BOUNDARY", "RR1", "RR2")]
TRAIN_END = pd.Timestamp("2026-04-01T00:00:00Z")
VALID_END = pd.Timestamp("2026-07-01T00:00:00Z")


def ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def load_data() -> pd.DataFrame:
    x = pd.read_csv(DATA)
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    if x["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps")
    x = x.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        x[c] = pd.to_numeric(x[c], errors="raise")
    bad = (
        (x.high < x[["open", "close"]].max(axis=1))
        | (x.low > x[["open", "close"]].min(axis=1))
        | (x.high < x.low)
    )
    if bad.any():
        raise ValueError("Invalid OHLC")
    return x


def build_boundaries(x: pd.DataFrame) -> pd.DataFrame:
    ts = x.timestamp.to_numpy()
    op = np.rint(x.open.to_numpy(float) * PRICE_SCALE).astype(np.int64)
    hi = np.rint(x.high.to_numpy(float) * PRICE_SCALE).astype(np.int64)
    lo = np.rint(x.low.to_numpy(float) * PRICE_SCALE).astype(np.int64)
    cl = np.rint(x.close.to_numpy(float) * PRICE_SCALE).astype(np.int64)

    rec_dir: list[int] = []
    rec_open: list[int] = []
    rec_target: list[int] = []
    rec_bar: list[int] = []
    rec_dual: list[int] = []
    rec_samebar: list[int] = []
    rec_active: list[bool] = []
    open_by: dict[tuple[int, int], int] = {}
    up_heap: list[tuple[int, int]] = []
    down_heap: list[tuple[int, int]] = []
    rows: list[dict[str, object]] = []

    for i in range(1, len(x)):
        closed_this: set[tuple[int, int]] = set()

        while up_heap and up_heap[0][0] <= hi[i]:
            _, rid = heapq.heappop(up_heap)
            if rec_active[rid] and rec_bar[rid] < i:
                rec_active[rid] = False
                key = (1, rec_open[rid])
                open_by.pop(key, None)
                closed_this.add(key)

        while down_heap:
            target = -down_heap[0][0]
            if lo[i] > target:
                break
            _, rid = heapq.heappop(down_heap)
            if rec_active[rid] and rec_bar[rid] < i:
                rec_active[rid] = False
                key = (-1, rec_open[rid])
                open_by.pop(key, None)
                closed_this.add(key)

        pc = int(cl[i - 1])
        upper_ns: list[int] = []
        lower_ns: list[int] = []
        if hi[i] > pc:
            first_n = (pc - UPPER_T) // STEP_T + 1
            last_n = (int(hi[i]) - UPPER_T) // STEP_T
            if last_n >= first_n:
                upper_ns = list(range(first_n, last_n + 1))
        if lo[i] < pc:
            first_n = ceil_div(int(lo[i]) - LOWER_T, STEP_T)
            last_n = ceil_div(pc - LOWER_T, STEP_T) - 1
            if first_n <= last_n:
                lower_ns = list(range(last_n, first_n - 1, -1))

        upper_set, lower_set = set(upper_ns), set(lower_ns)
        skip: set[int] = set()
        dual: set[int] = set()
        for n in upper_set & lower_set:
            up_target = (n + 1) * STEP_T
            down_target = n * STEP_T - LOWER_TARGET_OFFSET_T
            if hi[i] >= up_target or lo[i] <= down_target:
                skip.add(n)
            else:
                dual.add(n)

        def open_record(direction: int, level: int, target: int, is_dual: bool) -> None:
            key = (direction, level)
            if key in open_by or key in closed_this:
                return
            rid = len(rec_dir)
            rec_dir.append(direction)
            rec_open.append(level)
            rec_target.append(target)
            rec_bar.append(i)
            rec_dual.append(int(is_dual))
            rec_samebar.append(0)
            rec_active.append(True)
            open_by[key] = rid
            if direction == 1:
                heapq.heappush(up_heap, (target, rid))
            else:
                heapq.heappush(down_heap, (-target, rid))

            confirmed = cl[i] >= level if direction == 1 else cl[i] <= level
            rows.append(
                {
                    "boundary_id": rid + 1,
                    "signal_bar_index": i,
                    "signal_time": pd.Timestamp(ts[i]),
                    "direction": "UP" if direction == 1 else "DOWN",
                    "sign": direction,
                    "open_level": level / PRICE_SCALE,
                    "target_level": target / PRICE_SCALE,
                    "dual_bar_open": int(is_dual),
                    "confirmed_close": int(confirmed),
                    "signal_open": op[i] / PRICE_SCALE,
                    "signal_high": hi[i] / PRICE_SCALE,
                    "signal_low": lo[i] / PRICE_SCALE,
                    "signal_close": cl[i] / PRICE_SCALE,
                    "record_index": rid,
                }
            )

            target_hit = hi[i] >= target if direction == 1 else lo[i] <= target
            if target_hit:
                rec_samebar[rid] = 1
                rec_active[rid] = False
                open_by.pop(key, None)
                closed_this.add(key)

        for n in upper_ns:
            if n not in skip:
                open_record(1, n * STEP_T + UPPER_T, (n + 1) * STEP_T, n in dual)
        for n in lower_ns:
            if n not in skip:
                open_record(-1, n * STEP_T + LOWER_T, n * STEP_T - LOWER_TARGET_OFFSET_T, n in dual)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["same_bar_tp"] = [rec_samebar[int(i)] for i in result.record_index]
    return result.drop(columns="record_index")


def prepare_signals(x: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stages: list[dict[str, object]] = []

    def snap(name: str, q: pd.DataFrame) -> None:
        stages.append(
            {
                "stage": name,
                "count": len(q),
                "UP": int((q.direction == "UP").sum()),
                "DOWN": int((q.direction == "DOWN").sum()),
                "DUAL": int((q.dual_bar_open == 1).sum()),
                "SINGLE": int((q.dual_bar_open == 0).sum()),
            }
        )

    q = b.copy()
    snap("01_TOTAL_BOUNDARIES", q)
    q = q[q.confirmed_close == 1].copy()
    snap("02_CLOSE_CONFIRMED", q)
    q = q[q.same_bar_tp == 0].copy()
    snap("03_EXCLUDE_SAME_BAR_TP", q)
    q["entry_bar_index"] = q.signal_bar_index.astype(int) + 1
    q = q[q.entry_bar_index < len(x)].copy()
    snap("04_HAS_NEXT_BAR", q)
    idx = q.entry_bar_index.to_numpy(int)
    q["entry_time"] = x.timestamp.iloc[idx].to_numpy()
    q["entry_price"] = x.open.iloc[idx].to_numpy(float)
    q["boundary_target_distance"] = (
        (q.target_level - q.entry_price) * q.sign
    )
    q = q[q.boundary_target_distance > 0].copy()
    snap("05_TARGET_AHEAD_AT_ENTRY", q)
    q["side"] = np.where(q.sign == 1, "BUY", "SELL")
    q["dual_class"] = np.where(q.dual_bar_open == 1, "DUAL_BAR", "SINGLE_DIRECTION")
    q["entry_hour_utc"] = q.entry_time.dt.hour
    q["entry_hour_vn"] = q.entry_time.dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour
    q["session_utc"] = pd.cut(
        q.entry_hour_utc,
        bins=[-1, 5, 12, 20, 23],
        labels=["ASIA_00_06", "LONDON_06_13", "NEWYORK_13_21", "OFF_21_24"],
    ).astype(str)
    q["split"] = np.where(
        q.entry_time < TRAIN_END,
        "TRAIN",
        np.where(q.entry_time < VALID_END, "VALIDATION", "JULY_HOLDOUT"),
    )
    q["entry_month"] = q.entry_time.dt.strftime("%Y-%m")
    return q.reset_index(drop=True), pd.DataFrame(stages)


@njit(cache=True)
def simulate(hi, lo, cl, entry_idx, signs, entries, boundary_dist, spread):
    n_signal = len(entry_idx)
    n_config = 15
    n_out = n_signal * n_config
    outcome = np.empty(n_out, np.int8)  # 1 TP, -1 SL, 0 EOD
    exit_idx = np.empty(n_out, np.int64)
    r_net = np.empty(n_out, np.float64)
    gross = np.empty(n_out, np.float64)
    mfe = np.empty(n_out, np.float64)
    mae = np.empty(n_out, np.float64)
    same_bar = np.empty(n_out, np.int8)
    p = 0
    for s in range(n_signal):
        eidx = entry_idx[s]
        sign = signs[s]
        entry = entries[s]
        for si in range(5):
            sl = SLS[si]
            for mode in range(3):
                target = boundary_dist[s] if mode == 0 else sl if mode == 1 else 2.0 * sl
                best_f = 0.0
                best_a = 0.0
                out = 0
                ex = len(cl) - 1
                sb = 0
                for j in range(eidx, len(cl)):
                    if sign == 1:
                        fav = hi[j] - entry
                        adv = entry - lo[j]
                    else:
                        fav = entry - lo[j]
                        adv = hi[j] - entry
                    if fav > best_f:
                        best_f = fav
                    if adv > best_a:
                        best_a = adv
                    hit_sl = adv >= sl
                    hit_tp = fav >= target
                    if hit_sl:
                        out = -1
                        ex = j
                        sb = 1 if hit_tp else 0
                        break
                    if hit_tp:
                        out = 1
                        ex = j
                        break
                if out == -1:
                    gp = -sl
                elif out == 1:
                    gp = target
                else:
                    gp = sign * (cl[-1] - entry)
                outcome[p] = out
                exit_idx[p] = ex
                gross[p] = gp
                r_net[p] = (gp - spread) / (sl + spread)
                mfe[p] = best_f / (sl + spread)
                mae[p] = best_a / (sl + spread)
                same_bar[p] = sb
                p += 1
    return outcome, exit_idx, r_net, gross, mfe, mae, same_bar


def make_trades(x: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    hi = x.high.to_numpy(float)
    lo = x.low.to_numpy(float)
    cl = x.close.to_numpy(float)
    out, ex, r, gross, mfe, mae, samebar = simulate(
        hi,
        lo,
        cl,
        signals.entry_bar_index.to_numpy(np.int64),
        signals.sign.to_numpy(np.int8),
        signals.entry_price.to_numpy(float),
        signals.boundary_target_distance.to_numpy(float),
        SPREAD,
    )
    base = signals.loc[signals.index.repeat(15)].reset_index(drop=True)
    config = [f"SL{sl:g}_{mode}" for sl, mode in CONFIGS]
    base["config"] = np.tile(config, len(signals))
    base["sl_distance"] = np.tile([c[0] for c in CONFIGS], len(signals))
    base["tp_mode"] = np.tile([c[1] for c in CONFIGS], len(signals))
    base["outcome"] = np.where(out == 1, "TP", np.where(out == -1, "SL", "EOD"))
    base["exit_bar_index"] = ex
    base["exit_time"] = x.timestamp.iloc[ex].to_numpy()
    base["R"] = r
    base["gross_points"] = gross
    base["MFE_R"] = mfe
    base["MAE_R"] = mae
    base["same_bar_sl_tp"] = samebar
    base["bars_held"] = ex - base.entry_bar_index.to_numpy(int)
    return base


def max_dd(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    eq = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
    return float(np.max(peak - eq))


def metric(g: pd.DataFrame) -> dict[str, float | int]:
    g = g.sort_values(["exit_time", "entry_time", "boundary_id"], kind="stable")
    r = g.R.to_numpy(float)
    gain = r[r > 0].sum()
    loss = -r[r < 0].sum()
    monthly = g.groupby("entry_month").R.sum()
    return {
        "trades": len(g),
        "WR": float((r > 0).mean()),
        "TP_rate": float((g.outcome == "TP").mean()),
        "SL_rate": float((g.outcome == "SL").mean()),
        "EOD_rate": float((g.outcome == "EOD").mean()),
        "AvgR": float(r.mean()),
        "TotalR": float(r.sum()),
        "PF": float(gain / loss) if loss > 0 else math.inf,
        "MaxDD_R": max_dd(r),
        "MedianR": float(np.median(r)),
        "Avg_MAE_R": float(g.MAE_R.mean()),
        "P90_MAE_R": float(g.MAE_R.quantile(0.9)),
        "Avg_MFE_R": float(g.MFE_R.mean()),
        "P90_MFE_R": float(g.MFE_R.quantile(0.9)),
        "TradesPerActiveMonth": float(len(g) / max(1, len(monthly))),
        "PositiveMonthRate": float((monthly > 0).mean()),
    }


def summary(frame: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    rows = []
    keys = cols[0] if len(cols) == 1 else cols
    for key, g in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(cols, key))
        row["portfolio_mode"] = mode
        row.update(metric(g))
        rows.append(row)
    return pd.DataFrame(rows)


def campaigns(trades: pd.DataFrame) -> pd.DataFrame:
    accepted = []
    for config, g in trades.groupby("config", sort=True):
        last_exit = -1
        for _, row in g.sort_values(["entry_bar_index", "boundary_id"], kind="stable").iterrows():
            if int(row.entry_bar_index) <= last_exit:
                continue
            accepted.append(row)
            last_exit = int(row.exit_bar_index)
    return pd.DataFrame(accepted).reset_index(drop=True)


def main() -> None:
    x = load_data()
    boundaries = build_boundaries(x)
    signals, funnel = prepare_signals(x, boundaries)
    trades = make_trades(x, signals)
    campaign = campaigns(trades)

    outputs = {
        "00_data_audit.csv": pd.DataFrame(
            [
                {"metric": "bars", "value": len(x)},
                {"metric": "first_utc", "value": x.timestamp.iloc[0]},
                {"metric": "last_utc", "value": x.timestamp.iloc[-1]},
                {"metric": "boundaries", "value": len(boundaries)},
                {"metric": "eligible_signals", "value": len(signals)},
                {"metric": "spread", "value": SPREAD},
                {"metric": "intrabar", "value": "SL_FIRST"},
            ]
        ),
        "01_signal_funnel.csv": funnel,
        "02_config_independent.csv": summary(trades, ["config", "sl_distance", "tp_mode"], "INDEPENDENT"),
        "03_config_campaign.csv": summary(campaign, ["config", "sl_distance", "tp_mode"], "ONE_POSITION"),
        "04_split_independent.csv": summary(trades, ["config", "split"], "INDEPENDENT"),
        "05_split_campaign.csv": summary(campaign, ["config", "split"], "ONE_POSITION"),
        "06_direction.csv": summary(trades, ["config", "direction", "split"], "INDEPENDENT"),
        "07_session.csv": summary(trades, ["config", "session_utc", "split"], "INDEPENDENT"),
        "08_hour.csv": summary(trades, ["config", "entry_hour_utc", "split"], "INDEPENDENT"),
        "09_dual.csv": summary(trades, ["config", "dual_class", "split"], "INDEPENDENT"),
        "10_monthly.csv": summary(trades, ["config", "entry_month"], "INDEPENDENT"),
        "11_signals.csv": signals,
        "12_all_trades_independent.csv": trades,
        "13_all_trades_campaign.csv": campaign,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    split = outputs["05_split_campaign.csv"]
    rows = []
    for config, g in split.groupby("config"):
        by = {r.split: r for r in g.itertuples(index=False)}
        tr = by.get("TRAIN")
        va = by.get("VALIDATION")
        ju = by.get("JULY_HOLDOUT")
        train_avg = getattr(tr, "AvgR", np.nan)
        valid_avg = getattr(va, "AvgR", np.nan)
        rows.append(
            {
                "config": config,
                "eligible_pre_holdout": int(
                    tr is not None
                    and va is not None
                    and tr.trades >= 200
                    and va.trades >= 100
                    and train_avg > 0
                    and valid_avg > 0
                    and tr.PF > 1
                    and va.PF > 1
                ),
                "pre_holdout_score": min(train_avg, valid_avg),
                "train_n": getattr(tr, "trades", np.nan),
                "train_WR": getattr(tr, "WR", np.nan),
                "train_AvgR": train_avg,
                "train_PF": getattr(tr, "PF", np.nan),
                "train_DD": getattr(tr, "MaxDD_R", np.nan),
                "valid_n": getattr(va, "trades", np.nan),
                "valid_WR": getattr(va, "WR", np.nan),
                "valid_AvgR": valid_avg,
                "valid_PF": getattr(va, "PF", np.nan),
                "valid_DD": getattr(va, "MaxDD_R", np.nan),
                "july_n": getattr(ju, "trades", np.nan),
                "july_WR": getattr(ju, "WR", np.nan),
                "july_AvgR": getattr(ju, "AvgR", np.nan),
                "july_PF": getattr(ju, "PF", np.nan),
                "july_DD": getattr(ju, "MaxDD_R", np.nan),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["eligible_pre_holdout", "pre_holdout_score"], ascending=[False, False]
    )
    ranked.to_csv(OUT / "14_ranked_pre_holdout.csv", index=False, encoding="utf-8-sig")

    top = ranked.head(15)
    report = [
        "# Confirmed boundary next-bar research",
        "",
        f"- Bars: {len(x):,}",
        f"- Range: {x.timestamp.iloc[0]} -> {x.timestamp.iloc[-1]}",
        f"- Boundaries: {len(boundaries):,}",
        f"- Eligible signals: {len(signals):,}",
        "- Entry: next M1 open after close confirmation",
        "- Same-bar SL/TP: SL first",
        "- Spread: fixed 0.30 (not real bid/ask)",
        "- Selection uses TRAIN + VALIDATION only",
        "",
        "## Top campaign configurations",
        "",
        top.to_markdown(index=False),
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
