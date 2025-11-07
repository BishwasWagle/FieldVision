# trainer/plotter.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.config import LOG_CSV, FIG_DIR
from common.utils import ensure_dirs

def _safe_read_log():
    if not os.path.exists(LOG_CSV) or os.path.getsize(LOG_CSV) == 0:
        return None
    try:
        df = pd.read_csv(LOG_CSV)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def make_plots():
    ensure_dirs(FIG_DIR)
    df = _safe_read_log()
    if df is None:
        # nothing to plot yet
        return

    # Ensure expected columns exist (create defaults if missing)
    for col, default in [
        ("reward", 0.0),
        ("latency_ms", np.nan),
        ("regime", "unknown"),
        ("priority", "medium"),
        ("succ", 0.0),
        ("step", np.arange(len(df))),
    ]:
        if col not in df.columns:
            df[col] = default

    # 1) Learning-ish: rolling reward
    try:
        df["r_roll"] = df["reward"].rolling(500, min_periods=1).mean()
        plt.figure()
        plt.plot(df["r_roll"].values)
        plt.title("Rolling Reward (window=500)")
        plt.xlabel("row index")
        plt.ylabel("reward")
        plt.grid(True)
        plt.savefig(f"{FIG_DIR}/reward_curve.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # 2) Latency CDF
    try:
        s = df["latency_ms"].dropna().astype(float)
        if len(s) > 0:
            s = s.sort_values().values
            y = (np.arange(1, len(s) + 1)) / float(len(s))  # <-- fixed line
            plt.figure()
            plt.plot(s, y)
            plt.title("Latency CDF")
            plt.xlabel("Latency (ms)")
            plt.ylabel("F(x)")
            plt.grid(True)
            plt.savefig(f"{FIG_DIR}/latency_cdf.png", dpi=150)
            plt.close()
    except Exception:
        pass

    # 3) Avg latency by regime
    try:
        if "regime" in df.columns and "latency_ms" in df.columns:
            b = df.dropna(subset=["latency_ms"]).groupby("regime")["latency_ms"].mean().sort_values()
            if len(b) > 0:
                plt.figure()
                b.plot(kind="bar")
                plt.title("Avg Latency by Regime")
                plt.ylabel("Latency (ms)")
                plt.grid(True, axis="y")
                plt.savefig(f"{FIG_DIR}/regime_latency_bar.png", dpi=150)
                plt.close()
    except Exception:
        pass

    # 4) SLO success by priority
    try:
        if "succ" in df.columns and "priority" in df.columns:
            slo = df.groupby("priority")["succ"].mean()
            if len(slo) > 0:
                slo = slo.reindex(["high", "medium", "low"])
                plt.figure()
                slo.plot(kind="bar")
                plt.title("SLO Success by Priority")
                plt.ylabel("Rate")
                plt.ylim(0, 1)
                plt.grid(True, axis="y")
                plt.savefig(f"{FIG_DIR}/slo_success_priority.png", dpi=150)
                plt.close()
    except Exception:
        pass
