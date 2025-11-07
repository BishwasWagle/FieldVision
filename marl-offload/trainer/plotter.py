import os, pandas as pd, matplotlib.pyplot as plt
from common.config import LOG_CSV, FIG_DIR
from common.utils import ensure_dirs

def make_plots():
    ensure_dirs(FIG_DIR)
    df = pd.read_csv(LOG_CSV)
    # Learning-ish: reward over time (rolling mean)
    df["r_roll"] = df["reward"].rolling(500,min_periods=1).mean()
    plt.figure(); plt.plot(df["r_roll"]); plt.title("Rolling Reward"); plt.xlabel("step"); plt.ylabel("reward"); plt.grid(True)
    plt.savefig(f"{FIG_DIR}/reward_curve.png", dpi=150); plt.close()

    # Latency CDF
    s = df["latency_ms"].dropna().sort_values()
    y = (1+range(len(s)))/len(s)
    plt.figure(); plt.plot(s.values, y); plt.title("Latency CDF"); plt.xlabel("ms"); plt.ylabel("F(x)"); plt.grid(True)
    plt.savefig(f"{FIG_DIR}/latency_cdf.png", dpi=150); plt.close()

    # By regime bars
    b = df.groupby("regime")["latency_ms"].mean().sort_values()
    plt.figure(); b.plot(kind="bar"); plt.title("Avg Latency by Regime"); plt.ylabel("ms"); plt.grid(True, axis="y")
    plt.savefig(f"{FIG_DIR}/regime_latency_bar.png", dpi=150); plt.close()

    # SLO success by priority
    slo = df.groupby("priority")["succ"].mean().reindex(["high","medium","low"])
    plt.figure(); slo.plot(kind="bar"); plt.title("SLO Success by Priority"); plt.ylabel("rate"); plt.ylim(0,1); plt.grid(True, axis="y")
    plt.savefig(f"{FIG_DIR}/slo_success_priority.png", dpi=150); plt.close()
