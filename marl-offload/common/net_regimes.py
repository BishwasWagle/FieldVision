import numpy as np
from common.config import AGENT_CONGESTION, CONGESTION_MULTS

# Link types: drone_drone | drone_edge (edge/GPU) | drone_cloud
REGIMES = {
    "drone_drone": {
        "good":     ((40,80),   (10,20),   (0.00,0.01)),
        "nominal":  ((20,40),   (20,40),   (0.00,0.02)),
        "degraded": (( 5,20),   (40,80),   (0.01,0.05)),
        "outage":   (( 0, 2),   (150,500), (0.05,0.20)),
    },
    "drone_edge": {
        "good":     ((120,200), ( 5,10),   (0.00,0.005)),
        "nominal":  ((60,120),  (10,25),   (0.00,0.01)),
        "degraded": ((15,60),   (25,70),   (0.01,0.03)),
        "outage":   (( 0, 5),   (150,300), (0.03,0.10)),
    },
    "drone_cloud": {
        "good":     ((120,250), (25,45),   (0.00,0.005)),
        "nominal":  ((60,150),  (45,90),   (0.00,0.01)),
        "degraded": ((10,60),   (90,180),  (0.01,0.03)),
        "outage":   (( 0, 5),   (200,500), (0.03,0.10)),
    },
}

def _base_sample(link_key: str, regime: str):
    (bw_lo, bw_hi), (rtt_lo, rtt_hi), (p_lo, p_hi) = REGIMES[link_key][regime]
    bw  = np.random.uniform(bw_lo, bw_hi)     # Mbps
    rtt = np.random.uniform(rtt_lo, rtt_hi)   # ms
    loss= np.random.uniform(p_lo, p_hi)       # probability
    return bw, rtt, loss

def sample_link_for_agent(agent: str, link_key: str, regime: str):
    bw, rtt, loss = _base_sample(link_key, regime)
    level = AGENT_CONGESTION.get(agent, "light")
    mul   = CONGESTION_MULTS[level]
    bw  *= mul["bw_mult"]
    rtt *= mul["rtt_mult"]
    loss*= mul["loss_mult"]
    return max(bw, 0.01), rtt, min(loss, 0.99)

def transfer_time_seconds(size_mb, bandwidth_mbps, latency_ms):
    # Convert Mbps -> MB/s by dividing by 8
    bw_MBps = max(1e-6, bandwidth_mbps) / 8.0
    return (latency_ms / 1000.0) + (size_mb / bw_MBps)
