import math, numpy as np
from common.net_regimes import sample_link, transfer_time_seconds
from common.config import COMPUTE_MS

def _compute_ms_for_profile(profile: str, action: int):
    if action == 0:
        key = "local_low" if profile=="low" else "local_high"
    elif action == 1:
        key = "edge"
    else:
        key = "cloud"
    lo, hi = COMPUTE_MS[key]
    return (lo+hi)/2.0

def _predict_e2e_ms(payload_mb, profile, regime):
    """Return (ms_local, ms_edge, ms_cloud) using mean regime stats."""
    # crude means
    link_keys = [None, "drone_edge", "drone_cloud"]
    out = []
    for a in [0,1,2]:
        if a==0:
            xfer_ms = 0.0
        else:
            bw, rtt, loss = sample_link(link_keys[a], regime)
            xfer_ms = (transfer_time_seconds(payload_mb, bw, rtt)*1000.0) * (1.5 if loss>0.02 else 1.0)
        comp_ms = _compute_ms_for_profile(profile, a)
        out.append(comp_ms + xfer_ms)
    return tuple(out)

def mmde_choose_action(payload_mb, profile, regime, priority_weight, deadline_ms):
    ms_local, ms_edge, ms_cloud = _predict_e2e_ms(payload_mb, profile, regime)
    def U(ms):
        U_L = math.exp(-ms/500.0)
        U_S = 1.0 if ms <= deadline_ms else math.exp(-(ms-deadline_ms)/400.0)
        return 0.55*U_L + 0.35*U_S + 0.10*priority_weight
    scores = [U(ms_local), U(ms_edge), U(ms_cloud)]
    return int(np.argmax(scores))
