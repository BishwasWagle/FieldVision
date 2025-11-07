#!/usr/bin/env bash
# Usage:
#  sudo ./network/apply_netem_link.sh eth0 10.0.1.10/32 good drone_edge
# Regimes: good|nominal|degraded|outage
# Link types: drone_drone|drone_edge|drone_cloud (for your own labeling)

IFACE="$1"
DST="$2"         # e.g., Trainer IP / Edge IP / Cloud IP in CIDR
REGIME="$3"      # good|nominal|degraded|outage
LINK="$4"        # label only

# Pick params (ms delay, ms jitter, % loss, rate)
case "$REGIME" in
  good)     DELAY=10;  JITTER=3;  LOSS=0.1; RATE="100mbit" ;;
  nominal)  DELAY=30;  JITTER=10; LOSS=0.5; RATE="60mbit"  ;;
  degraded) DELAY=90;  JITTER=30; LOSS=2.0; RATE="20mbit"  ;;
  outage)   DELAY=250; JITTER=80; LOSS=5.0; RATE="2mbit"   ;;
  *) echo "unknown regime"; exit 1;;
esac

# Create a prio qdisc and attach netem on a class filtered by dst
sudo tc qdisc add dev "$IFACE" root handle 1: prio || true
sudo tc qdisc add dev "$IFACE" parent 1:3 handle 30: netem delay ${DELAY}ms ${JITTER}ms loss ${LOSS}% || true
sudo tc filter add dev "$IFACE" protocol ip parent 1:0 prio 3 u32 match ip dst ${DST} flowid 1:3 || true
sudo tc qdisc add dev "$IFACE" parent 30:1 handle 40: tbf rate ${RATE} burst 32kbit latency 400ms || true

echo "[netem] ${LINK} -> ${DST} set to ${REGIME} (${DELAY}ms ±${JITTER}ms, loss ${LOSS}%, rate ${RATE})"
