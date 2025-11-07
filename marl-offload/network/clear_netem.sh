#!/usr/bin/env bash
# Usage: sudo ./network/clear_netem.sh eth0
IFACE="$1"
sudo tc qdisc del dev "$IFACE" root || true
echo "[netem] cleared on $IFACE"
