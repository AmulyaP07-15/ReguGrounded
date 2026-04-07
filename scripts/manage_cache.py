#!/usr/bin/env python3
# scripts/manage_cache.py
#
# CLI tool for inspecting and managing ReguGrounded caches.
#
# Usage:
#   python scripts/manage_cache.py stats
#   python scripts/manage_cache.py clear
#   python scripts/manage_cache.py clear-expired
#   python scripts/manage_cache.py metrics

import argparse
import json
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.cache import QueryCache, ComponentCache
from utils.metrics import MetricsTracker


def cmd_stats(args) -> None:
    """Show cache statistics."""
    qcache   = QueryCache()
    ccache   = ComponentCache()
    qstats   = qcache.stats()
    cstats   = ccache.stats()

    print("\n=== Query Cache ===")
    print(f"  Memory entries : {qstats['memory_entries']}")
    print(f"  Disk entries   : {qstats['disk_entries']}")
    print(f"  Disk (expired) : {qstats['disk_expired']}")
    print(f"  TTL            : {qstats['ttl_seconds']}s ({qstats['ttl_seconds']//60} min)")

    print("\n=== Component Cache (in-memory only) ===")
    print(f"  Decomposition entries : {cstats['decomposition_entries']}")
    print(f"  Retrieval entries     : {cstats['retrieval_entries']}")
    print()


def cmd_clear(args) -> None:
    """Clear all caches."""
    qcache = QueryCache()
    ccache = ComponentCache()
    qcache.clear()
    ccache.clear()
    print("All caches cleared.")


def cmd_clear_expired(args) -> None:
    """Remove only expired entries."""
    qcache  = QueryCache()
    ccache  = ComponentCache()
    removed = qcache.clear_expired()
    evicted = ccache.evict_expired()
    print(f"Removed {removed} expired query-cache entries.")
    print(f"Evicted {evicted} expired component-cache entries.")


def cmd_metrics(args) -> None:
    """Display persisted metrics."""
    tracker = MetricsTracker()
    m = tracker.get_metrics()
    print(json.dumps(m, indent=2))


def cmd_rate_status(args) -> None:
    """Show current rate-limiter bucket levels."""
    from utils.rate_limiter import rate_limiter
    s = rate_limiter.status()
    print("\n=== Rate Limiter Status ===")
    print(f"  RPM available : {s['rpm_available']:.1f} / {s['rpm_limit']} req/min")
    print(f"  TPM available : {s['tpm_available']:,.0f} / {s['tpm_limit']:,} tok/min")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ReguGrounded cache & metrics management"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats",         help="Show cache statistics")
    sub.add_parser("clear",         help="Clear all caches")
    sub.add_parser("clear-expired", help="Remove only expired cache entries")
    sub.add_parser("metrics",       help="Display persisted metrics")
    sub.add_parser("rate-status",   help="Show rate-limiter bucket levels")

    args = parser.parse_args()

    dispatch = {
        "stats":         cmd_stats,
        "clear":         cmd_clear,
        "clear-expired": cmd_clear_expired,
        "metrics":       cmd_metrics,
        "rate-status":   cmd_rate_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
