"""Reproducible local throughput test on distinct, generated log records."""

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

from jselect import Index, select


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=int, default=100000)
    args = p.parse_args()
    messages = [
        "Database connections time out while waiting for a connection pool slot.",
        "The verification email never arrives and signup cannot finish.",
        "Please move the meeting from Tuesday to Friday.",
        "Credit card required to register a free account.",
    ]
    data = ({"id": str(i), "text": f"Ticket {i}: {messages[i % len(messages)]}"} for i in range(args.records))
    started = time.perf_counter()
    with Index.build(data) as index:
        build_seconds = time.perf_counter() - started
        bytes_on_disk = index.path.stat().st_size
        runs = [select(index, task="why does signup fail?", tokens=1500, mode="local") for _ in range(3)]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024
    report = {
        "kind": "generated short log records; unique IDs in text",
        "records": args.records,
        "python": platform.python_version(),
        "system": platform.platform(),
        "build_seconds": build_seconds,
        "index_bytes": bytes_on_disk,
        "median_query_seconds": statistics.median(r.stats["seconds"] for r in runs),
        "peak_rss_bytes": peak_rss,
        "cost": 0,
        "runs": [r.stats for r in runs],
    }
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    (out / f"scale-{args.records}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
