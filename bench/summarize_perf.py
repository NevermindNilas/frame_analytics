"""Compact table from bench_perf_metrics JSONL files: metric x shape x mode."""
import json
import sys

rows = {}
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        res = r["res"]
        key = (r["metric"], r["device"], "x".join(map(str, r["shape"])), r["mode"])
        if "error" in res:
            rows[key] = "ERR " + res["error"][:60]
            continue
        parts = []
        for arm, v in res.items():
            parts.append(f"{arm}={v['min_ms']:.3f}" + (f"({v['peak_mib']:.0f}MiB)" if v["peak_mib"] is not None else ""))
        if "new" in res and "ref" in res:
            parts.append(f"ratio={res['new']['min_ms'] / res['ref']['min_ms']:.3f}")
        rows[key] = " ".join(parts)
for k in sorted(rows):
    print(f"{k[0]:13s} {k[1]:4s} {k[2]:14s} {k[3]:3s} {rows[k]}")
