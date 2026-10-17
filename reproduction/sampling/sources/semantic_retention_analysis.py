#!/usr/bin/env python3
"""Semantic-retention comparison derived from measured serialized bytes.

This analysis distinguishes uniform window pruning from HBS. It does not claim
that HBS uses fewer bytes than a checkpoint-and-prune policy with identical
state semantics; the matched semantic policy is reported as equivalent.
"""
import argparse
import csv
import json
import os
import statistics


def load_means(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "updates": statistics.mean(float(x["client_updates_bytes"]) for x in rows),
        "checkpoint": statistics.mean(float(x["checkpoint_bytes"]) for x in rows),
        "metadata": statistics.mean(float(x["permanent_metadata_bytes"]) for x in rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measured", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--window", type=int, default=5)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = load_means(args.measured)
    rounds, window = args.rounds, min(args.window, args.rounds)

    full = rounds * (m["updates"] + m["checkpoint"] + m["metadata"])
    uniform = window * m["updates"] + m["checkpoint"] + window * m["metadata"]
    hbs = window * m["updates"] + m["checkpoint"] + rounds * m["metadata"]
    extra = hbs - uniform

    rows = [
        {
            "policy": "full_retention",
            "active_bytes": int(round(full)),
            "model_update_window_rounds": rounds,
            "permanent_evidence_rounds": rounds,
            "permanent_evidence_coverage": 1.0,
            "expired_payload_recoverable_without_archive": 1,
            "extra_active_bytes_vs_uniform": int(round(full - uniform)),
        },
        {
            "policy": "uniform_window_pruning",
            "active_bytes": int(round(uniform)),
            "model_update_window_rounds": window,
            "permanent_evidence_rounds": window,
            "permanent_evidence_coverage": window / rounds,
            "expired_payload_recoverable_without_archive": 0,
            "extra_active_bytes_vs_uniform": 0,
        },
        {
            "policy": "hbs_semantic_tiers",
            "active_bytes": int(round(hbs)),
            "model_update_window_rounds": window,
            "permanent_evidence_rounds": rounds,
            "permanent_evidence_coverage": 1.0,
            "expired_payload_recoverable_without_archive": 0,
            "extra_active_bytes_vs_uniform": int(round(extra)),
        },
        {
            "policy": "matched_semantic_checkpoint_prune",
            "active_bytes": int(round(hbs)),
            "model_update_window_rounds": window,
            "permanent_evidence_rounds": rounds,
            "permanent_evidence_coverage": 1.0,
            "expired_payload_recoverable_without_archive": 0,
            "extra_active_bytes_vs_uniform": int(round(extra)),
        },
    ]

    csv_path = os.path.join(args.out, "semantic_retention.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "rounds": rounds,
        "window": window,
        "measured_bytes": m,
        "policies": rows,
        "hbs_extra_bytes_vs_uniform": int(round(extra)),
        "hbs_extra_percent_vs_uniform": 100.0 * extra / uniform,
        "evidence_coverage_gain_percentage_points": 100.0 * (1.0 - window / rounds),
        "purpose": "Quantify the storage cost of retaining compact integrity/governance evidence after transient model-bearing states expire.",
        "interpretation": "HBS preserves evidence for every round at small metadata overhead relative to uniform pruning. A checkpoint-and-prune policy with the same semantic tiers is byte-equivalent to HBS.",
        "limitations": [
            "Derived from three measured Fashion-MNIST rounds replayed to a 50-round accounting horizon.",
            "A retained commitment supports provenance/integrity checking if a payload is later presented; it does not reconstruct an expired payload.",
            "Historical payload recovery requires the separately accounted external archive.",
        ],
    }
    with open(os.path.join(args.out, "semantic_retention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    assert rows[2]["active_bytes"] == rows[3]["active_bytes"]
    assert rows[2]["permanent_evidence_coverage"] == 1.0
    assert rows[1]["permanent_evidence_coverage"] == window / rounds
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
