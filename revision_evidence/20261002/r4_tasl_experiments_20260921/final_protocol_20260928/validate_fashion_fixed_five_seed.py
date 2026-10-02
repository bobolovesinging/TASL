"""Validate five-seed fixed-participation Fashion-MNIST TAS CSVs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SEEDS = (42, 123, 2024, 3407, 7711)
ATTACKS = ("lf", "scaling")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_case(root, attack, seed):
    config = root / "configs" / f"fashion_{attack}_seed{seed}.yaml"
    body = config.read_text(encoding="utf-8")
    for value in (f"seed: {seed}", "dataset: fashionmnist", "exclusion: false",
                  "clients: 10", "byzantine_ratio: 0.4", "rounds: 50",
                  "  - noexecutor", "  - trust_v1v2"):
        assert value in body, (config, value)
    assert body.count("  - ") == 2, config
    directory = root / "results" / f"{attack}_seed{seed}"
    raw = directory / "tasv2_mixed12_r50_b4.csv"
    summary = directory / "tasv2_summary.csv"
    rows = csv_rows(raw)
    summaries = csv_rows(summary)
    assert len(rows) == 100 and len(summaries) == 2
    groups = {name: {} for name in ("noexecutor", "trust_v1v2")}
    for row in rows:
        group = row["group"]
        number = int(row["round"])
        assert group in groups and number not in groups[group]
        groups[group][number] = row
    for group, rounds in groups.items():
        assert sorted(rounds) == list(range(1, 51)), (attack, seed, group)
        for number, row in rounds.items():
            assert int(row["skipped"]) == 0
            assert json.loads(row["excluded_clients"]) == []
            assert int(row["proposal_binding_valid"]) == 1
            if number > 1:
                assert row["parent_hash"] == rounds[number - 1]["state_hash"]
    full, reference = groups["trust_v1v2"], groups["noexecutor"]
    attacks = [row for row in full.values() if int(row["attack_active"])]
    honest = [row for row in full.values() if not int(row["attack_active"])]
    assert len(attacks) == 30 and len(honest) == 20
    overlap = dict.fromkeys(("both", "vs_only", "vp_only", "neither"), 0)
    kinds = {}
    for row in attacks:
        kind = row["attack_type"]
        kinds[kind] = kinds.get(kind, 0) + 1
        vs = bool(int(row["v1_trust"]) or int(row["v1_agg"]))
        vp = bool(int(row["v2_trust"]) or int(row["v2_agg"]))
        overlap["both" if vs and vp else "vs_only" if vs else
                "vp_only" if vp else "neither"] += 1
        assert int(row["consensus_passed"]) == 0
        for field in ("recovery_attempted", "recovery_consistent",
                      "recovery_endorsed", "recovery_v1_endorsed",
                      "recovery_v2_endorsed"):
            assert int(row[field]) == 1, (attack, seed, row["round"], field)
    assert sorted(kinds.values()) == [10, 20] and overlap["neither"] == 0
    assert all(int(row["consensus_passed"]) == 1 and
               int(row["honest_rejection"]) == 0 for row in honest)
    matches = sum(full[i]["state_hash"] == reference[i]["state_hash"] and
                  full[i]["accuracy"] == reference[i]["accuracy"]
                  for i in range(1, 51))
    assert matches == 50, (attack, seed, "state_accuracy")
    reported = {row["group"]: row for row in summaries}
    assert set(reported) == set(groups)
    for group, rounds in groups.items():
        average = statistics.mean(float(row["accuracy"]) for row in rounds.values())
        assert abs(average - float(reported[group]["avg_acc"])) < 0.001
    assert reported["trust_v1v2"]["avg_acc"] == reported["noexecutor"]["avg_acc"]
    assert int(reported["trust_v1v2"]["blocked_rounds"]) == 30
    return {"attack": attack, "seed": seed,
            "avg_accuracy_percent": float(reported["trust_v1v2"]["avg_acc"]),
            "blocked": len(attacks), "honest_accepted": len(honest),
            "recovered_with_dual_endorsement": len(attacks),
            "paired_state_accuracy_matches": matches, "overlap": overlap,
            "config_sha256": digest(config), "round_csv_sha256": digest(raw),
            "summary_csv_sha256": digest(summary)}


def main(root):
    cases = [validate_case(root, attack, seed)
             for attack in ATTACKS for seed in SEEDS]
    pooled = {}
    for attack in ATTACKS:
        subset = [row for row in cases if row["attack"] == attack]
        values = [row["avg_accuracy_percent"] for row in subset]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        half = 2.7764451051977987 * sd / math.sqrt(5)
        pooled[attack] = {
            "seeds": list(SEEDS), "avg_accuracy_mean_percent": mean,
            "avg_accuracy_sample_sd_percent": sd,
            "avg_accuracy_mean_95pct_t_interval": [mean - half, mean + half],
            "blocked": sum(row["blocked"] for row in subset),
            "honest_accepted": sum(row["honest_accepted"] for row in subset),
            "recovered_with_dual_endorsement": sum(
                row["recovered_with_dual_endorsement"] for row in subset),
            "paired_state_accuracy_matches": sum(
                row["paired_state_accuracy_matches"] for row in subset),
            "overlap": {key: sum(row["overlap"][key] for row in subset)
                        for key in ("both", "vs_only", "vp_only", "neither")}}
    print(json.dumps({"validation": "passed", "cases": cases, "pooled": pooled},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
