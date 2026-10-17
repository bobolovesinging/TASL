"""Paired Fashion-MNIST TAS experiments; results remain separate from old Table 5."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts" / "exp3" / "exp3_governance_tasv2.py"
spec = importlib.util.spec_from_file_location("final_protocol", SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_args(config, rounds, device, smoke_samples=0):
    exp, tas, executor = config["experiment"], config["tas"], config["executor_attack"]
    m.set_seed(exp["seed"])
    loaders, test = m.get_data_loaders(
        client_num=exp["clients"], batch_size=exp["batch_size"],
        alpha=config["data"]["alpha"], seed=exp["seed"],
        dataset=config["data"]["dataset"])
    if smoke_samples:
        from torch.utils.data import DataLoader, Subset
        loaders = [DataLoader(Subset(loader.dataset,
                                     range(min(smoke_samples, len(loader.dataset)))),
                              batch_size=min(16, smoke_samples), shuffle=True)
                   for loader in loaders]
        test = DataLoader(Subset(test.dataset, range(min(64, len(test.dataset)))),
                          batch_size=32, shuffle=False)
    return dict(
        client_loaders=loaders, test_loader=test, rounds=rounds,
        chaos_schedule=m.make_chaos_schedule(
            rounds, seed=executor["chaos_seed"], executor_attack=executor["type"],
            attacks_per_block=executor["attacks_per_block"]),
        device=device, seed=exp["seed"], local_epochs=exp["local_epochs"],
        lr=exp["lr"], momentum=exp["momentum"], weight_decay=exp["weight_decay"],
        byzantine_clients=list(range(int(exp["clients"]*exp["byzantine_ratio"]))),
        ema_alpha=tas["ema_alpha"], trust_power=tas["trust_power"],
        max_weight_ratio=tas["max_weight_ratio"],
        enable_exclusion=tas.get("exclusion", True),
        v1_sample_dim_ratio=tas["v1_sample_dim_ratio"],
        v2_sample_client_ratio=tas["v2_sample_client_ratio"],
        tolerance=tas["tolerance"], client_attack=config["client_attack"]["type"],
        v1_projection_tolerance=m.DEFAULT_PROJECTION_TOLERANCE)


def classify(rows):
    honest = [r for r in rows if not r["attack_active"]]
    attacks = [r for r in rows if r["attack_active"]]
    v1 = lambda r: bool(r["v1_trust"] or r["v1_agg"])
    v2 = lambda r: bool(r["v2_trust"] or r["v2_agg"])
    tp = sum(v1(r) or v2(r) for r in attacks)
    fp = sum(v1(r) or v2(r) for r in honest)
    return {
        "honest_proposals": len(honest), "attacked_proposals": len(attacks),
        "honest_accepted": len(honest)-fp, "false_rejected": fp,
        "true_rejected": tp, "attacks_missed": len(attacks)-tp,
        "honest_acceptance": (len(honest)-fp)/len(honest) if honest else None,
        "false_rejection_rate": fp/len(honest) if honest else None,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/len(attacks) if attacks else None,
        "v1_rejections": sum(v1(r) for r in attacks),
        "v2_rejections": sum(v2(r) for r in attacks),
        "v1_only": sum(v1(r) and not v2(r) for r in attacks),
        "v2_only": sum(v2(r) and not v1(r) for r in attacks),
        "both": sum(v1(r) and v2(r) for r in attacks),
    }


def run_one(scenario, seed, rounds, device, output_root, smoke_samples=0):
    config_file = ROOT / "configs" / f"table11_fashion_{scenario}.yaml"
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config["experiment"]["seed"] = seed
    output = output_root / scenario / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to mix or overwrite results: {output}")
    output.mkdir(parents=True)
    common = common_args(config, rounds, device, smoke_samples)
    write_json(output / "manifest.json", {
        "source_sha256": sha256(SOURCE), "driver_sha256": sha256(Path(__file__)),
        "exp2_sha256": sha256(ROOT / "scripts" / "exp2" / "exp2_run.py"),
        "model_sha256": sha256(ROOT / "core" / "model.py"),
        "config_sha256": sha256(config_file), "configuration": config,
        "data_root": os.environ.get("TASL_DATA_ROOT", str(ROOT / "data" / "temp")),
        "rounds": rounds, "scenario": scenario, "seed": seed,
        "smoke_samples_per_client": smoke_samples,
        "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scope": "single-process governance; local update hashes stand in for CIDs",
        "thresholds": {"weights": m.TOLERANCE, "aggregate": m.AGGREGATE_TOLERANCE,
                       "v1_projection": m.DEFAULT_PROJECTION_TOLERANCE},
        "attack_schedule": {r: asdict(a) for r, a in common["chaos_schedule"].items()},
    })
    results = {}
    for group in ("noexecutor", "trust_v1v2"):
        result = m.run_experiment_group(
            group, **common, round_record_dir=str(output / "round_records"))
        results[group] = result
        write_json(output / f"{group}.json", asdict(result))
        if len(result.round_logs) != rounds:
            raise AssertionError(f"Incomplete run: {scenario}, seed {seed}, {group}")
    control, full = results["noexecutor"], results["trust_v1v2"]
    for baseline, audited in zip(control.round_logs, full.round_logs):
        if baseline["state_hash"] != audited["state_hash"]:
            raise AssertionError(f"Recovered state differs at round {baseline['round']}")
        if baseline["accuracy"] != audited["accuracy"] or audited["skipped"]:
            raise AssertionError(f"Invalid recovery at round {baseline['round']}")
        if audited["recovery_attempted"] and not (
                audited["recovery_consistent"] and audited["recovery_v1_endorsed"]
                and audited["recovery_v2_endorsed"]):
            raise AssertionError(f"Recovery lacks dual endorsement at round {baseline['round']}")
    if full.blocked_rounds != full.attack_rounds:
        raise AssertionError("Attacked Executor proposal escaped audit")
    summary = classify(full.round_logs)
    summary.update({"scenario": scenario, "seed": seed, "rounds": rounds,
                    "noexecutor_avg": control.avg_accuracy, "full_avg": full.avg_accuracy,
                    "matching_state_rounds": rounds,
                    "recovery_attempts": sum(r["recovery_attempted"] for r in full.round_logs),
                    "dual_endorsed_recoveries": sum(r["recovery_v1_endorsed"] and
                                                    r["recovery_v2_endorsed"]
                                                    for r in full.round_logs)})
    write_json(output / "summary.json", summary)
    (output / "COMPLETE").write_text("paired run and recovery checks passed\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["lf", "scaling", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--smoke-samples", type=int, default=0,
                        help="Use a small per-client subset for logic preflight only")
    args = parser.parse_args()
    if args.device not in ("auto", "cpu", "cuda") and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("Invalid device; use auto, cpu, cuda, or cuda:N")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.device.startswith("cuda:") and int(args.device[5:]) >= torch.cuda.device_count():
        parser.error("Requested GPU index is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    scenarios = ["lf", "scaling"] if args.scenario == "both" else [args.scenario]
    for scenario in scenarios:
        for seed in args.seeds:
            print(f"START {scenario} seed={seed} rounds={args.rounds} device={device}", flush=True)
            result = run_one(scenario, seed, args.rounds, device,
                             args.output_root, args.smoke_samples)
            print(json.dumps(result, ensure_ascii=False), flush=True)
