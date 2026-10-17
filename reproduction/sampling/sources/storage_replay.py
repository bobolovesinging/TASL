#!/usr/bin/env python3
"""Actual-byte storage replay with an equivalent checkpoint-and-prune control."""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import tempfile
import time

import numpy as np
import torch


def torch_bytes(array):
    b = io.BytesIO(); torch.save(torch.from_numpy(array), b); return b.getvalue()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def metadata_blob(round_id, update_cids, checkpoint_cid):
    record = {
        "round": round_id,
        "participant_set": list(range(len(update_cids))),
        "update_cids": update_cids,
        "checkpoint_cid": checkpoint_cid,
        "weight_commitment": "a" * 64,
        "transcript_hash": "b" * 64,
        "executor_signature": "c" * 128,
        "auditor_signatures": ["d" * 128, "e" * 128],
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def verified_read(path, expected):
    t0 = time.perf_counter_ns()
    blob = open(path, "rb").read()
    elapsed = (time.perf_counter_ns()-t0)/1e6
    return digest(blob) == expected, elapsed, len(blob)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    captures = []
    for path in args.captures:
        with np.load(path) as z:
            captures.append((z["updates"].astype(np.float32), z["aggregate"].astype(np.float32)))

    round_stats = []
    representative_updates = []
    representative_checkpoints = []
    for r, (updates, aggregate) in enumerate(captures, 1):
        update_blobs = [torch_bytes(x) for x in updates]
        checkpoint_blob = torch_bytes(aggregate)
        update_cids = [digest(x) for x in update_blobs]
        checkpoint_cid = digest(checkpoint_blob)
        meta = metadata_blob(r, update_cids, checkpoint_cid)
        round_stats.append({
            "capture_round": r,
            "client_updates_bytes": sum(map(len, update_blobs)),
            "checkpoint_bytes": len(checkpoint_blob),
            "permanent_metadata_bytes": len(meta),
            "total_round_bytes": sum(map(len, update_blobs)) + len(checkpoint_blob) + len(meta),
        })
        representative_updates.append(update_blobs)
        representative_checkpoints.append(checkpoint_blob)

    avg_updates = statistics.mean(x["client_updates_bytes"] for x in round_stats)
    avg_checkpoint = statistics.mean(x["checkpoint_bytes"] for x in round_stats)
    avg_metadata = statistics.mean(x["permanent_metadata_bytes"] for x in round_stats)
    avg_total = avg_updates + avg_checkpoint + avg_metadata

    accounting = []
    for rounds in [50, 300]:
        full_total = rounds * avg_total
        for window in [3, 5, 10, 20]:
            # Full retention keeps every update and checkpoint active.
            full_active = full_total
            # Equivalent checkpoint-and-prune and HBS both keep permanent
            # digests, recent updates, and one current checkpoint active.
            equivalent_active = rounds*avg_metadata + min(window, rounds)*avg_updates + avg_checkpoint
            archived = max(rounds-window, 0)*avg_updates + max(rounds-1, 0)*avg_checkpoint
            for scheme, active in [
                ("full_retention", full_active),
                ("equivalent_checkpoint_prune", equivalent_active),
                ("hbs_state_classification", equivalent_active),
            ]:
                archive = 0 if scheme == "full_retention" else archived
                for replication in [1, 3]:
                    accounting.append({
                        "rounds": rounds,
                        "window": window,
                        "scheme": scheme,
                        "archive_replication_factor": replication,
                        "active_bytes": int(round(active)),
                        "archive_bytes_single_copy": int(round(archive)),
                        "replicated_archive_bytes": int(round(archive * replication)),
                        "total_physical_bytes": int(round(active + archive * replication)),
                        "active_reduction_vs_full": 1.0-active/full_active,
                    })

    # Bit-exact recent and archived retrieval checks using implemented local,
    # content-addressed semantics. The old object is moved, not deleted.
    work = tempfile.mkdtemp(prefix="tasl_storage_", dir=args.out)
    active_dir = os.path.join(work, "active"); archive_dir = os.path.join(work, "archive")
    os.makedirs(active_dir); os.makedirs(archive_dir)
    recent_blob = representative_checkpoints[-1]
    old_blob = representative_updates[0][0]
    recent_hash, old_hash = digest(recent_blob), digest(old_blob)
    recent_path = os.path.join(active_dir, recent_hash)
    old_active_path = os.path.join(active_dir, old_hash)
    open(recent_path,"wb").write(recent_blob); open(old_active_path,"wb").write(old_blob)
    old_archive_path = os.path.join(archive_dir, old_hash)
    shutil.move(old_active_path, old_archive_path)

    retrieval = []
    for name, path, expected in [
        ("recent_checkpoint", recent_path, recent_hash),
        ("historical_archived_update", old_archive_path, old_hash),
    ]:
        times=[]; ok=True; nbytes=0
        for _ in range(20):
            this_ok, ms, nbytes = verified_read(path, expected); ok &= this_ok; times.append(ms)
        retrieval.append({"object":name,"bytes":nbytes,"hash_verified":ok,
                          "median_retrieval_ms":statistics.median(times),
                          "p95_retrieval_ms":sorted(times)[18]})
    shutil.rmtree(work)

    def write_csv(name, records):
        with open(os.path.join(args.out,name),"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    write_csv("measured_round_bytes.csv",round_stats)
    write_csv("storage_accounting.csv",accounting)
    write_csv("retrieval_integrity.csv",retrieval)
    summary={
        "input_captures":args.captures,
        "measured_rounds":len(captures),
        "mean_client_updates_bytes_per_round":avg_updates,
        "mean_checkpoint_bytes":avg_checkpoint,
        "mean_permanent_metadata_bytes_per_round":avg_metadata,
        "accounting":accounting,
        "retrieval":retrieval,
        "interpretation":"HBS and an equivalent checkpoint-and-prune policy have identical active-byte accounting under matched retention and recovery semantics; moving old objects reduces active storage but does not reduce total physical storage unless the external archive is deleted.",
        "limitations":[
            "Actual torch serialization and SHA-256 are measured; 50/300-round totals replay the three captured rounds cyclically.",
            "Local content-addressed files model the repository's MockIPFSManager, not a deployed IPFS cluster.",
            "Replication, network transfer, blockchain node replication, and real transaction gas are excluded."
        ],
        "script_sha256":hashlib.sha256(open(__file__,"rb").read()).hexdigest(),
    }
    with open(os.path.join(args.out,"summary.json"),"w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--captures",nargs="+",required=True); p.add_argument("--out",required=True)
    run(p.parse_args())
