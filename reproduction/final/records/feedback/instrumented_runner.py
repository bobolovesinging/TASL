def run_transformed_group(
    group_name: str,
    client_loaders,
    test_loader,
    rounds: int,
    chaos_schedule: Dict[int, RoundAttack],
    device: torch.device,
    seed: int,
    local_epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    byzantine_clients: Optional[List[int]] = None,
    ema_alpha: float = 0.6,
    trust_power: float = 5.0,
    max_weight_ratio: float = 2.0,
    enable_exclusion: bool = True,
    v1_sample_dim_ratio: float = 0.3,
    v1_projection_tolerance: float = DEFAULT_PROJECTION_TOLERANCE,
    v2_sample_client_ratio: float = 0.3,
    tolerance: float = TOLERANCE,
    client_attack: str = "label_flip",
    round_record_dir: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
    client_attack_start: int = 1,
    client_scaling_factor: float = 5.0,
    exclusion_budget: Optional[int] = None,
    capture_round_dir: Optional[str] = None,
    final_state_path: Optional[str] = None,
    input_transform=None,
) -> Exp3Result:
    """
    运行一个实验组.

    group_name in {"fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"}
    """
    assert group_name in {"noexecutor", "fedavg", "trust_only", "trust_v1", "trust_v2", "trust_v1v2"}

    # Reset RNG state for every group so no-Executor and audited variants are
    # compared under the same initialization and dataloader stochasticity.
    set_seed(seed)

    num_clients = len(client_loaders)
    global_model = FedAvgCNN().to(device)
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
    byzantine_set = set(byzantine_clients or [])

    use_v1 = group_name in ("trust_v1", "trust_v1v2")
    use_v2 = group_name in ("trust_v2", "trust_v1v2")
    use_trust = group_name != "fedavg"

    accuracy_history = []
    logs = []
    ema_weights = None
    cos_history = []
    excluded_clients: Set[int] = set()
    violation_counts = {cid: 0 for cid in range(num_clients)}
    max_exclusions = (len(byzantine_set) if exclusion_budget is None else int(exclusion_budget)) if enable_exclusion else 0
    if capture_round_dir:
        os.makedirs(capture_round_dir,exist_ok=False)

    v1_trust_detections = 0
    v1_agg_detections = 0
    v2_trust_detections = 0
    prev_anchor = None
    temporal_anchor = None
    v2_agg_detections = 0
    attack_rounds = 0
    blocked_rounds = 0
    heal_source_counts = {"V1": 0, "V2": 0, "V1+V2": 0, "Binding": 0}

    rng = random.Random(seed)  # same seed for all groups
    # The audit challenge is drawn from system entropy after the transcript is
    # fixed. It cannot be reconstructed from the public training seed.
    if round_record_dir:
        os.makedirs(round_record_dir, exist_ok=True)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    for r in range(1, rounds + 1):
        prior_trust_state = copy.deepcopy((
            ema_weights, cos_history, prev_anchor, temporal_anchor,
            excluded_clients, violation_counts,
        ))
        parent_hash = protocol_state_hash(global_state, *prior_trust_state)
        events = []
        # --- 1. Client training ---
        local_updates = {}
        flat_updates = {}

        for cid in range(num_clients):
            if cid in excluded_clients:
                continue

            set_seed(seed + r * 10000 + cid)
            kwargs = dict(global_state=global_state,
                          loader=client_loaders[cid],
                          device=device,
                          local_epochs=local_epochs,
                          lr=lr, momentum=momentum, weight_decay=weight_decay)

            if cid in byzantine_set and r >= client_attack_start and client_attack != 'none':
                if client_attack == "scaling":
                    raw_state = train_one_client_scaling(**kwargs, scaling_factor=client_scaling_factor)
                else:
                    raw_state = train_one_client_label_flip(**kwargs)
            else:
                raw_state = train_one_client(**kwargs)

            local_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                           for k, v in raw_state.items()}
            upd = state_sub(local_state, global_state)
            local_updates[cid] = upd
            flat_updates[cid] = flatten_update(upd)

        if input_transform is not None:
            local_updates = input_transform(r, global_state, prior_trust_state, local_updates)
            flat_updates = {cid: flatten_update(upd) for cid, upd in local_updates.items()}

        chaos = chaos_schedule.get(r, RoundAttack(active=False))
        attack_type = chaos.attack_type if chaos.active else None

        if capture_round_dir:
            target=os.path.join(capture_round_dir,f'round_{r:02d}.pt')
            torch.save(dict(round=r,parent_model=global_state,prior_trust_state=prior_trust_state,
                            local_updates=local_updates,parent_hash=parent_hash,
                            client_attack=client_attack,byzantine_clients=sorted(byzantine_set),
                            exclusion_budget=max_exclusions),target)

        # --- 2. Shared current-gated direction-fused computation ---
        if use_trust:
            correct_trust, sim_scores, new_anchor, ema_history = full_trust(
                flat_updates,
                trust_power=trust_power, max_weight_ratio=max_weight_ratio,
                norm_penalty_strength=0.8, refine_anchor=True,
                return_history=True,
                ema_weights=ema_weights, ema_alpha=ema_alpha,
                cos_history=cos_history,
                prev_anchor=prev_anchor, temporal_anchor=temporal_anchor,
            )
            if sum(correct_trust.values()) <= 1e-12:
                # The model, EMA, cosine history, both anchors, exclusions and
                # violation counters remain the committed parent state.
                state_hash=protocol_state_hash(global_state,*prior_trust_state)
                assert state_hash==parent_hash
                acc=evaluate(global_model,test_loader,device)
                accuracy_history.append(acc)
                row=dict(round=r,group=group_name,skipped=1,
                         attack_active=0,scheduled_attack_active=int(chaos.active),
                         attack_type='none',v1_trust=0,v1_agg=0,v2_trust=0,v2_agg=0,
                         accuracy=float(acc),parent_hash=parent_hash,state_hash=state_hash,
                         healed_applied=0,consensus_passed=0,
                         final_weights=json.dumps(correct_trust,sort_keys=True),
                         events=['no_eligible_updates','round_skipped'])
                logs.append(row)
                if round_record_dir:
                    with open(os.path.join(round_record_dir,f'{group_name}.jsonl'),'a') as handle:
                        handle.write(json.dumps(row,allow_nan=False)+'\n')
                continue
            fresh_trust = dict(correct_trust)
            ema_weights = dict(ema_history)
            cos_history.append(dict(sim_scores))
            prev_anchor = new_anchor
        else:
            sim_scores = {cid: 1.0 for cid in range(num_clients)}
            correct_trust = {cid: 1.0 / num_clients for cid in range(num_clients)}
            fresh_trust = dict(correct_trust)

        def independently_recompute_weights():
            """Recompute the round weights from a copy of the committed prior state."""
            (prior_ema, prior_history, prior_anchor, prior_temporal,
             _, _) = copy.deepcopy(prior_trust_state)
            recomputed, _, _ = full_trust(
                flat_updates,
                trust_power=trust_power,
                max_weight_ratio=max_weight_ratio,
                norm_penalty_strength=0.8,
                refine_anchor=True,
                ema_weights=prior_ema,
                ema_alpha=ema_alpha,
                cos_history=prior_history,
                prev_anchor=prior_anchor,
                temporal_anchor=prior_temporal,
            )
            return recomputed

        # No-Executor proposal for this exact round. A successful audit must
        # restore this aggregate, not a separate rollback/exclusion path.
        baseline_weights = dict(correct_trust)
        baseline_agg = weighted_average_updates(local_updates, baseline_weights)

        if use_trust and len(excluded_clients) < max_exclusions:
            active_ids = [cid for cid in range(num_clients) if cid not in excluded_clients]
            if len(active_ids) >= 2:
                active_norms = [
                    np.linalg.norm(flat_updates.get(cid, np.zeros(1)))
                    for cid in active_ids
                    if cid in flat_updates
                ]
                median_norm_r = np.median(active_norms) + 1e-12 if active_norms else 1.0
                for cid in active_ids:
                    if cid not in flat_updates:
                        continue
                    cs = sim_scores.get(cid, 0.5)
                    nr = np.linalg.norm(flat_updates[cid]) / median_norm_r
                    violated = (cs < -0.1) or (nr > 2.0)
                    if violated:
                        violation_counts[cid] += 1
                    else:
                        violation_counts[cid] = max(0, violation_counts[cid] - 1)
                    if violation_counts[cid] >= 2 and len(excluded_clients) < max_exclusions:
                        excluded_clients.add(cid)
                        violation_counts[cid] = 0
                        print(f"    [Exclusion] Client {cid} excluded (round {r})")

        # --- 3. Corrupt E attack ---
        executor_attack_active = bool(chaos.active and group_name != "noexecutor")
        if executor_attack_active:
            attack_rounds += 1
            # Publish FRESH trust (pre-EMA) so V1/V2 can verify honestly.
            # EMA-smoothed correct_trust is only used for healing.
            published_w, agg_w, attackers = apply_corrupt_e_attack(
                fresh_trust, sim_scores,
                byzantine_clients,
                attack_type, rng, num_clients,
                active_client_ids=list(local_updates.keys()))
        else:
            published_w = dict(fresh_trust)
            agg_w = dict(correct_trust)  # EMA-smoothed for aggregation stability
            attackers = set()

        proposal_agg = weighted_average_updates(local_updates, agg_w)
        audit_reference_s = independently_recompute_weights() if use_v1 else {}
        audit_reference_p = independently_recompute_weights() if use_v2 else {}
        audit_reference = audit_reference_s if use_v1 else audit_reference_p
        for reference in (audit_reference_s,audit_reference_p):
            if reference:
                assert reference == correct_trust, "Independent role recomputation differs"
        events.append("proposal_created")
        update_hashes = {cid: hash_local_update(update)
                         for cid, update in local_updates.items()}
        proposal_hash = proposal_commitment(
            round_number=r, parent_hash=parent_hash, aggregate=proposal_agg,
            published_weights=published_w, update_hashes=update_hashes)
        events.append("proposal_committed")
        # Auditors reconstruct the transcript from their local round state and
        # client updates before endorsing an Executor proposal.
        proposal_binding_valid = proposal_transcript_matches(
            proposal_hash, round_number=r, parent_hash=parent_hash,
            aggregate=proposal_agg, published_weights=published_w,
            update_hashes={cid: hash_local_update(update)
                           for cid, update in local_updates.items()})
        v1_challenge_seed = None
        v1_challenge = None
        if use_v1:
            v1_challenge_seed = secrets.randbits(128)
            v1_challenge = make_postcommit_challenge(
                flatten_update(proposal_agg).size, v1_challenge_seed)
            events.append("v1_challenge_sampled")

        # --- 4. V1/V2 审计 ---
        v1_trust_detected = False
        v1_agg_detected = False
        v2_trust_detected = False
        v2_agg_detected = False
        v2_suspected: Set[int] = set()
        trust_v1: Dict[int, float] = {}
        v1_projection_residual = None
        consensus_ok = True
        healed_applied = False
        heal_source = ""

        # Run the same validators on attack and honest rounds so false rejection
        # is measured directly. Attack labels are used only for evaluation.
        if use_v1 or use_v2:
            s_trust = seed + r * 137
            s_agg = seed + r * 971

            if use_v1:
                v1_trust_detected, trust_v1, _ = v1_verify_trust(
                    published_w, audit_reference_s, num_clients)
                if v1_trust_detected:
                    v1_trust_detections += 1

                v1_agg_detected, v1_dim_checked, v1_projection_residual = v1_verify_agg(
                    proposal_agg,
                    published_w, local_updates,
                    challenge=v1_challenge,
                    tolerance=v1_projection_tolerance)
                if v1_agg_detected:
                    v1_agg_detections += 1

            if use_v2:
                v2_trust_detected, v2_suspected = v2_verify_trust(
                    published_w, audit_reference_p, num_clients,
                    sample_ratio=v2_sample_client_ratio, seed=s_trust)
                if v2_trust_detected:
                    v2_trust_detections += 1

                v2_agg_detected = v2_verify_agg(
                    proposal_agg,
                    published_w, local_updates, seed=s_agg)
                if v2_agg_detected:
                    v2_agg_detections += 1

            # --- 5. Consensus ---
            v1_detected = v1_trust_detected or v1_agg_detected
            v2_detected = v2_trust_detected or v2_agg_detected

            if v1_detected or v2_detected or not proposal_binding_valid:
                consensus_ok = False
                if executor_attack_active:
                    blocked_rounds += 1
                if v1_detected and v2_detected:
                    heal_source = "V1+V2"
                    heal_source_counts["V1+V2"] += 1
                elif v1_detected:
                    heal_source = "V1"
                    heal_source_counts["V1"] += 1
                elif v2_detected:
                    heal_source = "V2"
                    heal_source_counts["V2"] += 1
                else:
                    heal_source = "Binding"
                    heal_source_counts["Binding"] += 1

        events.append("audit_rejected" if not consensus_ok else "audit_accepted")

        # --- 6. Aggregation, recovery, and recovery consistency check ---
        recovery = {"attempted": False, "consistent": False, "relative_error": 0.0}
        recovery_audit = {"v1_endorsed": False, "v2_endorsed": False,
                          "endorsed": False}
        recovery_hash = None
        skipped = False
        if consensus_ok:
            final_agg = baseline_agg if group_name == "noexecutor" else proposal_agg
        else:
            recovered_weights = independently_recompute_weights()
            final_agg = weighted_average_updates(local_updates, recovered_weights)
            recovery = check_recovery_consistency(final_agg, baseline_agg)
            events.append("recovery_recomputed")
            recovery_hash = proposal_commitment(
                round_number=r, parent_hash=parent_hash, aggregate=final_agg,
                published_weights=recovered_weights, update_hashes=update_hashes)
            recovery_challenge = make_postcommit_challenge(
                flatten_update(final_agg).size, secrets.randbits(128))
            recovery_audit = audit_recovery_candidate(
                final_agg, recovered_weights, independently_recompute_weights(),
                local_updates, challenge=recovery_challenge,
                num_clients=num_clients, use_v1=use_v1, use_v2=use_v2,
                v2_sample_ratio=v2_sample_client_ratio,
                v2_sample_seed=seed + r * 137,
                projection_tolerance=v1_projection_tolerance,
                v2_reference_weights=independently_recompute_weights() if use_v2 else None)
            events.append("recovery_audited")
            if recovery["consistent"] and recovery_audit["endorsed"] and \
                    proposal_transcript_matches(
                        recovery_hash, round_number=r, parent_hash=parent_hash,
                        aggregate=final_agg, published_weights=recovered_weights,
                        update_hashes=update_hashes):
                healed_applied = True
                events.append("recovery_verified")
            else:
                skipped = True
                events.append("recovery_failed")

        if snapshot_dir and not consensus_ok:
            torch.save({
                "round": r,
                "prior_model": global_state,
                "prior_trust_state": prior_trust_state,
                "local_updates": local_updates,
                "published_weights": published_w,
                "aggregate_weights": agg_w,
                "independent_weights": audit_reference,
                "parent_hash": parent_hash,
                "attack_type": attack_type,
                "v2_sample_seed": seed + r * 137,
            }, os.path.join(snapshot_dir, f"round_{r:02d}.pt"))

        if skipped:
            (ema_weights, cos_history, prev_anchor, temporal_anchor,
             excluded_clients, violation_counts) = copy.deepcopy(prior_trust_state)
            events.append("round_skipped")
        else:
            global_state = state_add(global_state, final_agg)
            events.append("committed_once")

        if use_trust and not skipped:
            agg_flat = flatten_update(final_agg)
            if np.linalg.norm(agg_flat) > 1e-12:
                temporal_anchor = agg_flat

        state_hash = protocol_state_hash(
            global_state, ema_weights, cos_history, prev_anchor,
            temporal_anchor, excluded_clients, violation_counts)
        if skipped:
            assert state_hash == parent_hash

        # --- 7. Evaluation ---
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)

        # --- 8. Logging ---
        evidence = make_round_evidence(
            round_number=r,
            attack_active=executor_attack_active,
            audit_executed=(use_v1 or use_v2),
            rejected=(not consensus_ok),
            recovery=recovery,
            parent_hash=parent_hash,
            state_hash=state_hash,
            events=events,
        )
        logs.append({
            "round": r, "group": group_name,
            "attack_active": int(executor_attack_active),
            "attack_type": attack_type if executor_attack_active else "none",
            "v1_trust": int(v1_trust_detected),
            "v1_agg": int(v1_agg_detected),
            "v1_projection_residual": v1_projection_residual,
            "v1_projection_tolerance": float(v1_projection_tolerance),
            "v1_challenge_seed": v1_challenge_seed,
            "proposal_commitment": proposal_hash,
            "proposal_binding_valid": int(proposal_binding_valid),
            "update_hashes": json.dumps(update_hashes, sort_keys=True),
            "recovery_commitment": recovery_hash,
            "recovery_v1_endorsed": int(recovery_audit["v1_endorsed"]),
            "recovery_v2_endorsed": int(recovery_audit["v2_endorsed"]),
            "recovery_endorsed": int(recovery_audit["endorsed"]),
            "v2_trust": int(v2_trust_detected),
            "v2_agg": int(v2_agg_detected),
            "consensus_passed": int(consensus_ok),
            "healed_applied": int(healed_applied),
            "heal_source": heal_source,
            "accuracy": float(acc),
            "skipped": int(skipped),
            "proposal_relative_error": aggregate_relative_error(proposal_agg, baseline_agg),
            "committed_relative_error": (
                aggregate_relative_error(final_agg, baseline_agg) if not skipped else 0.0),
            "final_weights": json.dumps(correct_trust, sort_keys=True),
            "update_norms": json.dumps({c:float(np.linalg.norm(v)) for c,v in flat_updates.items()},sort_keys=True),
            "cosines": json.dumps(sim_scores,sort_keys=True),
            "published_weights": json.dumps(published_w, sort_keys=True),
            "used_weights": json.dumps(agg_w, sort_keys=True),
            "attack_targets": json.dumps(sorted(attackers)),
            "excluded_clients": json.dumps(sorted(excluded_clients)),
            **evidence,
        })
        if round_record_dir:
            with open(os.path.join(round_record_dir, f"{group_name}.jsonl"),
                      "a", encoding="utf-8") as handle:
                handle.write(json.dumps(logs[-1], allow_nan=False) + "\n")

        if r % 10 == 0 or r == 1:
            parts = [f"[{group_name}] r{r:02d}"]
            parts.append(f"atk={executor_attack_active}")
            if use_v1:
                parts.append(f"V1_t={int(v1_trust_detected)},a={int(v1_agg_detected)}")
            if use_v2:
                parts.append(f"V2_t={int(v2_trust_detected)},a={int(v2_agg_detected)}")
            parts.append(f"ok={int(consensus_ok)} heal={int(healed_applied)}")
            parts.append(f"acc={acc:.2f}%")
            print(" | ".join(parts))

    if final_state_path:
        target_dir=os.path.dirname(os.path.abspath(final_state_path))
        os.makedirs(target_dir,exist_ok=True)
        torch.save(dict(model=global_state,trust_state=(ema_weights,cos_history,prev_anchor,
                        temporal_anchor,excluded_clients,violation_counts),
                        state_hash=protocol_state_hash(global_state,ema_weights,cos_history,
                        prev_anchor,temporal_anchor,excluded_clients,violation_counts)),final_state_path)

    return Exp3Result(
        name=group_name,
        best_accuracy=float(max(accuracy_history)) if accuracy_history else 0.0,
        avg_accuracy=float(np.mean(accuracy_history)) if accuracy_history else 0.0,
        accuracy_history=accuracy_history,
        v1_trust_detections=v1_trust_detections,
        v1_agg_detections=v1_agg_detections,
        v2_trust_detections=v2_trust_detections,
        v2_agg_detections=v2_agg_detections,
        attack_rounds=attack_rounds,
        blocked_rounds=blocked_rounds,
        heal_source_counts=heal_source_counts,
        round_logs=logs,
    )
