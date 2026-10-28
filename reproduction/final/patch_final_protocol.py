from pathlib import Path
import difflib
ROOT=Path(__file__).resolve().parent
def once(s,a,b):
 assert s.count(a)==1,(a[:85],s.count(a));return s.replace(a,b,1)
p=ROOT/'scripts/exp3/exp3_governance_tasv2.py';old=p.read_text(encoding='utf-8');s=old
s=once(s,'                             projection_tolerance=DEFAULT_PROJECTION_TOLERANCE):',
 '                             projection_tolerance=DEFAULT_PROJECTION_TOLERANCE,\n                             v2_reference_weights=None):')
s=once(s,'            recovered_weights, reference_weights, num_clients,\n            sample_ratio=v2_sample_ratio, seed=v2_sample_seed)',
 '            recovered_weights, reference_weights if v2_reference_weights is None else v2_reference_weights, num_clients,\n            sample_ratio=v2_sample_ratio, seed=v2_sample_seed)')
s=once(s,'    client_scaling_factor: float = 5.0,\n','    client_scaling_factor: float = 5.0,\n    exclusion_budget: Optional[int] = None,\n    capture_round_dir: Optional[str] = None,\n    final_state_path: Optional[str] = None,\n')
s=once(s,'    max_exclusions = len(byzantine_set) if enable_exclusion else 0',
 '    max_exclusions = (len(byzantine_set) if exclusion_budget is None else int(exclusion_budget)) if enable_exclusion else 0\n    if capture_round_dir:\n        os.makedirs(capture_round_dir,exist_ok=False)')
s=once(s,'        # --- 2. Shared current-gated direction-fused computation ---',
 '''        if capture_round_dir:
            target=os.path.join(capture_round_dir,f'round_{r:02d}.pt')
            torch.save(dict(round=r,parent_model=global_state,prior_trust_state=prior_trust_state,
                            local_updates=local_updates,parent_hash=parent_hash,
                            client_attack=client_attack,byzantine_clients=sorted(byzantine_set),
                            exclusion_budget=max_exclusions),target)

        # --- 2. Shared current-gated direction-fused computation ---''')
s=once(s,'        audit_reference = independently_recompute_weights() if (use_v1 or use_v2) else {}\n        if audit_reference:\n            assert audit_reference == correct_trust, "Independent trust recomputation differs"',
 '''        audit_reference_s = independently_recompute_weights() if use_v1 else {}
        audit_reference_p = independently_recompute_weights() if use_v2 else {}
        audit_reference = audit_reference_s if use_v1 else audit_reference_p
        for reference in (audit_reference_s,audit_reference_p):
            if reference:
                assert reference == correct_trust, "Independent role recomputation differs"''')
s=once(s,'                    published_w, audit_reference, num_clients)',
 '                    published_w, audit_reference_s, num_clients)')
s=once(s,'                    published_w, audit_reference, num_clients,\n                    sample_ratio=v2_sample_client_ratio, seed=s_trust)',
 '                    published_w, audit_reference_p, num_clients,\n                    sample_ratio=v2_sample_client_ratio, seed=s_trust)')
s=once(s,'                projection_tolerance=v1_projection_tolerance)',
 '                projection_tolerance=v1_projection_tolerance,\n                v2_reference_weights=independently_recompute_weights() if use_v2 else None)')
s=once(s,'    return Exp3Result(\n        name=group_name,',
 '''    if final_state_path:
        target_dir=os.path.dirname(os.path.abspath(final_state_path))
        os.makedirs(target_dir,exist_ok=True)
        torch.save(dict(model=global_state,trust_state=(ema_weights,cos_history,prev_anchor,
                        temporal_anchor,excluded_clients,violation_counts),
                        state_hash=protocol_state_hash(global_state,ema_weights,cos_history,
                        prev_anchor,temporal_anchor,excluded_clients,violation_counts)),final_state_path)

    return Exp3Result(
        name=group_name,''')
p.write_text(s,encoding='utf-8')
(ROOT/'final_protocol.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='C1_protocol',tofile='final_protocol')),encoding='utf-8')
p=ROOT/'run_experiments.py';s=p.read_text(encoding='utf-8')
s=once(s,'def run_one(out,name,gate,group,attack,rounds,smoke=0):',
 'def run_one(out,name,gate,group,attack,rounds,smoke=0,extra=None):')
s=once(s,"        group=group,client_attack=attack,configuration=CONFIG,rounds=rounds,smoke_samples_per_client=smoke,",
 "        group=group,client_attack=attack,configuration=CONFIG,rounds=rounds,smoke_samples_per_client=smoke,extra=extra or {},")
s=once(s,"        round_record_dir=str(dest/'rounds'))", "        round_record_dir=str(dest/'rounds'),**(extra or {}))")
s=s.replace("comments=['R4 round2 C1']","comments=['R4 round2 C2']")
p.write_text(s,encoding='utf-8')
print('Independent audit-role recomputation, capture and fixed-budget controls prepared.')
