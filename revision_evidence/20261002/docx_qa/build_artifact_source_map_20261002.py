"""Index every current table/figure, explicitly distinguishing missing evidence."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'r4_tasl_experiments_20260921'
source=(R/'manuscript/main.tex').read_text(encoding='utf-8')
known={
 'tab:comprehensive_comparison':('literature comparison; no measured numeric baseline',['manuscript/references.bib']),
 'fig:system_model':('protocol schematic; DKG/IPFS architecture not a deployment measurement',['manuscript/Figure_1_TASL.pdf']),
 'tab:robustness_combined':('partially verified historical main table; missing original traces outside indexed CIFAR LF',['r4_tasl_experiments_20260921/legacy_main_sources_20261002/workbook_validation.json']),
 'tab:ratio_sweep':('CIFAR workbook extract verified; complete MNIST/Fashion trace provenance remains missing',['r4_tasl_experiments_20260921/legacy_main_sources_20261002/workbook_validation.json']),
 'tab:governance':('legacy saved precision verified, not final-protocol experiments',['r4_tasl_experiments_20260921/legacy_sources_20261002/validation.json','docx_qa/validate_legacy_sources_20261002.py']),
 'tab:governance_40':('legacy saved precision verified; some CIFAR reconstructed from different historical logs',['r4_tasl_experiments_20260921/legacy_sources_20261002/validation.json','docx_qa/validate_legacy_sources_20261002.py']),
 'tab:storage_growth':('metadata-only analytical estimate, not measured physical IPFS storage',['manuscript/main.tex']),
 'tab:semantic_retention':('serialized three-round replay extrapolated to the displayed retention horizons',['r4_tasl_experiments_20260921/storage_replay/results/semantic_retention_summary.json','r4_tasl_experiments_20260921/storage_replay/semantic_retention_analysis.py']),
 'tab:gas_analysis':('analytical gas estimate; no executable deployed-contract benchmark',['manuscript/main.tex']),
 'tab:appendix_tas_final':('final fixed-participation single-seed audit CSVs',['r4_tasl_experiments_20260921/final_protocol_20260928/results/no_exclusion_20260929']),
 'tab:appendix_tas_recovery':('paired state checks in the same final audit CSVs',['r4_tasl_experiments_20260921/final_protocol_20260928/results/no_exclusion_20260929']),
 'tab:appendix_cifar_final':('CIFAR final single-seed audit CSVs',['r4_tasl_experiments_20260921/final_protocol_20260928/results']),
 'tab:tas_five_seed':('five-seed Full TAS/reference validation',['r4_tasl_experiments_20260921/final_protocol_20260928/results/fashion_fixed_five_seed_20260930/validation.json','r4_tasl_experiments_20260921/final_protocol_20260928/validate_fashion_fixed_five_seed.py']),
 'tab:tas_single_five_seed':('five-seed single-auditor validation',['r4_tasl_experiments_20260921/final_protocol_20260928/results/fashion_single_auditor_five_seed_20261002/validation.json','r4_tasl_experiments_20260921/final_protocol_20260928/validate_single_auditor_five_seed.py']),
 'tab:appendix_v1_sensitivity':('captured-update component grid, not independent training seeds',['r4_tasl_experiments_20260921/audit_replay/results_refined_thresholds_20260923','r4_tasl_experiments_20260921/audit_replay/refined_threshold_sensitivity.py']),
 'tab:appendix_v2_sensitivity':('captured-update component grid, not deployment proof',['r4_tasl_experiments_20260921/audit_replay/results_refined_thresholds_20260923','r4_tasl_experiments_20260921/audit_replay/refined_threshold_sensitivity.py']),
 'tab:principal_five_seed':('five-seed matched learning, offline statistical regeneration',['r4_tasl_experiments_20260921/principal_5seed/aggregate_summary.csv','docx_qa/verify_local_reproduction_20261002.py']),
 'tab:client_weight_distribution':('full client-round distribution with zero weights included',['r4_tasl_experiments_20260921/principal_5seed/results','r4_tasl_experiments_20260921/principal_5seed/summarize_5seed.py']),
 'fig:principal_uncertainty':('five-seed mean/SD curves, coordinates verified',['r4_tasl_experiments_20260921/principal_5seed/aggregate_curves.csv','docx_qa/r4_continue_20260929/principal_uncertainty.tex']),
 'tab:appendix_trust_sensitivity':('single-seed EMA/min-cosine parameter runs',['r4_tasl_experiments_20260921/ema_sensitivity/results','r4_tasl_experiments_20260921/principal_5seed/threshold_results']),
 'tab:appendix_scale':('single-seed client scale/Dirichlet tests',['r4_tasl_experiments_20260921/alpha_sweep/results','r4_tasl_experiments_20260921/principal_5seed']),
 'tab:appendix_component_cost':('GPU tensor timings and linear CKKS projections, not networking',['r4_tasl_experiments_20260921/crypto_system/system_results/component_timing.csv','r4_tasl_experiments_20260921/crypto_system/results/client_scaling_projection.csv']),
 'tab:local_recovery_cost':('controlled CPU subpath timing, fixed captured weights',['r4_tasl_experiments_20260921/crypto_system/results/controlled_blas1_three_rounds_20261002.json','r4_tasl_experiments_20260921/crypto_system/results/controlled_blas1_three_rounds_repeat_20261002.json']),
 'tab:minmax_stress':('single-seed negative stress finding, not evidence of robust performance',['r4_tasl_experiments_20260921/minmax_20261002']),
 'tab:fixed_constant_sensitivity':('ten single-seed LF scans, defaults unchanged',['r4_tasl_experiments_20260921/recent_baseline_20261002/trust_sweep/validation.json','docx_qa/validate_trust_sweep_20261002.py']),
 'tab:recent_baseline':('six matched single-seed trajectories; LASA author aggregation',['r4_tasl_experiments_20260921/recent_baseline_20261002/validation.json','docx_qa/validate_recent_baseline_20261002.py'])
}
labels=re.findall(r'\\label\{((?:tab|fig):[^}]+)\}',source);assert len(labels)==len(set(labels))
assert set(labels)==set(known),(set(labels)-set(known),set(known)-set(labels))
records=[]
for label in labels:
 status,paths=known[label]
 items=[]
 for p in paths:
  q=R/p; assert q.exists(),p
  items.append({'path':p,'kind':'directory' if q.is_dir() else 'file',
                'sha256':None if q.is_dir() else hashlib.sha256(q.read_bytes()).hexdigest()})
 records.append({'label':label,'manuscript_line':source[:source.index('\\label{'+label+'}')].count('\n')+1,
                 'evidence_scope':status,'sources':items})
out=E/'reproduction_20261002/artifact_source_map.json'
out.write_text(json.dumps({'all_current_table_figure_labels':len(records),'records':records,
 'scope':'Every current artifact indexed; an index is not full original-trace recovery or all-table regeneration.'},indent=2))
print(json.dumps({'artifacts_indexed':len(records),'table_count':sum(r['label'].startswith('tab:') for r in records),
                  'figure_count':sum(r['label'].startswith('fig:') for r in records)}))
