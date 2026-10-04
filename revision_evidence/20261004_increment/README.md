# TASL additional revision evidence 20261004

This additive snapshot supports the new or clarified recorded results in the
revision. Earlier tags `revision-evidence-20261002` and
`revision-sources-20261002` remain unchanged.

## Purpose and evidence scope

| Reviewer comments | Files | Supported conclusion |
| --- | --- | --- |
| R4.6, R4.13 | sampling/ and Table D.7 excerpt | Conditional weight-sampling detection budget: 9,207 finite-subset cases, 90 settings, three captured seed-42 rounds. Component checks, not protocol interception or distributed cost. |
| R4.10, R4.15 | cifar_sources/ | 60/60 CIFAR-10 LF seed summaries and 180 Best/Avg/ASR values agree with the author-designated workbook. 59 archived summaries plus one explicitly identified workbook completion; not 60 recovered historical training logs. |
| R4.1, R4.14 | storage/ and Table 7 excerpt | Four local retention policies and 48 archive-accounting rows; plaintext serialized tensor bytes and fixed-width placeholder metadata, not deployed ciphertext, proofs or ledger bytes. |
| R4.9 | learning/ | Existing five-seed lower-tail evidence: within-run 10th percentile across all 50 rounds, LF TASL minus FLTrust paired interval. Exploratory secondary comparison, without multiplicity correction. |

## Verify a fresh download

Download the repository ZIP at the fixed tag `revision-evidence-20261004`, then
from `revision_evidence/20261004_increment` run:

    python verify.py

Python 3.9+ standard library is sufficient. SHA-256 and byte counts cover every
manifest-listed file. The verifier checks CSV/workbook-extract agreement,
enumerates the finite subsets from captured weights, regenerates Table D.7,
checks all archive-accounting formulas and the four Table 7 policies, and
recomputes every saved P10 value from 30 raw learning trajectories (1,500 rounds).
The paired five-seed interval uses an exactly calculated df=4 Student t critical
value. It writes `generated/verification.json` and a Table D.7 CSV.

The timing rows are recorded CPU weight-check timings; their integrity and
coverage are checked, but the verifier does not reproduce runtime performance.
Only compact weights are extracted from the three capture NPZ files; their
original hashes are recorded. Large model-update payloads are not included.
The `sources/` files preserve original computation code for provenance; old
comments use historical auditor names. `original_run_sampling_ratio.py` expects
the original directory structure and full capture NPZ inputs. It is not the
self-contained download verifier. Storage serialization reruns also require
Torch and the original captures; the included standard-library check regenerates
accounting from the saved measured sizes rather than remeasuring payloads.

## Boundaries

No new training was performed for this publication. Summaries, captured rounds,
and independent training seeds have separate denominators. Model consistency
checks do not prove arbitrary client-poisoning robustness. Complete historical
main-table trajectories, executable/environment certification, threshold CKKS,
IPFS network measurements and multi-node ledger deployment remain outside this
snapshot. The designated spreadsheet is represented by its hash and cell-addressed
extract; no original spreadsheet, review letter or full manuscript is published.
