# TASL

Research code for **TASL: Byzantine-Resilient Swarm Learning with Tripartite Auditing and Hierarchical Blockchain Storage**.

The final second-revision implementation and reproducible results are in [`reproduction/final`](reproduction/final). Start with its README and run `python reproduction/final/reproduce.py verify` to validate the downloaded numerical records without a GPU. The `paper-v2.4` tag identifies the current revision; the commit object fixes its contents. The MNIST/Fashion-MNIST governance comparisons and both exclusion budgets are in [`reproduction/governance`](reproduction/governance). Run `python reproduction/governance/verify.py` to verify those numerical records. The updated auditor implementation, governance rows, sampling study, and verifier timings are in [`reproduction/auditors`](reproduction/auditors); run `python reproduction/auditors/verify.py` to validate them.

The final source implements direction-screened trust weighting with post-smoothing norm exclusion, bound two-role proposal checking, complete-state recovery, and differentiated local retention. Its measured scope is a single-host prototype. Threshold key services, independent networked audits, IPFS transfer and ledger confirmation are proposed deployment interfaces.

## Earlier records

The other training entry points and reproduction directories contain the preceding configuration. They are retained for historical traceability in `paper-v1` and are not final-method validation. For current results use the final directory above.

The final Table 4 V_P rows use normalized A1 proposals. Their executed source bundle, raw round records, and verification/reproduction commands are in `reproduction/auditors/README.md` (`paper-v2.4`).
