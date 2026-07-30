# CIFAR-10 Scaling Governance Unified Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CIFAR-10 Scaling governance experiment whose `no_executor` group is the exact Sweep TASL runner under a stable explicit seed, while V1, V2, and TAS defend A1 and A2 only at the Executor proposal layer.

**Architecture:** Extend `scripts/exp2/exp2_cifar10_v3.py::run_single()` with an optional typed Executor hook after the correct TASL aggregate is computed. The new `scripts/exp3/exp3_governance_cifar10_scaling.py` imports that runner and contains only deterministic Executor attacks, audits, recovery policy, group orchestration, and result output.

**Tech Stack:** Python 3, PyTorch, NumPy, `unittest`, CSV, JSON.

## Global Constraints

- Do not modify `scripts/exp3/exp_governance_v9_cifar10.py`.
- Do not change TASL trust-scoring mathematics.
- Scaling is exactly `global + 5 * (trained_local - global)`.
- Sweep and governance `no_executor` use the same `run_single()` call and explicit seed.
- V1 uses full trust comparison as its primary A1 defense and a cosine
  aggregate check as weaker A2 coverage.
- V2 uses random projection as its primary A2 defense and 30% sampled trust
  comparison as weaker, probabilistic A1 coverage.
- TAS blocks when V1 or V2 detects manipulation.
- Recovery commits the current round's precomputed TASL baseline aggregate.
- Preserve the existing Sweep result dictionary fields and add only compatible fields.

---

### Task 1: Restore Scaling and Stable Sweep Seeds

**Files:**
- Modify: `scripts/exp2/exp2_cifar10_v3.py:372-535`
- Modify: `scripts/exp2/exp2_cifar10_v3.py:1013-1120`
- Create: `tests/test_cifar_governance_scaling.py`

**Interfaces:**
- Produces: `scale_model_update(global_state, trained_state, scale_factor=5.0) -> Dict[str, torch.Tensor]`
- Produces: `train_scaling(global_state, loader, device, local_epochs=5, lr=0.1, scale_factor=5.0, **kwargs)`
- Produces: `resolve_run_seed(base_seed: int, run_seed: Optional[int]) -> int`
- Updates: `ATTACK_TRAIN_FNS["scaling"]`

- [ ] **Step 1: Write failing Scaling and seed tests**

```python
class ScalingTests(unittest.TestCase):
    def test_scale_model_update_multiplies_only_floating_delta(self):
        global_state = {
            "weight": torch.tensor([1.0, -2.0]),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        trained_state = {
            "weight": torch.tensor([2.0, 0.0]),
            "counter": torch.tensor(4, dtype=torch.int64),
        }

        result = sweep.scale_model_update(global_state, trained_state, 5.0)

        torch.testing.assert_close(result["weight"], torch.tensor([6.0, 8.0]))
        self.assertEqual(result["counter"].item(), 4)

    def test_scaling_attack_is_registered(self):
        self.assertIs(sweep.ATTACK_TRAIN_FNS["scaling"], sweep.train_scaling)

    def test_explicit_run_seed_replaces_process_hash_seed(self):
        self.assertEqual(sweep.resolve_run_seed(42, None), 42)
        self.assertEqual(sweep.resolve_run_seed(42, 314), 314)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling.ScalingTests -v
```

Expected: failures because `scale_model_update`, `train_scaling`, and
`resolve_run_seed` do not exist and `scaling` is absent from
`ATTACK_TRAIN_FNS`.

- [ ] **Step 3: Implement the minimal Scaling helper and trainer**

```python
def scale_model_update(global_state, trained_state, scale_factor=5.0):
    scaled = {}
    for key, value in trained_state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            base = global_state[key].to(value.device)
            scaled[key] = base + scale_factor * (value - base)
        else:
            scaled[key] = value
    return scaled


def train_scaling(global_state, loader, device, local_epochs=5, lr=0.1,
                  scale_factor=5.0, **kwargs):
    trained_state = train_honest(
        global_state, loader, device,
        local_epochs=local_epochs, lr=lr, **kwargs,
    )
    return scale_model_update(global_state, trained_state, scale_factor)
```

Register `"scaling": train_scaling`.

- [ ] **Step 4: Implement explicit seed resolution**

```python
def resolve_run_seed(base_seed, run_seed):
    return int(base_seed if run_seed is None else run_seed)
```

Add `--run-seed` to the Sweep CLI and replace
`args.seed + hash(algo + attack) % 1000` with the resolved explicit seed.
Print both partition seed and run seed.

- [ ] **Step 5: Run Task 1 tests and compile**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling.ScalingTests -v
python -m compileall -q scripts/exp2/exp2_cifar10_v3.py
```

Expected: three tests pass and compilation exits zero.

- [ ] **Step 6: Commit Task 1**

```powershell
git add scripts/exp2/exp2_cifar10_v3.py tests/test_cifar_governance_scaling.py
git commit -m "fix: restore deterministic CIFAR scaling sweep"
```

---

### Task 2: Add the Shared Executor Hook

**Files:**
- Modify: `scripts/exp2/exp2_cifar10_v3.py:1-65`
- Modify: `scripts/exp2/exp2_cifar10_v3.py:757-895`
- Modify: `tests/test_cifar_governance_scaling.py`

**Interfaces:**
- Produces: `ExecutorRoundContext`
- Produces: `ExecutorRoundDecision`
- Updates: `run_single(..., executor_hook=None)`
- Produces: result field `executor_history: List[Dict[str, Any]]`

- [ ] **Step 1: Write failing baseline decision tests**

```python
class ExecutorHookTests(unittest.TestCase):
    def test_select_executor_aggregate_uses_baseline_without_hook(self):
        baseline = {"w": torch.tensor([1.0])}
        context = make_context(baseline)

        decision = sweep.select_executor_aggregate(context, None)

        self.assertIs(decision.aggregate, baseline)
        self.assertEqual(decision.metadata, {})

    def test_select_executor_aggregate_uses_hook_decision(self):
        baseline = {"w": torch.tensor([1.0])}
        attacked = {"w": torch.tensor([9.0])}
        context = make_context(baseline)

        decision = sweep.select_executor_aggregate(
            context,
            lambda _: sweep.ExecutorRoundDecision(
                aggregate=attacked,
                metadata={"attack": "A1"},
            ),
        )

        self.assertIs(decision.aggregate, attacked)
        self.assertEqual(decision.metadata["attack"], "A1")
```

`make_context()` constructs an `ExecutorRoundContext` with two one-dimensional
client updates, normalized trust weights, one Byzantine client, and the
provided baseline aggregate.

- [ ] **Step 2: Run hook tests and verify RED**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling.ExecutorHookTests -v
```

Expected: failure because the context, decision, and selector do not exist.

- [ ] **Step 3: Implement typed context, decision, and selector**

```python
@dataclass(frozen=True)
class ExecutorRoundContext:
    round_number: int
    seed: int
    local_updates: Dict[int, Dict[str, torch.Tensor]]
    trust_weights: Dict[int, float]
    similarity_scores: Dict[int, float]
    baseline_aggregate: Dict[str, torch.Tensor]
    byzantine_ids: Tuple[int, ...]


@dataclass
class ExecutorRoundDecision:
    aggregate: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)


def select_executor_aggregate(context, executor_hook):
    if executor_hook is None:
        return ExecutorRoundDecision(context.baseline_aggregate)
    return executor_hook(context)
```

- [ ] **Step 4: Integrate the hook into `run_single()`**

After `compute_tasl_trust_weights()`:

```python
baseline_aggregate = fedavg_aggregate(local_updates, trust_weights)
context = ExecutorRoundContext(
    round_number=r,
    seed=seed,
    local_updates=local_updates,
    trust_weights=dict(trust_weights),
    similarity_scores=dict(cos_sims),
    baseline_aggregate=baseline_aggregate,
    byzantine_ids=tuple(sorted(byzantine_set)),
)
decision = select_executor_aggregate(context, executor_hook)
agg_update = decision.aggregate
if decision.metadata:
    executor_history.append(dict(decision.metadata))
```

Reject a non-`None` hook when `algo != "tasl"`. Return
`executor_history` with existing metrics.

- [ ] **Step 5: Run hook tests and all current tests**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling -v
python -m compileall -q scripts/exp2/exp2_cifar10_v3.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 6: Commit Task 2**

```powershell
git add scripts/exp2/exp2_cifar10_v3.py tests/test_cifar_governance_scaling.py
git commit -m "feat: add Executor hook to CIFAR TASL runner"
```

---

### Task 3: Implement A1/A2, V1/V2, TAS, and Recovery

**Files:**
- Create: `scripts/exp3/exp3_governance_cifar10_scaling.py`
- Modify: `tests/test_cifar_governance_scaling.py`

**Interfaces:**
- Produces: `normalize_weights(weights) -> Dict[int, float]`
- Produces: `apply_executor_attack(context, attack_type) -> ExecutorProposal`
- Produces: `verify_v1(proposal, context, tolerance=1e-4) -> bool`
- Produces: `verify_v2(proposal, context, projection_seed, rtol=0.03, atol=1e-8) -> bool`
- Produces: `GovernanceExecutor(group, schedule, seed)`
- Produces: `GovernanceExecutor.__call__(context) -> ExecutorRoundDecision`

- [ ] **Step 1: Write failing A1 and A2 tests**

```python
class ExecutorAttackTests(unittest.TestCase):
    def test_a1_has_full_v1_and_sampled_v2_trust_detection(self):
        context = make_context({"w": torch.tensor([2.0, 2.0])})
        proposal = governance.apply_executor_attack(context, "A1")

        self.assertTrue(governance.verify_v1(proposal, context))
        self.assertFalse(
            governance.verify_v2(proposal, context, projection_seed=7)
        )

    def test_a2_has_weaker_v1_cosine_and_full_v2_projection_detection(self):
        context = make_context({"w": torch.tensor([2.0, 2.0])})
        proposal = governance.apply_executor_attack(context, "A2")

        self.assertFalse(governance.verify_v1(proposal, context))
        self.assertTrue(
            governance.verify_v2(proposal, context, projection_seed=7)
        )
```

Use linearly independent client updates in `make_context()` so the A2
tampering changes the projection deterministically.

- [ ] **Step 2: Run attack tests and verify RED**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling.ExecutorAttackTests -v
```

Expected: import or attribute failure because the governance module and
functions do not exist.

- [ ] **Step 3: Implement proposals and attack transforms**

```python
@dataclass(frozen=True)
class ExecutorProposal:
    attack_type: str
    published_weights: Dict[int, float]
    aggregation_weights: Dict[int, float]
    aggregate: Dict[str, torch.Tensor]


def apply_executor_attack(context, attack_type):
    published = dict(context.trust_weights)
    aggregation = dict(context.trust_weights)
    if attack_type == "A1":
        for cid in context.byzantine_ids:
            published[cid] = 0.75
            aggregation[cid] = 0.75
    elif attack_type == "A2":
        for cid in context.byzantine_ids:
            aggregation[cid] = min(0.50, aggregation.get(cid, 0.0) + 0.25)
    else:
        raise ValueError(f"unsupported Executor attack: {attack_type}")
    normalized = normalize_weights(aggregation)
    return ExecutorProposal(
        attack_type=attack_type,
        published_weights=published,
        aggregation_weights=normalized,
        aggregate=sweep.fedavg_aggregate(context.local_updates, normalized),
    )
```

- [ ] **Step 4: Implement defense checks**

V1 compares every published client weight with `context.trust_weights` and
also performs the historical soft-weight cosine aggregate check at threshold
`0.97`. V2 samples 30% of clients for trust comparison, then normalizes
published weights, computes the expected aggregate, creates a unit random
vector from `np.random.RandomState(projection_seed)`, and compares actual and
expected projections using:

```python
difference = abs(actual_projection - expected_projection)
scale = max(abs(actual_projection), abs(expected_projection), 1e-12)
return difference > atol + rtol * scale
```

- [ ] **Step 5: Write failing group policy tests**

```python
class GovernancePolicyTests(unittest.TestCase):
    def test_trust_only_accepts_a1(self):
        decision = policy("trust_only", "A1")(make_context())
        self.assertFalse(decision.metadata["blocked"])
        self.assertFalse(decision.metadata["healed"])

    def test_v1_heals_a1_but_not_a2(self):
        a1 = policy("v1", "A1")(make_context())
        a2 = policy("v1", "A2")(make_context())
        self.assertTrue(a1.metadata["healed"])
        self.assertFalse(a2.metadata["healed"])

    def test_v2_heals_a2_but_not_a1(self):
        a1 = policy("v2", "A1")(make_context())
        a2 = policy("v2", "A2")(make_context())
        self.assertFalse(a1.metadata["healed"])
        self.assertTrue(a2.metadata["healed"])

    def test_tas_heals_both_attacks_to_exact_baseline(self):
        for attack in ("A1", "A2"):
            context = make_context()
            decision = policy("tas", attack)(context)
            self.assertTrue(decision.metadata["healed"])
            for key in context.baseline_aggregate:
                torch.testing.assert_close(
                    decision.aggregate[key],
                    context.baseline_aggregate[key],
                    rtol=0,
                    atol=0,
                )
```

- [ ] **Step 6: Run policy tests and verify RED**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling.GovernancePolicyTests -v
```

Expected: failures because `GovernanceExecutor` is not implemented.

- [ ] **Step 7: Implement group policy**

`GovernanceExecutor.__call__()` returns the baseline immediately on
non-attack rounds. On attack rounds it constructs the proposal, runs only the
checks enabled for the canonical group, and returns either the proposal or
`context.baseline_aggregate`. Metadata includes:

```python
{
    "round": context.round_number,
    "attack": attack_type,
    "v1_detected": v1_detected,
    "v2_detected": v2_detected,
    "blocked": blocked,
    "healed": blocked,
}
```

- [ ] **Step 8: Run Task 3 and full tests**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling -v
python -m compileall -q scripts/exp3/exp3_governance_cifar10_scaling.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 9: Commit Task 3**

```powershell
git add scripts/exp3/exp3_governance_cifar10_scaling.py tests/test_cifar_governance_scaling.py
git commit -m "feat: add CIFAR Executor attack defenses"
```

---

### Task 4: Add Governance CLI and Equivalence Verification

**Files:**
- Modify: `scripts/exp3/exp3_governance_cifar10_scaling.py`
- Modify: `tests/test_cifar_governance_scaling.py`
- Create: `configs/exp3_cifar10_scaling_governance.yaml`

**Interfaces:**
- Produces: `run_governance_groups(args) -> Dict[str, Dict[str, Any]]`
- Produces: CLI groups `no_executor,trust_only,v1,v2,tas`
- Produces: summary CSV and detailed JSON

- [ ] **Step 1: Write failing schedule and no-Executor wiring tests**

```python
class GovernanceRunnerTests(unittest.TestCase):
    def test_mixed_schedule_is_repeatable(self):
        first = governance.make_attack_schedule(
            rounds=20, seed=42, attack_type="mixed12",
            block_size=10, attacks_per_block=6,
        )
        second = governance.make_attack_schedule(
            rounds=20, seed=42, attack_type="mixed12",
            block_size=10, attacks_per_block=6,
        )
        self.assertEqual(first, second)

    def test_no_executor_group_builds_no_hook(self):
        self.assertIsNone(
            governance.build_executor_hook(
                "no_executor", schedule={1: "A1"}, seed=42
            )
        )

    def test_unknown_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported governance group"):
            governance.build_executor_hook("unknown", {}, 42)
```

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling.GovernanceRunnerTests -v
```

Expected: failures because schedule and hook builders do not exist.

- [ ] **Step 3: Implement deterministic schedule and CLI**

Add arguments for dataset partition, model run, attack schedule, groups,
Executor attack, explicit seed, and output directory. Defaults match the
historical Sweep configuration:

```text
clients=10
byzantine_ratio=0.4
alpha=0.3
batch_size=128
rounds=200
local_epochs=5
lr=0.1
seed=42
scale_factor=5.0
block_size=10
attacks_per_block=6
executor_attack=mixed12
groups=no_executor,trust_only,v1,v2,tas
```

The no-Executor group passes `executor_hook=None`; all groups call
`sweep.run_single("tasl", "scaling", ...)`.

- [ ] **Step 4: Implement JSON and CSV output**

Write a JSON object containing the full configuration and each group's
metrics/history. Write a CSV with:

```text
group,best_acc,avg_acc,last_acc,attack_rounds,blocked_rounds
```

Create the output directory during CLI execution only.

- [ ] **Step 5: Run full unit and CLI smoke tests**

Run:

```powershell
python -m unittest tests.test_cifar_governance_scaling -v
python scripts/exp3/exp3_governance_cifar10_scaling.py --help
python -m compileall -q scripts/exp2/exp2_cifar10_v3.py scripts/exp3/exp3_governance_cifar10_scaling.py
```

Expected: unit tests pass, help exits zero, and compilation exits zero.

- [ ] **Step 6: Run short equivalence check**

Run Sweep and governance `no_executor` with the same explicit seed, client
count, partition, rounds, local epochs, and learning rate. Compare every
accuracy-history element. On a CUDA host use:

```powershell
python scripts/exp2/exp2_cifar10_v3.py --attack scaling --algorithms tasl --skip-clean --clients 10 --byzantine-ratio 0.4 --rounds 2 --local-epochs 1 --seed 42 --run-seed 42 --output results/smoke_sweep
python scripts/exp3/exp3_governance_cifar10_scaling.py --groups no_executor --rounds 2 --local-epochs 1 --seed 42 --output results/smoke_governance
```

Expected: both histories contain exactly the same values. If local CUDA is
unavailable, run the commands on the configured GPU server or report the
environment limitation together with passing CPU unit tests.

- [ ] **Step 7: Commit Task 4**

```powershell
git add scripts/exp3/exp3_governance_cifar10_scaling.py tests/test_cifar_governance_scaling.py configs/exp3_cifar10_scaling_governance.yaml
git commit -m "feat: add unified CIFAR governance experiment"
```

---

### Task 5: Final Regression Review

**Files:**
- Verify: `scripts/exp2/exp2_cifar10_v3.py`
- Verify: `scripts/exp3/exp3_governance_cifar10_scaling.py`
- Verify: `tests/test_cifar_governance_scaling.py`
- Verify: `configs/exp3_cifar10_scaling_governance.yaml`

**Interfaces:**
- Consumes: all interfaces produced by Tasks 1-4
- Produces: verified branch ready for integration

- [ ] **Step 1: Run all targeted verification**

```powershell
python -m unittest tests.test_cifar_governance_scaling -v
python scripts/exp2/exp2_cifar10_v3.py --help
python scripts/exp3/exp3_governance_cifar10_scaling.py --help
python -m compileall -q scripts/exp2 scripts/exp3
git diff --check
```

Expected: tests and help commands pass, compilation exits zero, and
`git diff --check` prints no errors.

- [ ] **Step 2: Inspect the final diff**

Confirm that:

- the old CIFAR governance v9 file is unchanged;
- no existing user experiment configuration was modified;
- `hash(algo + attack)` is absent from the Sweep execution path;
- the new governance file imports and calls Sweep `run_single()`;
- `no_executor` builds no hook;
- healing returns `context.baseline_aggregate`.

- [ ] **Step 3: Record verification outcome**

Update this plan's checkboxes and report exact test counts, short-run
equivalence results, branch name, and worktree path in the final handoff.
