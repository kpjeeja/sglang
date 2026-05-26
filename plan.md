# Add Zipfian Shared-Prefix Sampling to `bench_serving`

## Goal Description

Add an opt-in rank-based Zipfian (power-law) prefix-popularity workload mode to the existing `generated-shared-prefix` (GSP) dataset in `python -m sglang.bench_serving`. The mode samples each request's prefix group from the distribution

```text
weight(rank) = 1 / rank ** alpha          (rank starts at 1)
probability(rank) = weight(rank) / sum(weight)
```

while keeping the legacy uniform behavior byte-equivalent by default. The change is scoped to the benchmark/dataset ownership boundary: CLI flag registration in `python/sglang/bench_serving.py`, the `GeneratedSharedPrefixDataset` dataclass and the `sample_generated_shared_prefix_requests` generator in `python/sglang/benchmark/datasets/generated_shared_prefix.py`, the dataset-cache key/bypass policy, the user-facing doc at `docs/developer_guide/bench_serving.md`, and a deterministic CPU-only test in `test/registered/bench_fn/test_benchmark_datasets_api.py`. No server-runtime, scheduling, warmup, or unrelated dataset code is modified.

The deliverable is a polished pull request: implementation, CPU-only tests, user-facing documentation, and a clear PR description that explains the API and how it follows existing `bench_serving` dataset patterns.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. All tests are CPU-only and use a fixed `--seed`.

- AC-1: Backward compatibility for default (uniform) GSP behavior.
  - Positive Tests (expected to PASS):
    - With no new flags supplied, `sample_generated_shared_prefix_requests(...)` returns a `List[DatasetRow]` whose length equals `num_groups * prompts_per_group * num_turns`, with each group's prefix appearing exactly `prompts_per_group * num_turns` times (counted by routing_key when `send_routing_key=True`, or by reconstructed group membership otherwise), for fixed seed.
    - Default-mode invocation with the same `--seed` and same other args produces identical `DatasetRow` lists (same `prompt`, `prompt_len`, `output_len`, `routing_key`, and same order) across two invocations.
    - Default-mode cache read/write behavior (the path and condition `range_ratio == 1 and not send_routing_key and num_turns == 1`) is unchanged: a uniform-mode run still reads from and writes to the existing cache file path.
  - Negative Tests (expected to FAIL):
    - Any change that alters the uniform-mode `DatasetRow` list for fixed args+seed must cause AC-1 tests to fail.
    - Any change that alters the uniform-mode cache filename or cache hit/miss condition must cause AC-1 tests to fail.

- AC-2: Opt-in CLI surface for Zipfian sampling.
  - AC-2.1: New flags exist on `python -m sglang.bench_serving` for the `generated-shared-prefix` dataset.
    - Positive: `--gsp-group-distribution {uniform,zipf}` (default `uniform`) and `--gsp-zipf-alpha FLOAT` (default `None`, only meaningful when `--gsp-group-distribution zipf`) are registered. `--help` shows both flags with the Zipf formula and the meaning of `alpha`.
    - Negative: A test that runs `python -m sglang.bench_serving --help` fails if either flag is missing from the GSP section or if `--help` does not state the rank-based Zipf formula.
  - AC-2.2: Argparse-level rejection of bad distribution choices.
    - Positive: `--gsp-group-distribution invalid_name` exits non-zero with an argparse error mentioning the allowed choices.
    - Negative: Any silent acceptance of an unknown distribution name causes the test to fail.
  - AC-2.3: Validation rejects malformed combinations.
    - Positive (all should raise a clear `ValueError`/argparse error at parse or `from_args` time):
      - `--gsp-group-distribution zipf` without `--gsp-zipf-alpha`.
      - `--gsp-group-distribution uniform` with `--gsp-zipf-alpha` explicitly set.
      - `--gsp-group-distribution zipf --gsp-zipf-alpha 0`.
      - `--gsp-group-distribution zipf --gsp-zipf-alpha -0.5`.
      - `--gsp-group-distribution zipf --gsp-zipf-alpha nan`.
      - `--gsp-group-distribution zipf --gsp-zipf-alpha inf`.
    - Negative: Any of the above combinations being silently accepted causes the test to fail.

- AC-3: Zipf sampling math is exactly rank-based with rank starting at 1.
  - Positive:
    - A unit test importing the implementation's helper (or the generator under a test-only seam) verifies the probability vector for `num_groups=N, alpha=a` equals (within floating-point tolerance) `np.array([1.0 / ((i + 1) ** a) for i in range(N)]) / sum_of_weights`.
    - For at least one non-trivial `(N, a)` pair (e.g. `N=4, a=1.5`), the computed probability vector is asserted against precomputed reference numbers to three decimal places.
  - Negative:
    - A test that compares the helper's output against the LoRA `skewed` math `weights = [alpha ** -i for i in range(N)]` must reject the implementation if the LoRA formula is used.
    - A test passing `alpha = 1.0`, `N = 3` must yield probabilities approximately `[6/11, 3/11, 2/11]`; any other vector fails.

- AC-4: Reproducibility under fixed `--seed`.
  - Positive:
    - Two calls to `sample_generated_shared_prefix_requests(...)` with `--gsp-group-distribution zipf`, the same `seed`, and the same other args return identical `DatasetRow` lists (same `prompt`, `prompt_len`, `output_len`, `routing_key`, and same order).
    - The Zipf branch obtains randomness from an isolated `numpy.random.Generator` seeded from `--seed`. The Zipf branch must not consume any draws from the module-level `random.Random` or `numpy.random` global state that uniform-mode generation uses for system prompts and questions; therefore the system prompts and questions generated under uniform mode and under Zipf mode for the same args+seed are byte-equal.
  - Negative:
    - Two calls with different `--seed` values must yield different sampled group sequences (sanity assertion: at least one differing slot).
    - If the Zipf branch mutates the global `numpy.random` state, the per-slot questions for Zipf mode will diverge from uniform mode for the same seed; a test comparing the generated question pool across the two modes will fail. The implementation must therefore use an isolated RNG.

- AC-5: Deterministic skew check under Zipf mode.
  - Positive:
    - With `num_groups=4, prompts_per_group=25, num_turns=1, alpha=2.0, seed=0`, the per-group request counts are computed once against the implementation and asserted to exact values (no tolerance window, no statistical inequality). The expected counts are derived from the implementation in a test fixture and pinned.
    - Independent assertion: with the same configuration, the rank-1 group's count is strictly greater than the rank-`num_groups` group's count.
  - Negative:
    - A flaky statistical assertion (e.g., "rank-1 should hold at least X% of requests" with no fixed expectation) must not be present.
    - Any change that perturbs the deterministic per-group counts for the pinned `(num_groups, prompts_per_group, alpha, seed)` tuple must fail this test.

- AC-6: Dataset-cache safety — Zipf bypasses the on-disk cache.
  - Positive:
    - In Zipf mode, the dataset is always regenerated. The cache file at `~/.cache/sglang/benchmark/gen_shared_prefix_*.pkl` is neither read nor written by a Zipf invocation, regardless of whether `range_ratio == 1 and not send_routing_key and num_turns == 1`.
    - The uniform-mode cache filename and gating condition are unchanged. A uniform-mode invocation continues to read/write the same cache filename as before.
    - A test that pre-populates a cache file under uniform mode and then invokes Zipf mode for the same `(seed, num_groups, prompts_per_group, system_prompt_len, question_len, output_len, tokenizer)` confirms the Zipf result is computed fresh and does not match the pre-populated payload.
  - Negative:
    - Any path where a Zipf invocation reuses a uniform cache file (or vice versa) causes the AC-6 test to fail.

- AC-7: Coherent request-count semantics under Zipf.
  - Positive:
    - The total number of returned `DatasetRow` objects in Zipf mode equals `num_groups * prompts_per_group * num_turns` — identical to uniform mode. `--num-prompts` continues to be ignored by the GSP dataset, as it is today.
    - `--help` and `docs/developer_guide/bench_serving.md` state that under Zipf mode, the total request count is still `num_groups * prompts_per_group * num_turns`; only the per-slot group assignment changes.
    - In Zipf mode, every `DatasetRow.prompt` is a unique string (each slot pairs a sampled group's system prompt with a per-slot unique question), even though sampled group prefixes may repeat across slots.
  - Negative:
    - Any return value of total length different from `num_groups * prompts_per_group * num_turns` fails AC-7.
    - Duplicate `DatasetRow.prompt` values within a single Zipf invocation fail AC-7.

- AC-8: `--gsp-ordered` and shuffling have consistent semantics across modes.
  - Positive:
    - When `--gsp-ordered` is set under Zipf mode, the returned `DatasetRow` list preserves the order in which slots were generated/sampled (no final shuffle). Two runs with the same seed produce identical order.
    - When `--gsp-ordered` is NOT set under Zipf mode, the implementation applies the same final shuffle it applies in uniform mode, using the same RNG path it uses today. Two runs with the same seed still produce identical order.
  - Negative:
    - A test that pins ordering under Zipf with `--gsp-ordered` and detects an unexpected reshuffle fails AC-8.
    - A test that detects the Zipf branch using a different shuffle path than uniform mode (such that the shuffle's RNG state diverges) fails AC-8.

- AC-9: Documentation and `--help` are accurate and complete.
  - Positive:
    - `docs/developer_guide/bench_serving.md` gains a subsection describing: the new `--gsp-group-distribution` and `--gsp-zipf-alpha` flags, the exact rank-based Zipf formula, the acceptable range of `alpha` (finite, strictly positive), an example command using multiple shared-prefix groups, a note that under Zipf the cache is bypassed, and a disclaimer stating this controls prefix-popularity shape only and does not by itself claim a production trace or guarantee an observed cache-hit rate.
    - `--help` output for both new flags contains the rank-based Zipf formula and the `alpha` constraint.
  - Negative:
    - Missing formula, missing alpha constraint, missing example, or missing disclaimer in either `--help` or the doc file fails AC-9.

- AC-10: Tests live in the existing CPU-only benchmark dataset test suite.
  - Positive:
    - All new tests are added to `test/registered/bench_fn/test_benchmark_datasets_api.py` (or a sibling unit file in the same directory if a clean split is needed), follow the existing `CustomTestCase`/`register_cpu_ci` registration pattern, use the existing lightweight tokenizer fixture, and run without launching a server.
    - The targeted test file passes locally via `python -m unittest test.registered.bench_fn.test_benchmark_datasets_api` (exact command captured in the PR description).
  - Negative:
    - Adding the new tests under `test/registered/bench_fn/test_bench_serving_functionality.py` (the server-backed suite) when behavior is coverable at the dataset/unit level fails AC-10.
    - Any test that requires GPU, network access, or a live model server fails AC-10.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation:

- Registers `--gsp-group-distribution {uniform,zipf}` (default `uniform`) and `--gsp-zipf-alpha FLOAT` (default `None`) in `python/sglang/bench_serving.py` under the existing GSP argument group, with `--help` text that states the rank-based Zipf formula and the `alpha` constraint.
- Performs validation of `alpha` and flag-combination compatibility either at argparse time (via `type=` callable / mutually-exclusive checks) or in `GeneratedSharedPrefixDataset.from_args`, with a clear error message in each rejection path.
- Adds two fields, `group_distribution: str = "uniform"` and `zipf_alpha: Optional[float] = None`, to the `GeneratedSharedPrefixDataset` dataclass, threaded through `from_args` and into the call to `sample_generated_shared_prefix_requests`.
- Extends `sample_generated_shared_prefix_requests` with two new keyword arguments `group_distribution="uniform"` and `zipf_alpha=None`, preserving the existing positional/keyword call signature for current callers.
- Adds an internal helper (e.g. `_zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray`) that returns the exact rank-based probability vector, callable directly from tests.
- Uses an isolated `numpy.random.Generator` seeded from the existing `seed` argument for Zipf sampling, leaving global `random`/`numpy.random` state untouched by the new code path.
- Bypasses the on-disk cache entirely when `group_distribution != "uniform"`: neither reads nor writes the cache file, regardless of other parameters.
- Updates the user-facing doc `docs/developer_guide/bench_serving.md` with a Zipf subsection including formula, alpha range, example command, cache-bypass note, and non-production-trace disclaimer.
- Adds CPU-only deterministic tests to `test/registered/bench_fn/test_benchmark_datasets_api.py` covering AC-1, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, plus CLI validation tests (AC-2.2, AC-2.3) wherever the existing test harness allows invoking the parser without a live server.
- Runs `pre-commit run --files <changed files>` and the targeted test command locally; the PR description records exactly what was run.

### Lower Bound (Minimum Acceptable Scope)

The implementation does everything in the upper bound except:

- The internal `_zipf_group_probs` helper may be inlined into the generator if doing so keeps the math auditable by reading the function; in that case the test asserts against the implementation directly through a public seam (e.g., by exposing the probability vector via a returned attribute, a debug log captured by `caplog`, or a thin module-level function call).
- The `--help` text may be a single concise sentence per flag, as long as the formula and alpha constraint are present.
- Tests may consolidate AC-4, AC-5, AC-8 into a single fixture if doing so does not weaken any individual assertion.

### Allowed Choices

- Can use:
  - `numpy.random.default_rng(seed)` for the isolated Zipf RNG stream.
  - argparse `type=` callable for `--gsp-zipf-alpha` to perform finite/positive validation at parse time; alternatively perform the same validation in `from_args`.
  - Either inlining the Zipf math in `sample_generated_shared_prefix_requests` or factoring it into a small module-private helper.
  - Existing `compute_random_lens` and `gen_prompt` helpers from `python/sglang/benchmark/datasets/common.py` for question generation under Zipf, exactly as uniform mode uses them.
- Cannot use:
  - The LoRA `skewed` distribution helper or its `alpha ** -i` math for Zipf. Zipf must use `1 / rank ** alpha`.
  - Mutating module-level `random` or `numpy.random` global state from the Zipf branch.
  - A new dataset name or a new top-level dataset class. The feature is an additive option on the existing `generated-shared-prefix` dataset.
  - Modifications to server runtime, request scheduler, `--request-rate`, `--max-concurrency`, `--flush-cache`, `--warmup-requests`, or unrelated datasets.
  - Extending the existing cache key for Zipf mode (per locked decision DEC-7, Zipf bypasses the cache entirely).

> **Note on Deterministic Designs**: The draft and resolved user decisions narrow the design significantly. CLI flag names (`--gsp-group-distribution`, `--gsp-zipf-alpha`) follow the existing `--gsp-*` convention. Zipf math is fixed (`1 / rank^alpha`, rank starts at 1). Total request count is `num_groups * prompts_per_group * num_turns`. Cache is bypassed in Zipf mode. `--gsp-ordered` has the same meaning as in uniform mode. `alpha` must be strictly positive and finite. The upper and lower bounds therefore converge on a narrow design.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Pseudocode for the Zipf branch inside `sample_generated_shared_prefix_requests`:

```python
def sample_generated_shared_prefix_requests(
    num_groups, prompts_per_group, system_prompt_len, question_len, output_len,
    range_ratio, tokenizer, seed,
    send_routing_key=False, num_turns=1, fast_prepare=False, ordered=False,
    group_distribution="uniform", zipf_alpha=None,
):
    # Existing path: build system_prompts (length num_groups) and
    # questions (length num_groups * prompts_per_group * num_turns),
    # using the existing module-level random / numpy.random seeded by `seed`.
    # This must be byte-identical between uniform and zipf modes.

    if group_distribution == "uniform":
        # Existing assignment: slot i pairs questions[i] with
        # system_prompts[ i // (prompts_per_group * num_turns) ].
        group_ids = _uniform_group_ids(num_groups, prompts_per_group, num_turns)
    elif group_distribution == "zipf":
        # Isolated RNG: does not perturb global random state.
        rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        total_slots = num_groups * prompts_per_group * num_turns
        group_ids = rng.choice(num_groups, size=total_slots, replace=True, p=probs)
    else:
        raise ValueError(...)

    rows = [
        DatasetRow(
            prompt=system_prompts[g] + questions[i],
            prompt_len=...,
            output_len=output_lens[i],
            routing_key=(str(g) if send_routing_key else None),
            ...,
        )
        for i, g in enumerate(group_ids)
    ]

    if not ordered:
        # Existing shuffle path, using the same RNG it uses today.
        random.shuffle(rows)

    return rows


def _zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray:
    # alpha is pre-validated: finite, strictly positive.
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** alpha)
    return weights / weights.sum()
```

Cache bypass:

```python
def _gsp_cache_path(...):
    # Unchanged.

def sample_generated_shared_prefix_requests(...):
    cache_enabled = (
        group_distribution == "uniform"
        and range_ratio == 1
        and not send_routing_key
        and num_turns == 1
    )
    if cache_enabled and cache_file.exists():
        return load(cache_file)
    rows = _generate(...)
    if cache_enabled:
        save(cache_file, rows)
    return rows
```

CLI registration sketch:

```python
parser.add_argument(
    "--gsp-group-distribution",
    type=str,
    choices=["uniform", "zipf"],
    default="uniform",
    help="...rank-based Zipf formula...",
)
parser.add_argument(
    "--gsp-zipf-alpha",
    type=_finite_positive_float,
    default=None,
    help="Zipf exponent; required when --gsp-group-distribution=zipf. Must be finite and > 0.",
)
```

`from_args` performs the cross-flag validation: zipf requires alpha; uniform forbids alpha.

### Relevant References

- `python/sglang/bench_serving.py` — argparse setup for GSP flags, dataset dispatch via `get_dataset`, seed handling, LoRA `skewed` (exponential, NOT rank-based — do not reuse).
- `python/sglang/benchmark/datasets/__init__.py` — `DATASET_MAPPING` and `get_dataset` entry point.
- `python/sglang/benchmark/datasets/common.py` — `DatasetRow`, `BaseDataset`, `compute_random_lens`, `gen_prompt`.
- `python/sglang/benchmark/datasets/generated_shared_prefix.py` — `GeneratedSharedPrefixDataset` dataclass, `sample_generated_shared_prefix_requests` generator, current cache key/condition.
- `python/sglang/benchmark/datasets/random.py`, `python/sglang/benchmark/datasets/mooncake.py` — examples of `from_args` pattern for mode-specific args.
- `python/sglang/test/bench_one_batch_server_internal.py` — uses `get_dataset` indirectly; no direct import of `sample_generated_shared_prefix_requests`, so signature extension is safe with default-valued keyword args.
- `test/registered/bench_fn/test_benchmark_datasets_api.py` — CPU-only test pattern, lightweight tokenizer fixture.
- `docs/developer_guide/bench_serving.md` — existing GSP doc style and section to extend.
- `docs_new/docs/developer_guide/contribution_guide.mdx` — scoped changes, pre-commit, branch/PR conventions.

## Dependencies and Sequence

### Milestones

1. Milestone A: Codebase confirmation and contract lock.
   - Phase A.1: Re-read `sample_generated_shared_prefix_requests` end-to-end, confirm the exact shuffle path used today (`random.shuffle` vs `np.random.shuffle`) and the exact relationship between slot index `i`, group index, and `questions[i]`. This locks the behavior the uniform branch must preserve.
   - Phase A.2: Confirm whether any caller other than `bench_serving.py` constructs `GeneratedSharedPrefixDataset` directly. If yes, default-valued new fields are mandatory for source-level compatibility.

2. Milestone B: CLI surface.
   - Phase B.1: Add the two flags to `bench_serving.py` argparse with help text containing the Zipf formula and alpha constraint.
   - Phase B.2: Implement `_finite_positive_float` (or equivalent) argparse type for alpha. Validate the cross-flag rule (zipf↔alpha) in `GeneratedSharedPrefixDataset.from_args`.

3. Milestone C: Dataset and generator.
   - Phase C.1: Extend `GeneratedSharedPrefixDataset` dataclass with `group_distribution` and `zipf_alpha`. Thread through `from_args` and the call into the generator.
   - Phase C.2: Extend `sample_generated_shared_prefix_requests` signature. Implement the Zipf branch using an isolated `numpy.random.default_rng(seed)`. Implement (or inline) `_zipf_group_probs`.
   - Phase C.3: Apply cache bypass: `cache_enabled` becomes `group_distribution == "uniform" and <existing condition>`.

4. Milestone D: Tests.
   - Phase D.1: Default-mode regression test (AC-1) — fixture pinning current shape.
   - Phase D.2: Probability helper test (AC-3) — exact reference vector for at least one `(N, alpha)` pair plus the LoRA-formula rejection assertion.
   - Phase D.3: Reproducibility tests (AC-4) — same seed equality, different seed inequality, isolated-RNG question-pool equality.
   - Phase D.4: Deterministic skew test (AC-5) — pin per-group counts for `num_groups=4, prompts_per_group=25, alpha=2.0, seed=0` (or equivalent).
   - Phase D.5: Cache-bypass test (AC-6) — pre-populate uniform cache file, run Zipf, assert result is fresh and cache file untouched.
   - Phase D.6: CLI validation tests (AC-2.2, AC-2.3) — argparse rejections.
   - Phase D.7: Ordering tests (AC-8) — `--gsp-ordered` semantics under Zipf.
   - Phase D.8: Total-rows + unique-prompt tests (AC-7).

5. Milestone E: Documentation.
   - Phase E.1: Update `docs/developer_guide/bench_serving.md` GSP section.
   - Phase E.2: Verify `--help` output covers AC-9.

6. Milestone F: Quality gates and PR.
   - Phase F.1: Run targeted tests; capture command and output for PR.
   - Phase F.2: Run `pre-commit run --files <changed files>`; fix any findings.
   - Phase F.3: Push to feature branch; open draft PR with problem statement, API rationale referencing existing `bench_serving` dataset patterns (random.py / mooncake.py from_args), backward-compatibility summary, testing performed, and any deferred follow-ups.

Dependencies: Milestone A precedes all others. Milestone B and Milestone C can be implemented after A in either order, but the test fixtures in Milestone D depend on C. Milestone E depends on B (final help text) and C (final behavior). Milestone F depends on D and E.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via Codex / read-only exploration).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Re-read `sample_generated_shared_prefix_requests` and confirm exact shuffle path, slot→group mapping, and existing cache gating. Confirm no out-of-tree caller constructs `GeneratedSharedPrefixDataset` directly. | AC-1 | analyze | - |
| task2 | Add `--gsp-group-distribution` and `--gsp-zipf-alpha` argparse registration in `python/sglang/bench_serving.py`, including `--help` text with formula + alpha constraint and the `_finite_positive_float` type validator. | AC-2.1, AC-2.2, AC-9 | coding | task1 |
| task3 | Extend `GeneratedSharedPrefixDataset` dataclass with `group_distribution` and `zipf_alpha`; thread through `from_args`; perform cross-flag validation (zipf↔alpha) and reject alpha NaN/inf/≤0 in `from_args`. | AC-2.3, AC-7 | coding | task2 |
| task4 | Extend `sample_generated_shared_prefix_requests` signature with default-valued kwargs and implement Zipf branch using isolated `numpy.random.default_rng(seed)` and an auditable `_zipf_group_probs` helper (inline or module-private). | AC-3, AC-4, AC-7, AC-8 | coding | task3 |
| task5 | Apply cache-bypass: gate the existing read/write on `group_distribution == "uniform"` in addition to existing conditions. Do not change the uniform cache filename. | AC-6 | coding | task4 |
| task6 | Add CPU-only deterministic tests to `test/registered/bench_fn/test_benchmark_datasets_api.py` covering uniform regression, helper math, reproducibility, skew, cache bypass, CLI validation, total-rows/unique-prompt, and ordering. | AC-1, AC-2.2, AC-2.3, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-10 | coding | task4, task5 |
| task7 | Update `docs/developer_guide/bench_serving.md` with Zipf subsection (flags, formula, alpha range, example command, cache-bypass note, disclaimer). Confirm `--help` text mirrors the doc on the relevant points. | AC-9 | coding | task4 |
| task8 | Run targeted tests (`python -m unittest test.registered.bench_fn.test_benchmark_datasets_api`) and `pre-commit run --files <changed files>`; capture commands and output for PR description. | AC-10 | coding | task6, task7 |

## Claude-Codex Deliberation

### Agreements

- The feature must be opt-in via a `--gsp-*` flag and must keep default GSP behavior byte-equivalent for fixed seed and args.
- Zipf math is rank-based `1 / rank ** alpha`. The LoRA `skewed` exponential `alpha ** -i` helper must not be reused.
- The dataset cache must not silently return a uniform-mode payload for a Zipf invocation or vice versa.
- Tests must be deterministic CPU-only and live alongside existing benchmark dataset tests in `test/registered/bench_fn/test_benchmark_datasets_api.py`.
- Documentation in `docs/developer_guide/bench_serving.md` must include the Zipf formula, the alpha constraint, an example command, and a non-production-trace disclaimer.
- The Zipf branch must use an isolated `numpy.random.Generator` so it does not perturb the existing global random state that the question/prompt generation depends on.
- Tests for skew under Zipf must pin exact per-group counts for a fixed `(N, prompts_per_group, alpha, seed)` tuple rather than use a probabilistic threshold.
- alpha validation must reject `NaN`, `±inf`, and non-positive values.

### Resolved Disagreements

- Total request count under Zipf: Codex initially recommended `--num-prompts` as authoritative with `--gsp-prompts-per-group` redefined as a per-group suffix pool. User chose to keep the legacy formula `total = num_groups × prompts_per_group × num_turns` authoritative; `--num-prompts` continues to be ignored by GSP as today. Rationale: this minimizes the contract change, keeps uniform and Zipf totals identical (so users can A/B compare without rescaling), and avoids redefining the meaning of `--gsp-prompts-per-group`.
- Cache strategy: Codex offered "extend cache key" vs "bypass cache entirely"; user chose bypass-entirely for Zipf. Rationale: simplest, safest, removes the risk that a future param affecting Zipf sampling silently collides with an existing cache entry. Uniform-mode cache behavior is untouched.
- `--gsp-ordered` under Zipf: Codex offered reject/disable, document, or sort-by-rank. User chose to keep the same semantics as uniform mode (preserve sampling order; no shuffle when `--gsp-ordered`). Rationale: consistent user mental model across modes; ordering by rank would be a different and unrelated feature.
- `alpha == 0`: Codex recommended reject. User chose reject. Rationale: although mathematically `1/rank^0 = 1` is uniform, accepting it silently is a footgun; rejecting forces the user to pick `--gsp-group-distribution uniform` explicitly.
- Group rank assignment: Locked to rank-by-index (rank 1 = group index 0) based on Codex recommendation and the draft's wording "group ranks begin at 1, lower-ranked groups are hotter". No randomized rank.
- AC-1 equality definition: Locked to observable equality on (`prompt`, `prompt_len`, `output_len`, `routing_key`, order), not raw object identity. Codex flagged that strict byte-for-byte object equality may fail on serialization metadata; observable equality is sufficient to express backward compatibility.
- RNG isolation: Locked to `numpy.random.default_rng(seed)` for the Zipf branch. Codex recommended this; user did not contradict. Rationale: prevents Zipf sampling from perturbing the global `random`/`numpy.random` state used by uniform-mode question and prompt generation, which preserves the AC-4 invariant that uniform-mode and Zipf-mode share identical generated question pools for the same seed.

### Convergence Status

- Final Status: `converged`
- Convergence rounds executed: 2 (Codex first-pass analysis + Codex second-pass reasonability review). User decisions resolved all `REQUIRED_CHANGES` from the second pass.

## Pending User Decisions

All decisions are resolved. The list below documents resolved decisions for traceability; none are `PENDING`.

- DEC-1: Authoritative request-count contract under Zipf.
  - Claude Position: Originally leaned toward `--num-prompts` authoritative for clarity.
  - Codex Position: Originally recommended `--num-prompts` authoritative with `--gsp-prompts-per-group` redefined as suffix pool size.
  - Tradeoff Summary: Codex's option offered cleaner workload modeling; the legacy formula option preserves existing CLI mental model and keeps uniform/Zipf totals identical.
  - Decision Status: User chose **Legacy formula authoritative**: `total = num_groups × prompts_per_group × num_turns`. `--num-prompts` ignored by GSP, same as today.

- DEC-2: Meaning of `--gsp-prompts-per-group` in Zipf mode.
  - Claude Position: Follow from DEC-1.
  - Codex Position: Suffix pool size if `--num-prompts` authoritative; unchanged if legacy formula authoritative.
  - Tradeoff Summary: N/A — consequential.
  - Decision Status: **Unchanged from uniform mode** (consequence of DEC-1). `--gsp-prompts-per-group` still contributes to the total slot count formula. Per-group request counts under Zipf vary because group selection is sampled, but the total remains `N × P × T`.

- DEC-3: Suffix reuse policy when a hot group is sampled many times.
  - Claude Position: Generate one unique question per slot, independent of sampled group.
  - Codex Position: Generate-on-demand per sampled group with a monotonic per-group counter (when `--num-prompts` authoritative).
  - Tradeoff Summary: Under DEC-1 = legacy, the suffix pool is already sized to total slots, so each slot trivially gets a unique suffix without per-group cycling.
  - Decision Status: **One unique question per slot** (consequence of DEC-1). The existing per-slot question generation in `sample_generated_shared_prefix_requests` continues to produce `N × P × T` unique questions; the Zipf branch only changes the group index paired with each slot.

- DEC-4: Group rank assignment.
  - Claude Position: Rank by group index (rank 1 = group 0).
  - Codex Position: Same — rank by group index for simplicity and reproducibility.
  - Tradeoff Summary: Randomized rank would shuffle which group is hot per seed; rank-by-index keeps the mapping inspectable and matches the draft's wording.
  - Decision Status: **Rank by group index** (Claude/Codex agreement; not raised to user). Group `0` has rank `1` and is the hottest.

- DEC-5: `--gsp-ordered` interaction with Zipf.
  - Claude Position: Keep same meaning as uniform.
  - Codex Position: Either reject under Zipf or sort-by-rank.
  - Tradeoff Summary: Same-meaning is consistent across modes; reject/sort is more explicit about cache-locality implications.
  - Decision Status: User chose **Same meaning as today**: `--gsp-ordered` preserves sampling order; absence triggers the existing shuffle path.

- DEC-6: `alpha == 0`.
  - Claude Position: Reject.
  - Codex Position: Reject.
  - Tradeoff Summary: Mathematically alias-to-uniform; in practice a footgun.
  - Decision Status: User chose **Reject `alpha <= 0` (and NaN/inf)** with clear error.

- DEC-7: Cache strategy under Zipf.
  - Claude Position: Bypass cache for Zipf.
  - Codex Position: Bypass cache for Zipf.
  - Tradeoff Summary: Bypass is simplest and removes silent-collision risk; extending the key is feasible but couples cache invariants to future sampling parameters.
  - Decision Status: User chose **Bypass cache entirely for Zipf**.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "DEC-", or similar workflow markers. These belong in this plan document, not in the resulting codebase or commit messages.
- Use descriptive, domain-appropriate naming in code:
  - Argparse flag names: `--gsp-group-distribution`, `--gsp-zipf-alpha`.
  - Dataclass field names: `group_distribution`, `zipf_alpha`.
  - Helper name: `_zipf_group_probs` (module-private) is acceptable; inline is also acceptable per the lower-bound option.
  - Test method names: `test_gsp_zipf_*` / `test_gsp_uniform_*` aligned with existing `test_gsp_*` patterns in `test/registered/bench_fn/test_benchmark_datasets_api.py`.
- The `.claude/rules/speculative-naming.md` rule applies to speculative decoding code, not to bench_serving; do not import its conventions into this PR.

### Out-of-Scope (deferred follow-ups, to mention in PR description)

- Extending Zipf to non-GSP datasets (random, sharegpt, mooncake) — separable, possibly desirable later.
- A workload-summary log line that records the chosen distribution, alpha, and the hottest-group count — small ergonomics improvement; can be added without changing the contract.
- Server-backed end-to-end test exercising Zipf through `run_benchmark` in `test_bench_serving_functionality.py` — adds CI cost; revisit if a regression slips past the unit tests.



--- Original Design Draft Start ---

# PR Task: Add Zipfian Shared-Prefix Sampling to `bench_serving`

## Objective

Implement a general-purpose Zipfian/power-law prefix-popularity workload mode for
`python -m sglang.bench_serving`.

The motivating gap is that `generated-shared-prefix` can create reusable long
prefixes, but its existing workload shape assigns the same number of prompts to
each prefix group. Real cache-oriented serving studies often need a hot-head and
long-tail prefix distribution, where a few shared prefixes are requested often
and many are requested infrequently.

The likely home for this capability is an optional extension to the existing
`generated-shared-prefix` dataset mode, rather than a new benchmark script or a
new dataset name. Confirm this after studying the codebase conventions described
below, and keep the final implementation closely scoped.

The deliverable is a polished pull request suitable for upstream review:
implementation, tests, user-facing documentation, and a clear PR description.

## Start Here: Repository and Contribution Conventions

Before designing or editing code, read and follow:

- `docs_new/docs/developer_guide/contribution_guide.mdx`

In particular, follow its expectations around scoped changes, adding tests for
new features, documentation updates, formatting/pre-commit checks, branch/PR
workflow, and avoiding unnecessary changes to performance-sensitive runtime
paths.

This is benchmark workload-generation functionality, not a model execution
change. Keep the implementation in the benchmark/dataset ownership boundary
unless the existing architecture clearly requires otherwise.

## Required Discovery Before Implementation

Develop a comprehensive understanding of how SGLang maintainers structure
`bench_serving` dataset modes before choosing the API or changing behavior.
Read the relevant code end to end rather than editing immediately.

At minimum, inspect:

- `python/sglang/bench_serving.py`
  - CLI argument registration and validation.
  - Dataset loading.
  - Random seeds and reproducibility.
  - Request scheduling, `--request-rate`, and `--max-concurrency`.
  - Warmup and `--flush-cache` behavior.
  - Existing LoRA request-distribution options, including the currently named
    Zipf-related flag, as prior art to understand rather than automatically copy.
- `python/sglang/benchmark/datasets/__init__.py`
- `python/sglang/benchmark/datasets/common.py`
- Every dataset implementation in `python/sglang/benchmark/datasets/`, with
  particular attention to:
  - `generated_shared_prefix.py`
  - `random.py`
  - `mooncake.py`
  - dataset classes that expose mode-specific CLI behavior or per-request data.
- `docs/developer_guide/bench_serving.md`
- `docs/developer_guide/benchmark_and_profiling.md`
- `test/registered/bench_fn/test_benchmark_datasets_api.py`
- `test/registered/bench_fn/test_bench_serving_functionality.py`
- `python/sglang/test/bench_one_batch_server_internal.py`, to determine whether
  related GSP interfaces need corresponding compatibility updates.
- `benchmark/hicache/` only as related background on prefix-reuse benchmarking;
  do not move the primary implementation there unless investigation provides a
  strong reason.

Also use repository search and, if useful, git history to learn how prior
benchmark dataset options were named, validated, tested, and documented. In the
PR description, briefly explain the existing patterns the implementation follows.

## Intended Feature Semantics

Add an opt-in distribution policy for selecting shared-prefix groups in the
`generated-shared-prefix` workload.

The current/default GSP behavior must remain backward compatible. Without new
flags, existing commands must generate the same style of uniform workload as
before.

For the new mode, model prefix popularity using a conventional rank-based
power-law/Zipf distribution:

```text
weight(group_rank) = 1 / group_rank**alpha
probability(group_rank) = weight(group_rank) / sum(all weights)
```

where group ranks begin at `1`, and lower-ranked groups are hotter.

The implementation should make it possible to express:

```text
request = sampled_shared_prefix_group + unique_question_or_suffix
```

so repeated requests to the same group reuse the group prefix while retaining
per-request variation after that prefix.

## API Design Guidance

Do not treat proposed flag names as mandatory until you have compared them with
the established CLI style. A natural initial design to evaluate is:

```bash
--gsp-group-distribution uniform|zipf
--gsp-zipf-alpha FLOAT
```

Requirements for the chosen CLI:

- Existing GSP commands keep their behavior without opting into Zipf sampling.
- The Zipf/power-law formula and the meaning of `alpha` are unambiguous in
  `--help` text and documentation.
- Invalid argument combinations or invalid `alpha` values fail clearly.
- Seeds produce reproducible workload generation.
- The semantics of `--num-prompts`, `--gsp-num-groups`, and
  `--gsp-prompts-per-group` are coherent in Zipf mode.
- Do not silently ignore an existing user argument. If an argument is
  incompatible, reject it clearly; if it has a new interpretation, document it
  explicitly.
- Preserve or deliberately define how `--gsp-ordered` and default shuffling
  interact with Zipf-generated requests.
- Preserve existing cache warmup/flush behavior unless a scoped change is
  necessary. Document how a Zipf workload starts cold or interacts with the
  existing benchmark warmup.

One issue to handle carefully: the existing LoRA `skewed` code is relevant prior
art, but verify its mathematical distribution and intended contract before
reusing its implementation or terminology. The new GSP behavior should be
precisely documented as Zipfian if it uses rank-based Zipf weights.

## Scope Control

Keep the PR focused on Zipfian shared-prefix workload generation and the
supporting CLI/docs/tests.

Do not broaden the PR into unrelated cache instrumentation, server runtime
changes, strict token-ID length modes, or a rework of all benchmark datasets
unless investigation shows a necessary compatibility issue. If an adjacent
improvement is valuable but separable, mention it as a possible follow-up in
the PR instead.

## Implementation Considerations

The exact implementation should follow the codebase after inspection, but it
will likely involve:

1. Adding optional GSP distribution configuration to the CLI/parser path in
   `python/sglang/bench_serving.py`.
2. Carrying that configuration through `GeneratedSharedPrefixDataset`.
3. Extending `sample_generated_shared_prefix_requests` to create a Zipf-sampled
   request assignment to prefix groups while retaining existing uniform behavior.
4. Ensuring any dataset caching key is correct for the new distribution
   configuration, so workloads from different policies or parameters cannot be
   accidentally reused as one another.
5. Printing or recording enough workload configuration for results to be
   interpretable and reproducible.
6. Updating documentation with the new flags, semantics, and a concise example.

Be especially careful with the existing generated-dataset on-disk cache. A new
sampling policy or parameter must not collide with a cache entry created under a
different policy.

## Testing Expectations

Add focused, deterministic coverage appropriate for this feature. At minimum,
cover:

- Existing uniform/default GSP behavior remains supported.
- Zipf mode produces the expected number of request rows.
- Zipf group assignment is reproducible for a fixed seed.
- Zipf mode actually produces a skewed group-frequency result under a
  deterministic test setup, without writing a flaky statistical assertion.
- Validation of unsupported/invalid new argument values.
- Any changed dataset-cache key or caching behavior.
- Any interaction changed for multi-turn GSP or request ordering.

Prefer lightweight CPU-only tests in the existing benchmark dataset test suite
where possible, particularly:

- `test/registered/bench_fn/test_benchmark_datasets_api.py`

Only extend server-backed functionality tests when behavior cannot be
meaningfully covered at the dataset/unit level or an existing integration
contract changes.

Run the targeted tests relevant to modified code, and run the applicable
formatting/lint checks described in
`docs_new/docs/developer_guide/contribution_guide.mdx`. Report exactly what was
run in the PR.

## Documentation Expectations

Update the user-facing benchmark documentation, likely
`docs/developer_guide/bench_serving.md`, to include:

- The new GSP distribution option.
- The mathematical/operational meaning of Zipf sampling.
- The meaning and acceptable range of `alpha`.
- An example command using multiple shared-prefix groups.
- A short note that this controls prefix popularity/reuse shape; it does not by
  itself claim a production trace or guarantee an observed cache-hit rate.

If contributor-facing or new documentation conventions in `docs_new/` indicate
that another source document should also be updated, follow the established
documentation workflow rather than duplicating content inconsistently.

## Acceptance Criteria

The PR is ready when:

- A user can opt into a documented Zipfian shared-prefix workload through
  `python -m sglang.bench_serving`.
- Existing GSP behavior is unchanged by default.
- Workload generation is deterministic under a fixed seed.
- CLI semantics are clear, validated, and consistent with other benchmark
  dataset modes.
- Generated dataset caching cannot return a workload generated with the wrong
  distribution configuration.
- Unit or functional tests cover the new behavior and backward compatibility.
- Documentation and CLI help explain the feature accurately.
- The implementation is narrowly scoped and follows
  `docs_new/docs/developer_guide/contribution_guide.mdx`.

## PR Delivery

Work through the feature end to end:

1. Create an appropriately named feature branch rather than committing on
   `main`.
2. Implement the smallest maintainable version consistent with repository
   conventions.
3. Add and run tests.
4. Update documentation.
5. Run relevant formatting/pre-commit checks where feasible.
6. Commit and push the work.
7. Open a draft pull request with:
   - A concise problem statement.
   - The selected API and why it fits existing `bench_serving` dataset patterns.
   - A summary of backward compatibility behavior.
   - Testing performed.
   - Any consciously deferred follow-up work.
--- Original Design Draft End ---
