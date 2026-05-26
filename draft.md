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