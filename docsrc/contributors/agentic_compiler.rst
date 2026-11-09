.. _agentic_compiler:

Agentic Optimization Sessions
=============================

``torch_tensorrt.agentic`` (see :ref:`torch_tensorrt_agentic_py`) searches a
workload's compiler configuration toward a measured goal. This note describes
its structure and the invariants the implementation keeps.

Components
----------

``_session.OptimizationSession`` (the controller)
    Owns the session directory, budgets, checkpoints and the acceptance rules.
    It asks the planner for proposals, validates them, launches workers, judges
    results and finally confirms and exports the winner.

``_planner`` (the planners)
    Map JSON *evidence* to a ``Proposal``: changes to the incumbent recipe, or
    stop. The ``AgentPlanner`` reuses the provider layer of region kernel
    generation (``region_compilation/_generation/_providers.py``).

``_tools`` (the compiler tools)
    A ``Tool`` is a named option (``<layer>.<option>``) with finite allowed
    values, a default and the profiler signals it can relieve. A *recipe* maps
    tool names to non-default values, so equal configurations have equal
    digests.

``_trial`` and ``_evaluator`` (the independent evaluator)
    ``python -m torch_tensorrt.agentic._trial REQUEST.json`` rebuilds the case
    from its ``Workload`` factory and runs one job: ``reference`` (outputs and
    case description), ``describe`` (description only) or ``trial`` (build,
    check, time, profile and optionally export one recipe).

``_profile`` (diagnosis)
    Turns a ``torch.profiler`` trace and the candidate's structure into
    per-unit statistics and bottleneck signals.

``_report``
    Writes ``report.md`` and ``report.json``.

Invariants
----------

* **The planner only proposes.** Proposals are validated against the tool
  catalog (unknown tools, disallowed values, no-op changes and repeated recipes
  are rejected and recorded as invalid), and only the controller's gate promotes
  a candidate, based on what the worker measured.
* **One candidate per process.** Every job runs in a fresh worker started with
  ``python -P``. Candidates cannot leak GPU memory, process-global lowering
  passes or converter registrations into each other, and a crash or a timeout
  (the process group is killed after ``trial_timeout_s``) becomes a failed
  trial.
* **One measurement protocol.** Every candidate is checked on its first call and
  again after timing, timed with the same ``Timing``, and judged against the
  same bounds. Accuracy uses the reference computed once per session.
* **Promotion needs a real gain.** Quartiles within one run are far tighter
  than the spread between processes (on an RTX 5090 decode, 0.02-0.9% within a
  run against up to 3% between processes of the same recipe), so they cannot
  show a gain of a few percent. The session measures that spread at the start
  (an A/A calibration), sets the threshold to a multiple of it, and races any
  candidate that clears it in interleaved incumbent/candidate worker pairs.
  Workers wait for an idle GPU and re-time unstable samples; an unstable
  measurement is never promoted. The final winner is confirmed in more
  interleaved baseline/best pairs; a winner that does not hold up is dropped and
  the baseline is kept.
* **Only fenced code changes.** Patch slots let a planner edit source inside
  ``EVOLVE-BLOCK`` fences. Edits must match exactly once inside a fence, the
  file must compile, and the result is stored by digest and named in the
  recipe. The evaluator, the reference, the accuracy bounds and the timing
  protocol are never editable.
* **Resuming never skips evaluation.** ``session.json`` records the session's
  identity (workload, goal, constraints, baseline changes, timing), the case's
  tool catalog and the environment. A run with a different identity, catalog or
  environment is refused; completed trials are replayed from their
  ``decision.json``, an interrupted trial is evaluated again, and the final
  confirmation always measures anew.

Adding tools
------------

Generic tools belong in ``_tools.GENERIC_TOOLS`` with a mapping in
``compile_settings`` or in ``ModuleCase``. Workload-specific options belong in
the case's ``tools()`` with the ``workload`` layer. Choose values in the order a
heuristic search should try them and name the signals they address; the
heuristic planner and the agent planner both rely on these declarations.

Findings from the decode validation
-----------------------------------

Validating sessions on Qwen3-0.6B decode surfaced two runtime behaviors, both
handled by the evaluator rather than hidden:

* ``tools/llm`` ``static_cache_v2`` slices the cache to the current length,
  which makes TensorRT synchronize with the host twice per layer and step and
  makes the engine impossible to capture in a CUDA graph. ``static_cache_v3``
  keeps all shapes static.
* The Python runtime's ``tensorrt::execute_engine`` custom op is declared
  without mutated arguments, so it returns clones of outputs that alias inputs:
  an in-place KV-cache engine still copies every cache per step, and a loop that
  feeds those clones back breaks the stable addresses a captured CUDA graph
  relies on. The decode workload therefore keeps passing the same cache tensors.
  The Python runtime now runs such engines directly in plain eager execution,
  returning the caches without a copy; this removed 56 copies per token (about
  4.5% of decode latency). Under ``torch.compile``, ``torch.jit.trace``,
  ``torch.func`` transforms, any dispatch mode (export, ``make_fx``, fake or
  functional tensors) or inputs that require gradients, the op still runs.

Kernel-level findings on Qwen3 decode
-------------------------------------

Fusing whole decoder blocks into Triton kernels inside the TensorRT engine
(``tools/llm/qwen3_regions.py``, through ``fused_kernel`` regions) took
Qwen3-0.6B decode from 1.18 to 1.02-1.03 ms/token on an RTX 5090 (339 to 202
kernels per token). What carried over to the region path:

* The gain is fusion, not GEMV bandwidth: TensorRT's in-model GEMVs are within
  0.3-0.5 us of the Triton kernels; each block removes small kernels (RMSNorm,
  residual add, SiLU, q/k norm, RoPE, q scaling) of 0.5-0.9 us plus a 0.19 us
  gap each. Under CUDA graphs a plugin boundary costs the same gap as a Myelin
  kernel boundary, but blocks Myelin fusion across it.
* Weight streams must load with an L2 ``evict_first`` policy; otherwise they
  evict the KV cache and TensorRT's attention slows (+97 us/token for the
  lm_head block alone).
* The QDP AOT compile path drops Triton's pointer-alignment specialization:
  kernels need ``tl.multiple_of(..., 16)`` hints to get 128-bit loads. The
  plugin compile could pass ``tt.divisibility`` attributes instead.
* ``fused_kernel`` outputs follow production order in the region body; a body
  that produced q after k silently swapped the outputs (and wrote past a
  buffer). The API should check output order against the kernel signature.
* QDP launches support neither programmatic dependent launch nor cooperative
  grids, so kernels cannot overlap one another's tails and a persistent,
  grid-synchronized decoder (the hand-written "megakernel" approach) is out of
  reach; attention and the KV write must stay in TensorRT because plugins
  cannot alias inputs.

What we took from OpenEvolve and autoresearch
---------------------------------------------

`autoresearch <https://github.com/karpathy/autoresearch>`_ lets a coding agent
edit one training script under a fixed harness and keep a change only if the
metric improves; a human steers it through a Markdown brief.
`OpenEvolve <https://github.com/algorithmicsuperintelligence/openevolve>`_
re-implements AlphaEvolve: an archive of programs, diff-based edits confined by
``EVOLVE-BLOCK`` markers, cascade evaluation and structured evaluator feedback.
From them the session takes:

* **Code-level actions:** patch slots with fenced SEARCH/REPLACE edits, stored
  and de-duplicated by digest, judged by the same evaluator as knob changes.
* **A human-editable brief** (``program.md``) shown to the agent with every
  proposal, and a ``ledger.tsv`` written after every decision.
* **Structured evidence:** profiles, bottleneck signals (now including a
  memory-bandwidth roofline gap and copy overhead), paired gains and errors.
* **Revisiting rejected options** once the incumbent changes, since options
  interact.

It deliberately does not take:

* **Single-run fitness.** Both systems accept a candidate on one measurement;
  autoresearch's published ledger keeps a random-seed change as an
  "improvement", and OpenEvolve's flagship GPU-kernel example finished 3.2%
  slower than its baseline once its evaluator was fixed. The session's frozen,
  independent evaluator with calibrated, paired, interleaved acceptance is the
  part to keep.
* **LLM judges, novelty embeddings, large populations and template
  randomness:** promotion depends only on measured latency, accuracy and
  memory, and each trial here costs a TensorRT build, not milliseconds.
* **An agent that controls the loop.** The planner proposes; only the
  controller decides.
