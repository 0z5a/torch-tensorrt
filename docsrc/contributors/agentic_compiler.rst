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
* **Promotion needs a real gain.** A candidate must beat the incumbent by more
  than ``Goal.min_improvement`` with non-overlapping latency quartiles. Because
  trials run at different times, the final winner is confirmed in fresh,
  back-to-back baseline/best pairs; a winner that does not hold up is dropped
  and the baseline is kept.
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
