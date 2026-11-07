.. _torch_tensorrt_agentic_py:

torch_tensorrt.agentic
======================

.. automodule:: torch_tensorrt.agentic

.. note::

   This module is experimental; its API may change.

An optimization session turns Torch-TensorRT into a measured, agent-guided
compiler. You set the goal and the budget; every result comes with evidence.

* **You provide** a workload (the model and representative inputs), a latency or
  throughput goal, accuracy and memory limits, and a time and trial budget.
* **The session** establishes a validated baseline, profiles it, and repeats:
  a planner reads the evidence and proposes one experiment, a fresh worker
  process builds, checks, measures and profiles the candidate, and the session
  keeps it only if the gain is real. Every experiment is checkpointed.
* **You receive** the best validated artifact, a before/after comparison, a
  reproducible recipe and the full experiment record. If nothing beats the
  baseline, the baseline is kept.

Inference runs the exported artifact: it needs no agent, planner or session.

Quick start
-----------

.. code-block:: python

    from torch_tensorrt.agentic import (
        Budget, Constraints, Goal, HeuristicPlanner, OptimizationSession, Workload,
    )

    session = OptimizationSession(
        Workload("my_project/cases.py:create_case", {"batch": 8}),
        directory="sessions/my-model",
        goal=Goal(metric="latency_ms", min_improvement=0.02),
        constraints=Constraints(accuracy={"cosine_similarity": (">=", 0.999)}),
        budget=Budget(max_trials=10, max_time_s=3600),
        planner=HeuristicPlanner(),
    )
    result = session.run()
    print(result.decision, result.recipe)
    print(result.report())

``create_case`` returns a :class:`Case`. For a module with fixed example
inputs, :class:`ModuleCase` is enough:

.. code-block:: python

    # my_project/cases.py
    import torch
    from torch_tensorrt.agentic import ModuleCase

    def create_case(batch: int = 8):
        model = build_model().eval().cuda()
        inputs = [torch.randn(batch, 3, 224, 224, device="cuda")]
        return ModuleCase(model, inputs, compile_options={"min_block_size": 1})

The workload is rebuilt in every worker process from this factory, so its
arguments must be JSON data and constructing it should be cheap.

Running the same session again in the same directory resumes it: completed
trials are reused, an interrupted trial is evaluated again, and the final
confirmation always measures anew. A larger budget may be passed on resume.

The four optimization layers
----------------------------

The planner can only request changes through *tools*: named options with a
finite list of allowed values, declared by the workload. Every proposal is
validated against them, and the recipe of a candidate is the incumbent's recipe
with the proposed changes. Tools are grouped by the layer they act on, and
each names the profiler signals it can relieve:

.. list-table::
   :widths: 22 30 48
   :header-rows: 1

   * - Bottleneck signal
     - Layer
     - Example tools
   * - ``fallback``: PyTorch partitions or many engines
     - ``graph`` (graph and regions)
     - ``graph.min_block_size``
   * - ``precision``, ``dominant_kernel``
     - ``engine`` (TensorRT build)
     - ``engine.precision``, ``engine.optimization_level``,
       ``engine.tiling_optimization_level``
   * - ``dominant_kernel``
     - ``kernels`` (kernel generation)
     - ``kernels.<region>``: compile a region with a
       :class:`~torch_tensorrt.region.KernelGenerator` kernel
   * - ``launch_overhead``, ``host_syncs``
     - ``runtime`` (execution)
     - ``runtime.cudagraphs``, ``runtime.pre_allocated_outputs``

Workloads add their own ``workload.*`` tools; the decode workload below adds
its KV-cache lowering. :data:`GENERIC_TOOLS` lists the generic ones. For
:class:`ModuleCase`, ``compile_options`` become the defaults of the tools that
control them, and ``regions={"name": "submodule.path"}`` with
``generator=KernelGenerator(...)`` adds a ``kernels.name`` tool.

The measured loop
-----------------

1. **Observe.** The reference outputs are computed once (in a worker) and the
   baseline recipe is evaluated. A baseline that fails, or misses the accuracy
   or memory limits, ends the session: nothing could be compared with it.
2. **Profile.** Each evaluation traces a few calls and reports, per work unit,
   the GPU time against the untraced wall time (how long the GPU waits for the
   host), kernel launches, device-to-host synchronizations and the top kernels,
   plus the engines, PyTorch partitions and plugin regions of the candidate.
   These become bottleneck *signals*.
3. **Improve.** The planner reads the evidence (goal, constraints, budget, the
   baseline and the incumbent with their profiles and signals, every past trial
   with its outcome, and the tool catalog) and proposes changes.
4. **Evaluate.** A fresh worker process builds the candidate, checks its outputs
   against the reference before and after timing (so replayed CUDA graphs are
   checked too), times it with the same protocol as every other candidate,
   measures its memory and profiles it. Failures at any stage are recorded as
   evidence.
5. **Select.** The candidate is promoted only if it passes the accuracy and
   memory constraints and improves the goal metric by more than
   ``Goal.min_improvement`` with non-overlapping quartiles.

The search stops at the goal's ``target``, when the trial or time budget is
used, after ``Budget.patience`` trials without promotion, after three invalid
proposals in a row, or when the planner stops. The best candidate is then
measured again against the baseline in ``confirm_repeats`` fresh, back-to-back
worker pairs and kept only if the median paired gain still exceeds
``min_improvement``. The winner is exported to ``directory/artifact`` with its
``recipe.json``.

Planners
--------

:class:`HeuristicPlanner` follows the measured bottleneck: it tries the tools
that relieve the incumbent's strongest signal first, then every other option,
one change at a time, and never proposes an option that was already explored.

:class:`AgentPlanner` asks a coding agent (the logged-in ``claude`` or ``codex``
CLI, or a trusted ``module:attribute`` adapter, as for
:class:`~torch_tensorrt.region.KernelGenerator`) to choose up to ``max_changes``
tool changes from the evidence, with a JSON-schema reply. If the provider fails,
its ``fallback`` planner (a :class:`HeuristicPlanner` by default) proposes
instead and the trial records why. Neither planner can build, measure or accept
a candidate.

Inspecting a session
--------------------

.. code-block:: bash

    python -m torch_tensorrt.agentic status sessions/my-model
    python -m torch_tensorrt.agentic report sessions/my-model
    python -m torch_tensorrt.agentic resume sessions/my-model --max-trials 20

The session directory holds ``session.json`` (identity, budget, environment),
``reference.pt``, one directory per trial under ``trials/`` (the planner's
``evidence.json`` and ``proposal.json``, the worker's ``request.json``,
``result.json`` and ``worker.log``, and the gate's ``decision.json``), the
``confirm/`` measurements, ``artifact/``, ``report.md``, ``report.json`` and an
``events.jsonl`` log. A directory holds one session: resuming with a different
workload, goal, constraints or baseline, or after the software or GPU changed,
is refused.

Example: Qwen3-0.6B decode
--------------------------

``tools/llm/run_agentic_decode.py`` optimizes batch-1 decode of a Hugging Face
causal LM with the workload in ``tools/llm/agentic_decode.py``. Its unit is one
decoded token; decoding is teacher-forced on a fixed text and checked against an
FP32 eager reference at every position (top-1 agreement and KL divergence). Its
baseline is the documented ``tools/llm`` configuration (FP16, ``static_v2``
cache). The workload adds ``workload.cache``, which can select
``static_cache_v3``: a shape-static decode engine without the per-layer host
synchronizations of ``static_v2``.

.. code-block:: bash

    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode --max-trials 8
    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode-agent \
        --max-trials 8 --planner agent

On an RTX 5090 (TensorRT 11.4, Python runtime), both planners found
``static_cache_v3`` from the ``host_syncs`` signal and then turned off FP32
accumulation: the heuristic session confirmed 2.92 ms to 1.25 ms per token and
the agent session 2.92 ms to 1.24 ms (about 2.35x tokens per second), with
unchanged top-1 agreement. See ``tools/llm/README.md`` for the full results.

Limitations
-----------

* Candidates are measured on the workload you provide: the result is the best
  found within the budget for that workload, hardware and software, not a
  global optimum.
* Workers run on the machine and GPU of the session; the session does not lock
  clocks or check for other GPU users. Use an exclusive GPU for stable results.
* Memory is the process's GPU memory from NVML where the process ID is visible,
  otherwise PyTorch's peak allocation, which excludes TensorRT's own memory.
* The workload factory and an agent adapter are trusted code. Agent replies are
  validated against the tool catalog, but tool implementations run as written.
* The artifact is what the workload's ``Candidate.export`` saves. Runtime tools
  (``runtime.*``, such as CUDA graphs) are execution modes, not part of a saved
  program: apply them when deploying, as ``recipe.json`` records.

API reference
-------------

.. autoclass:: OptimizationSession
   :members: run, resume, judge

.. autoclass:: SessionResult
   :members:

.. autoclass:: Workload
   :members: load

.. autoclass:: Goal

.. autoclass:: Constraints

.. autoclass:: Budget

.. autoclass:: Timing

.. autoclass:: Case
   :members:

.. autoclass:: Candidate
   :members:

.. autoclass:: ModuleCase

.. autoclass:: Tool

.. autodata:: GENERIC_TOOLS
   :no-value:

.. autoclass:: Planner
   :members: propose

.. autoclass:: HeuristicPlanner

.. autoclass:: AgentPlanner

.. autoclass:: Proposal

.. autofunction:: compare_outputs

.. autofunction:: describe_module

.. autoexception:: SessionError
