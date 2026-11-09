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
its KV-cache lowering. A tool can declare ``requires``: settings of other tools
that its non-default values need (fused kernels that only run under CUDA
graphs, say). A recipe that breaks a requirement is refused before anything is
built, and the heuristic planner proposes the requirements together with the
option. :data:`GENERIC_TOOLS` lists the generic ones. For
:class:`ModuleCase`, ``compile_options`` become the defaults of the tools that
control them, and ``regions={"name": "submodule.path"}`` with
``generator=KernelGenerator(...)`` adds a ``kernels.name`` tool.

The measured loop
-----------------

1. **Observe.** The reference outputs are computed once (in a worker) and the
   baseline recipe is evaluated. A baseline that fails, misses the accuracy or
   memory limits, or is measured unstably ends the session: nothing could be
   compared with it.
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
   checked too), waits for an idle GPU, times it with the same protocol as
   every other candidate, measures its memory and profiles it. Failures at any
   stage are recorded as evidence.
5. **Select.** The candidate must pass the accuracy and memory constraints, be
   measured stably, and improve the goal metric by more than the noise
   threshold in its own run. It is then raced: fresh incumbent/candidate
   worker pairs run back to back in alternating order, and it is promoted only
   if every pair shows a gain and their median gain clears the threshold.

Measurement validity
^^^^^^^^^^^^^^^^^^^^

Samples within one run vary far less than the same recipe measured in two
processes: engine builds pick different tactics, clocks drift and other jobs
share the GPU. A gain of a few percent seen in one run is not evidence, so the
session (configured by :class:`Measurement`):

* runs ``calibration_runs`` extra fresh-process measurements of the baseline
  (an A/A test) and sets the promotion threshold to ``Goal.min_improvement`` or
  ``noise_factor`` times the observed spread, whichever is larger;
* races every candidate that clears the threshold in ``race_pairs``
  interleaved pairs against the incumbent before promoting it (use an even
  number: the incumbent runs first in the extra pair of an odd count);
* makes each worker wait for an idle GPU (at most 5% busy by nvidia-smi;
  ``quiet_gpu_s`` seconds in total, ``0`` turns the wait off) before timing,
  re-time up to ``retries`` times when the samples' interquartile range
  exceeds ``max_spread`` or the GPU was busy, and never promote a measurement
  that stays unstable;
* confirms the final winner against the baseline in ``confirm_repeats`` more
  interleaved pairs.

A resumed session reuses its calibration when it was made with the same
number of runs and measured a spread; the report says so when no calibration
run was accepted.

A candidate that declares ``bytes_per_unit`` (the decode workload counts its
weights and KV cache) also gets a memory-bandwidth roofline: the peak read
bandwidth is measured in the worker, and the report shows how close each
candidate runs to bandwidth-bound speed (the floor latency over its latency).
Below 80% a ``roofline_gap`` signal is raised: the heuristic planner tries the
tools that declare it first (in the decode workload, the fused-kernel blocks),
and the agent planner sees it as evidence.

The search stops at the goal's ``target``, when the trial or time budget is
used, after ``Budget.patience`` trials without promotion, after three invalid
proposals in a row, or when the planner stops. The winner is exported to
``directory/artifact`` with its ``recipe.json``. ``ledger.tsv`` lists every
trial as it is decided: parent, decision, metric, paired gain, roofline share,
accuracy, changes and the reason.

Code-level experiments
^^^^^^^^^^^^^^^^^^^^^^

Knobs only reach what their authors anticipated; the largest gains often need
new code. A case can declare :class:`PatchSlot` objects for source files it
builds from (a lowering pass, region kernels). Lines containing
``EVOLVE-BLOCK-START`` and ``EVOLVE-BLOCK-END`` fence the editable parts:

.. code-block:: python

    # my_project/kernels.py
    # EVOLVE-BLOCK-START
    BLOCK_N = 8
    # EVOLVE-BLOCK-END

    class MyCase(Case):
        def patch_slots(self):
            return [PatchSlot("kernels", "/abs/path/my_project/kernels.py",
                              "Launch parameters of the fused GEMV kernel")]

        def build(self, values):
            path = values["patch.kernels"] or "/abs/path/my_project/kernels.py"
            kernels = load_module(path, "my_kernels")
            ...

A planner proposes SEARCH/REPLACE edits. Each search text must occur exactly
once inside the fenced blocks of the incumbent's version of the file; the
edits must leave the fence lines unchanged and every block non-empty, and the
edited file must compile. The session stores it under its SHA-256 in
``directory/patches`` and records ``patch.<slot>`` in the recipe, so patched
candidates are reproducible, de-duplicated and judged by the same evaluator
and gate as knob changes; edits that restore the original file select the
original. The slot's original file is part of the recorded case, so editing
it between resumes is detected like any other workload change. The
evaluator, the reference and the accuracy bounds stay out of the planner's
reach.

A ``brief`` (Markdown) is written to ``directory/program.md`` and shown to the
agent planner with every proposal. Edit it between resumes to steer the
search, as with a research brief; it is replaced only when a different brief
is passed.

Planners
--------

:class:`HeuristicPlanner` follows the measured bottleneck: it tries the tools
that relieve the incumbent's strongest signal first, then every other option,
one change at a time. It never reverts to the baseline's options. Once every
option has been tried, it revisits options rejected under an earlier incumbent,
because options interact (CUDA graphs that did not pay under one KV cache can
pay under another).

:class:`AgentPlanner` asks a coding agent (the logged-in ``claude`` or ``codex``
CLI, or a trusted ``module:attribute`` adapter, as for
:class:`~torch_tensorrt.region.KernelGenerator`) to choose up to ``max_changes``
tool changes from the evidence, with a JSON-schema reply. If the provider fails,
its ``fallback`` planner (a :class:`HeuristicPlanner` by default) proposes
instead and the trial records why. When the workload declares patch slots,
the agent may also propose code edits; see `Code-level experiments`_. Neither
planner can build, measure or accept a candidate.

Inspecting a session
--------------------

.. code-block:: bash

    python -m torch_tensorrt.agentic status sessions/my-model
    python -m torch_tensorrt.agentic report sessions/my-model
    python -m torch_tensorrt.agentic resume sessions/my-model --max-trials 20

The session directory holds ``session.json`` (identity, budget, environment),
``reference.pt``, the baseline's ``calibration/`` runs, one directory per
trial under ``trials/`` (the planner's ``evidence.json`` and ``proposal.json``,
the worker's ``request.json``, ``result.json`` and ``worker.log``, the paired
``race/`` runs and the gate's ``decision.json``), patched sources under
``patches/``, ``program.md``, the ``confirm/`` measurements, ``artifact/``,
``ledger.tsv``, ``report.md``, ``report.json`` and an ``events.jsonl`` log. A directory holds one session: resuming with a different
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
synchronizations of ``static_v2``. For Qwen3 models with power-of-two
hidden and head sizes it also adds ``kernels.decode_blocks``, which compiles
whole decoder blocks (RMSNorm + GEMVs + activation or residual add) into the
engine as fused Triton kernels through
:func:`torch_tensorrt.region.fused_kernel` and requires ``static_v3``, FP16 and
CUDA graphs, and two patch slots that let an agent planner edit those kernels'
launch settings (``tools/llm/qwen3_regions.py``) and Triton code
(``tools/llm/qwen3_kernels.py``).

.. code-block:: bash

    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode --max-trials 8
    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode-agent \
        --max-trials 8 --planner agent

On an RTX 5090 (TensorRT 11.4, Python runtime), with calibration and two race
pairs per promotion, an agent session with a research brief went from 2.947 to
1.031 ms per token in five trials: ``static_cache_v3`` with CUDA graphs, then
the fused decoder blocks (73% of memory-bandwidth-bound speed). Two edits of
the blocks' launch settings and an attention build route measured within
noise and were rejected, and the agent stopped; fresh baseline/best pairs
confirmed +64.7% (2.86x tokens per second) with top-1 agreement 1.0. Earlier
sessions, before the fused blocks existed, reached 1.24-1.25 ms per token. See
``tools/llm/README.md`` for the full results.

Limitations
-----------

* Candidates are measured on the workload you provide: the result is the best
  found within the budget for that workload, hardware and software, not a
  global optimum.
* Workers run on the machine and GPU of the session. They wait for the GPU to
  be idle before timing, but the session does not lock clocks; a GPU that
  stays busy longer than ``quiet_gpu_s`` makes measurements unstable, and an
  unstable baseline ends the session. Use an exclusive GPU, or set
  ``quiet_gpu_s=0`` and a looser ``max_spread`` on a GPU that is never idle
  (one driving a display, for example).
* Memory is the process's GPU memory from NVML where the process ID is visible,
  otherwise PyTorch's peak allocation, which excludes TensorRT's own memory.
* The workload factory and an agent adapter are trusted code. Agent replies are
  validated against the tool catalog, but tool implementations run as written.
* The artifact is what the workload's ``Candidate.export`` saves. Runtime tools
  (``runtime.*``, such as CUDA graphs) are execution modes, not part of a saved
  program: apply them when deploying, as ``recipe.json`` records.
* A program loaded with ``torch_tensorrt.load(path).module()`` checks every
  input on every call. For the 58-input Qwen3 decode engine that costs about
  0.3 ms per token (1.56 ms instead of 1.24 ms on an RTX 5090);
  ``module(check_guards=False)`` with the module's input-check pre-hook removed
  avoids it. A loaded program also still copies in-place KV-cache outputs
  (about 0.04 ms per token), which the eager runtime path no longer does.

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

.. autoclass:: Measurement

.. autoclass:: PatchSlot

.. autofunction:: load_module

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
