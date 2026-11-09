# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Planners read the session's evidence and propose the next experiment.

A planner only proposes. It cannot build, measure or accept a candidate: the
session validates every proposal against the tool catalog and the independent
evaluator decides what a candidate achieved.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch_tensorrt.agentic._patches import is_patch_key
from torch_tensorrt.agentic._spec import digest
from torch_tensorrt.agentic._tools import (
    LAYERS,
    Tool,
    apply,
    catalog,
    is_member,
    unmet_requirements,
)

_LAYER_ORDER = ("runtime", "engine", "graph", "workload", "kernels")
_HEURISTIC_FALLBACK: Any = object()


@dataclass(frozen=True)
class Proposal:
    """The next experiment: tool changes and code patches to the incumbent, or stop.

    ``patches`` maps a patch slot's name to SEARCH/REPLACE edits
    (``{"search": ..., "replace": ...}``) of that slot's current source; see
    :class:`~torch_tensorrt.agentic.PatchSlot`.
    """

    changes: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    stop: bool = False
    planner: str = ""
    note: str = ""
    patches: Mapping[str, Sequence[Mapping[str, str]]] = field(default_factory=dict)


class Planner(ABC):
    """Base class: ``propose`` maps the session's evidence to a :class:`Proposal`."""

    name: str = "planner"

    @abstractmethod
    def propose(self, evidence: Mapping[str, Any], workdir: Path) -> Proposal: ...

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.name}


def tools_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Tool]:
    return catalog(
        [
            Tool(
                tool["name"],
                tuple(tool["values"]),
                tool["default"],
                tool["description"],
                tuple(tool["signals"]),
                tool.get("requires", {}),
            )
            for tool in evidence["tools"]
        ]
    )


def _tried_options(
    evidence: Mapping[str, Any],
    tools: Mapping[str, Tool],
    parent: str | None = None,
) -> set[tuple[str, str]]:
    """Options already explored: the baseline's values and every value a trial set.

    With ``parent``, only trials proposed from that incumbent count, so an
    option judged under an earlier incumbent can be revisited.
    """
    baseline = evidence["baseline"]["recipe"]
    tried = {
        (name, json.dumps(baseline.get(name, tool.default)))
        for name, tool in tools.items()
    }
    for trial in evidence.get("history", ()):
        if parent is not None and trial.get("parent") != parent:
            continue
        for name, value in trial.get("changes", {}).items():
            tried.add((name, json.dumps(value)))
    return tried


class HeuristicPlanner(Planner):
    """Follow the measured bottleneck, one change at a time.

    The incumbent's strongest profile signal picks the first options to try:
    the tools that declare they relieve it. Then every other option is tried,
    runtime first, then engine, graph, workload and kernel tools, each value in
    the order its tool lists. The baseline's own options are never proposed,
    so promoted changes are never reverted. Options are tried once in that
    order; when none is left, an option rejected under an earlier incumbent is
    revisited under the current one, since options interact. The planner stops
    once every option has been tried under the current incumbent.
    """

    name = "heuristic"

    def propose(self, evidence: Mapping[str, Any], workdir: Path) -> Proposal:
        tools = tools_from_evidence(evidence)
        incumbent = evidence["incumbent"]
        recipe = incumbent["recipe"]
        tried_recipes = {digest(dict(r)) for r in evidence.get("tried", ())}
        tried_anywhere = _tried_options(evidence, tools)
        tried_here = _tried_options(evidence, tools, parent=incumbent.get("trial"))
        current = {name: recipe.get(name, tool.default) for name, tool in tools.items()}

        ordered: list[tuple[Tool, str]] = []
        for signal in incumbent.get("signals", ()):
            for tool in tools.values():
                if signal["signal"] in tool.signals:
                    ordered.append((tool, f"{signal['signal']}: {signal['detail']}"))
        for layer in _LAYER_ORDER:
            for tool in tools.values():
                if tool.layer == layer:
                    ordered.append((tool, f"untried {layer}-layer option"))

        for tried, prefix in (
            (tried_anywhere, ""),
            (tried_here, "revisit under the new incumbent: "),
        ):
            for tool, reason in ordered:
                for value in tool.values:
                    if is_member(value, (current[tool.name],)):
                        continue
                    if (tool.name, json.dumps(value)) in tried:
                        continue
                    # An option comes with the settings it requires.
                    changes = {
                        **unmet_requirements(current, tool.name, tools),
                        tool.name: value,
                    }
                    candidate = apply(recipe, changes, tools)
                    if digest(candidate) in tried_recipes:
                        continue
                    needs = "".join(
                        f", with {name}={changes[name]!r}"
                        for name in changes
                        if name != tool.name
                    )
                    return Proposal(
                        changes=changes,
                        rationale=f"{prefix}{reason} -> try {tool.name}={value!r}"
                        f"{needs}",
                        planner=self.name,
                    )
        return Proposal(
            stop=True,
            rationale="Every exposed option has been tried",
            planner=self.name,
        )


_PROMPT = """\
You are the planner of a Torch-TensorRT optimization session. The user's goal,
constraints and budget, the validated baseline, the current best candidate (the
incumbent), its profile and bottleneck signals, the history of every trial and
the catalog of compiler tools are in the EVIDENCE below.

Choose the next experiment: at most {max_changes} tool changes applied to the
incumbent recipe, each a tool name and one of that tool's allowed values. Or
stop, when no remaining experiment is likely to help. An independent evaluator
builds and measures your experiment on the same workload; the session keeps it
only if it improves {metric} by more than {min_improvement:.1%} beyond noise and
passes the accuracy and memory constraints. You cannot evaluate or accept
results yourself.

Prefer the change that addresses the strongest measured bottleneck. Learn from
the history: do not repeat a recipe that was already tried, and do not retry an
option that failed for a reason the new experiment does not address. A tool's
"requires" lists settings its non-default values need; the incumbent must
already have them or the experiment must set them too. Explain in "rationale"
which evidence motivates the experiment and what you expect.
{patches}{instructions}
EVIDENCE:
{evidence}
"""

_PATCH_PROMPT = """
The workload also exposes source files you may edit ("patch_slots" in the
EVIDENCE: each slot's description and its editable EVOLVE-BLOCK regions as they
are in the incumbent). An experiment may change tools, patch slots, or both.
A patch is a list of SEARCH/REPLACE edits: "search" must match the slot's
current text exactly once inside its EVOLVE-BLOCK regions, and "replace" is the
new text. The session refuses edits outside those regions, edits that change a
fence line or empty a region, and files that do not compile; the evaluator
checks accuracy against the reference and only measured gains count. Send an
empty "patches" list when you change only tools. Keep patches small, focused
and correct; equal speed with simpler code is better.
"""


def proposal_schema(max_changes: int, slots: Sequence[str] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["experiment", "stop"]},
            "changes": {
                "type": "array",
                "maxItems": max_changes,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool": {"type": "string"},
                        "value": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "required": ["tool", "value"],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["action", "changes", "rationale"],
    }
    if slots:
        schema["properties"]["patches"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "slot": {"type": "string", "enum": list(slots)},
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "search": {"type": "string"},
                                "replace": {"type": "string"},
                            },
                            "required": ["search", "replace"],
                        },
                    },
                },
                "required": ["slot", "edits"],
            },
        }
        # Strict structured output needs every property required; an empty
        # list means no patch.
        schema["required"].append("patches")
    return schema


class AgentPlanner(Planner):
    """Ask a coding agent to choose the next experiment from the evidence.

    ``provider`` is ``"claude"``, ``"codex"`` (logged-in CLIs) or a trusted
    ``"module:attribute"`` adapter, as for
    :class:`~torch_tensorrt.region.KernelGenerator`. The agent receives the
    evidence as JSON and must answer with tool changes from the catalog. If the
    provider fails, ``fallback`` (by default a :class:`HeuristicPlanner`)
    proposes instead and the proposal records why; with ``fallback=None`` the
    error propagates and ends the session.
    """

    name = "agent"

    def __init__(
        self,
        provider: str = "claude",
        model: str | None = None,
        *,
        timeout_s: float = 600.0,
        max_changes: int = 2,
        instructions: str | None = None,
        fallback: Planner | None = _HEURISTIC_FALLBACK,
    ) -> None:
        if not isinstance(max_changes, int) or not 1 <= max_changes <= 5:
            raise ValueError("AgentPlanner max_changes must be between 1 and 5")
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_changes = max_changes
        self.instructions = instructions or ""
        self.fallback = (
            HeuristicPlanner() if fallback is _HEURISTIC_FALLBACK else fallback
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "provider": self.provider,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "max_changes": self.max_changes,
            "instructions": self.instructions,
            "fallback": self.fallback.to_json() if self.fallback else None,
        }

    def prompt(self, evidence: Mapping[str, Any]) -> str:
        goal = evidence["goal"]
        threshold = (evidence.get("noise") or {}).get("promotion_threshold")
        brief = evidence.get("brief")
        guidance = "\n\n".join(part for part in (brief, self.instructions) if part)
        return _PROMPT.format(
            max_changes=self.max_changes,
            metric=goal["metric"],
            min_improvement=(
                threshold if threshold is not None else goal["min_improvement"]
            ),
            patches=_PATCH_PROMPT if evidence.get("patch_slots") else "",
            instructions=(
                f"\nAdditional guidance from the user:\n{guidance}\n"
                if guidance
                else ""
            ),
            evidence=json.dumps(evidence, indent=1, sort_keys=True, default=str),
        )

    def propose(self, evidence: Mapping[str, Any], workdir: Path) -> Proposal:
        from torch_tensorrt.dynamo.region_compilation._generation import _providers

        try:
            reply = _providers.propose(
                provider=self.provider,
                model=self.model,
                prompt=self.prompt(evidence),
                workdir=workdir,
                timeout_s=self.timeout_s,
                schema=proposal_schema(
                    self.max_changes,
                    [slot["name"] for slot in evidence.get("patch_slots", ())],
                ),
                evidence=dict(evidence),
            )
            return _proposal_from_reply(reply, self.name)
        except (_providers.ProviderError, TimeoutError, ValueError, TypeError) as error:
            if self.fallback is None:
                raise
            proposal = self.fallback.propose(evidence, workdir)
            return replace(
                proposal,
                note=f"agent planner unavailable ({type(error).__name__}: {error}); "
                f"{self.fallback.name} planner proposed instead",
            )


def _proposal_from_reply(reply: Mapping[str, Any], planner: str) -> Proposal:
    action = reply.get("action")
    if action not in ("experiment", "stop"):
        raise ValueError(f"Planner reply has no valid action: {action!r}")
    rationale = str(reply.get("rationale", ""))
    if action == "stop":
        return Proposal(stop=True, rationale=rationale, planner=planner)
    changes: dict[str, Any] = {}
    for change in reply.get("changes") or ():
        if not isinstance(change, Mapping) or "tool" not in change:
            raise ValueError(f"Planner reply has a malformed change: {change!r}")
        changes[str(change["tool"])] = change.get("value")
    patches: dict[str, list[dict[str, str]]] = {}
    for patch in reply.get("patches") or ():
        if (
            not isinstance(patch, Mapping)
            or not isinstance(patch.get("slot"), str)
            or not isinstance(patch.get("edits"), (list, tuple))
        ):
            raise ValueError(f"Planner reply has a malformed patch: {patch!r}")
        edits = patches.setdefault(patch["slot"], [])
        for edit in patch["edits"]:
            if not (
                isinstance(edit, Mapping)
                and isinstance(edit.get("search"), str)
                and isinstance(edit.get("replace"), str)
            ):
                raise ValueError(f"Planner reply has a malformed edit: {edit!r}")
            edits.append({"search": edit["search"], "replace": edit["replace"]})
    if not changes and not patches:
        raise ValueError("Planner reply proposes an experiment without changes")
    return Proposal(
        changes=changes, rationale=rationale, planner=planner, patches=patches
    )


def planner_from_json(config: Mapping[str, Any]) -> Planner:
    """Rebuild a built-in planner from :meth:`Planner.to_json` (for resuming)."""
    kind = config.get("kind")
    if kind == HeuristicPlanner.name:
        return HeuristicPlanner()
    if kind == AgentPlanner.name:
        fallback = config.get("fallback")
        return AgentPlanner(
            config["provider"],
            config.get("model"),
            timeout_s=config.get("timeout_s", 600.0),
            max_changes=config.get("max_changes", 2),
            instructions=config.get("instructions") or None,
            fallback=planner_from_json(fallback) if fallback else None,
        )
    raise ValueError(f"Cannot rebuild planner {kind!r}; pass the planner explicitly")


def validate_changes(
    proposal: Proposal, recipe: Mapping[str, Any], tools: Mapping[str, Tool]
) -> dict[str, Any]:
    """The incumbent recipe with ``proposal``'s changes, or ValueError explaining why not."""
    if not proposal.changes:
        raise ValueError("The proposal changes nothing")
    for name in proposal.changes:
        if not is_patch_key(name) and name.partition(".")[0] not in LAYERS:
            raise ValueError(f"Unknown tool {name!r}")
    return apply(recipe, proposal.changes, tools)
