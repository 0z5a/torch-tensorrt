# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import pytest

from torch_tensorrt.agentic import AgentPlanner, HeuristicPlanner, Proposal
from torch_tensorrt.agentic._planner import (
    _proposal_from_reply,
    planner_from_json,
    proposal_schema,
    validate_changes,
)
from torch_tensorrt.agentic._spec import digest
from torch_tensorrt.agentic._tools import apply, catalog
from torch_tensorrt.dynamo.region_compilation._generation import _providers

from ._cases import FAKE_TOOLS

TOOLS = catalog(FAKE_TOOLS)


def evidence(recipe=None, signals=(), history=(), baseline=None, tried=()):
    recipe = recipe or {}
    return {
        "goal": {"metric": "latency_ms", "min_improvement": 0.02, "target": None},
        "baseline": {"recipe": baseline or {}},
        "incumbent": {"recipe": recipe, "signals": list(signals)},
        "history": list(history),
        "tools": [
            {**tool.to_json(), "current": recipe.get(name, tool.default)}
            for name, tool in TOOLS.items()
        ],
        "tried": sorted({digest(baseline or {}), digest(recipe), *tried}),
    }


def signal(name, detail="measured"):
    return {"signal": name, "strength": 0.9, "detail": detail}


def test_heuristic_planner_tries_what_relieves_the_strongest_signal(tmp_path):
    proposal = HeuristicPlanner().propose(
        evidence(signals=[signal("precision"), signal("launch_overhead")]), tmp_path
    )
    assert proposal.changes == {"engine.precision": "fp16"}
    assert proposal.rationale.startswith("precision: measured")
    # Without a signal, runtime options come first.
    plain = HeuristicPlanner().propose(evidence(), tmp_path)
    assert plain.changes == {"runtime.cudagraphs": True}


def test_heuristic_planner_skips_explored_options_and_never_reverts(tmp_path):
    history = [
        {
            "trial": "0001",
            "changes": {"runtime.cudagraphs": True},
            "decision": "promoted",
        },
        {
            "trial": "0002",
            "changes": {"engine.precision": "fp16"},
            "decision": "rejected",
        },
    ]
    proposal = HeuristicPlanner().propose(
        evidence(recipe={"runtime.cudagraphs": True}, history=history), tmp_path
    )
    # cudagraphs=False is the baseline's value and fp16 was tried already.
    assert proposal.changes == {"engine.optimization_level": 5}


def test_heuristic_planner_stops_when_every_option_was_tried(tmp_path):
    history = [
        {"changes": {"runtime.cudagraphs": True}},
        {"changes": {"engine.precision": "fp16"}},
        {"changes": {"engine.optimization_level": 5}},
        {"changes": {"workload.variant": "b"}},
        {"changes": {"workload.variant": "c"}},
    ]
    proposal = HeuristicPlanner().propose(evidence(history=history), tmp_path)
    assert proposal.stop and not proposal.changes


ADAPTER = """
import json, os

def reply(*, model, prompt, workdir, timeout_s, schema, evidence):
    with open(os.path.join(workdir, "seen.json"), "w") as stream:
        json.dump({"model": model, "prompt": prompt, "schema": schema, "evidence": evidence}, stream)
    return {"action": "experiment", "changes": [{"tool": "runtime.cudagraphs", "value": True}],
            "rationale": "launch overhead dominates"}

def broken(**kwargs):
    raise RuntimeError("provider is down")

def stop(**kwargs):
    return {"action": "stop", "changes": [], "rationale": "nothing left"}
"""


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    (tmp_path / "planner_fixture.py").write_text(ADAPTER)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    )
    return "planner_fixture"


def test_agent_planner_asks_the_provider_with_the_evidence(adapter, tmp_path):
    planner = AgentPlanner(
        f"{adapter}:reply", "model-x", instructions="Prefer runtime changes."
    )
    request = evidence(signals=[signal("launch_overhead")])
    proposal = planner.propose(request, tmp_path / "planner")
    assert proposal == Proposal(
        changes={"runtime.cudagraphs": True},
        rationale="launch overhead dominates",
        planner="agent",
    )
    seen = json.loads((tmp_path / "planner" / "seen.json").read_text())
    assert seen["model"] == "model-x"
    assert seen["evidence"] == request
    assert seen["schema"] == proposal_schema(2)
    assert "EVIDENCE:" in seen["prompt"] and "Prefer runtime changes." in seen["prompt"]
    assert '"launch_overhead"' in seen["prompt"]


def test_agent_planner_can_stop(adapter, tmp_path):
    proposal = AgentPlanner(f"{adapter}:stop").propose(evidence(), tmp_path)
    assert proposal.stop and proposal.rationale == "nothing left"


def test_agent_planner_falls_back_when_the_provider_fails(adapter, tmp_path):
    proposal = AgentPlanner(f"{adapter}:broken").propose(evidence(), tmp_path)
    assert proposal.changes == {"runtime.cudagraphs": True}
    assert proposal.planner == "heuristic"
    assert "agent planner unavailable" in proposal.note
    strict = AgentPlanner(f"{adapter}:broken", fallback=None)
    assert strict.to_json()["fallback"] is None
    with pytest.raises(_providers.ProviderError):
        strict.propose(evidence(), tmp_path / "strict")


@pytest.mark.parametrize(
    "reply,match",
    [
        ({"action": "maybe", "changes": [], "rationale": ""}, "no valid action"),
        ({"action": "experiment", "changes": [], "rationale": ""}, "without changes"),
        ({"action": "experiment", "changes": ["x"], "rationale": ""}, "malformed"),
    ],
)
def test_malformed_replies_are_rejected(reply, match):
    with pytest.raises(ValueError, match=match):
        _proposal_from_reply(reply, "agent")


def test_proposals_are_validated_against_the_catalog():
    assert validate_changes(
        Proposal(changes={"engine.optimization_level": 5}), {}, TOOLS
    ) == {"engine.optimization_level": 5}
    with pytest.raises(ValueError, match="changes nothing"):
        validate_changes(Proposal(), {}, TOOLS)
    with pytest.raises(ValueError, match="Unknown tool"):
        validate_changes(Proposal(changes={"engine.fast": True}), {}, TOOLS)
    with pytest.raises(ValueError, match="does not allow"):
        validate_changes(Proposal(changes={"engine.optimization_level": 4}), {}, TOOLS)
    assert apply({}, {"workload.variant": "a"}, TOOLS) == {}


def test_planners_round_trip_through_json():
    assert isinstance(planner_from_json(HeuristicPlanner().to_json()), HeuristicPlanner)
    agent = AgentPlanner("codex", "gpt", timeout_s=30, max_changes=3, instructions="hi")
    rebuilt = planner_from_json(json.loads(json.dumps(agent.to_json())))
    assert rebuilt.to_json() == agent.to_json()
    with pytest.raises(ValueError, match="Cannot rebuild"):
        planner_from_json({"kind": "custom"})
    with pytest.raises(ValueError, match="max_changes"):
        AgentPlanner(max_changes=0)
