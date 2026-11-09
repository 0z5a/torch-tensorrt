# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Code-level experiments: fenced source files the planner may edit.

A workload declares :class:`PatchSlot` objects for source files it builds
from, such as a lowering pass or region kernels. Lines containing
``EVOLVE-BLOCK-START`` and ``EVOLVE-BLOCK-END`` fence the editable parts. A
planner proposes SEARCH/REPLACE edits; each search text must occur exactly
once in the current version of the file and lie inside a fenced block. The
edited file must still compile. It is stored under its SHA-256 in the session
directory, and the recipe records it as ``patch.<slot>``, so a patched
candidate is reproducible, de-duplicated and reviewable like any other.

The worker passes the patched file's path to :meth:`Case.build` as
``values["patch.<slot>"]`` (``None`` for the original); :func:`load_module`
imports it. The evaluator, the reference and the accuracy bounds stay out of
the planner's reach: only fenced code changes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

FENCE_START = "EVOLVE-BLOCK-START"
FENCE_END = "EVOLVE-BLOCK-END"
PATCH_PREFIX = "patch."
_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")
_SHA = re.compile(r"[0-9a-f]{64}")
# Edits beyond this many changed lines per proposal are refused: reviewable
# diffs, and less room for an agent to rewrite what it should tune.
MAX_EDITED_LINES = 400


@dataclass(frozen=True)
class PatchSlot:
    """A source file the planner may edit inside its EVOLVE-BLOCK fences.

    ``name`` is a lowercase identifier; the recipe key is ``patch.<name>``.
    ``path`` is the original file. ``description`` tells the planner what the
    fenced code does and what may change.
    """

    name: str
    path: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(
                f"PatchSlot name must be a lowercase identifier; got {self.name!r}"
            )
        # Resolved, so a session resumed from another directory finds the file.
        object.__setattr__(self, "path", str(Path(self.path).expanduser().resolve()))
        source = Path(self.path)
        if not source.is_file():
            raise ValueError(f"PatchSlot {self.name}: no file at {self.path}")
        if not fenced_spans(source.read_text()):
            raise ValueError(
                f"PatchSlot {self.name}: {self.path} has no "
                f"{FENCE_START}/{FENCE_END} block"
            )

    @property
    def key(self) -> str:
        return PATCH_PREFIX + self.name

    def to_json(self) -> dict[str, Any]:
        """The slot as recorded in the case, with the original file's SHA-256.

        The digest makes a session detect an edit to the original between
        resumes, as it detects any other change to the workload.
        """
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "sha256": hashlib.sha256(Path(self.path).read_bytes()).hexdigest(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PatchSlot":
        return cls(data["name"], data["path"], data["description"])


def is_patch_key(name: str) -> bool:
    return name.startswith(PATCH_PREFIX)


def is_patch_value(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def fenced_spans(source: str) -> list[tuple[int, int]]:
    """Character spans of the editable text between fence lines (fences excluded)."""
    spans = []
    offset = 0
    start: int | None = None
    for line in source.splitlines(keepends=True):
        if FENCE_START in line:
            if start is not None:
                raise ValueError(f"nested {FENCE_START}")
            start = offset + len(line)
        elif FENCE_END in line:
            if start is None:
                raise ValueError(f"{FENCE_END} without {FENCE_START}")
            spans.append((start, offset))
            start = None
        offset += len(line)
    if start is not None:
        raise ValueError(f"unterminated {FENCE_START}")
    return spans


def _fence_lines(source: str) -> list[str]:
    return [
        line
        for line in source.splitlines(keepends=True)
        if FENCE_START in line or FENCE_END in line
    ]


def _fenced_matches(
    source: str, search: str, spans: Sequence[tuple[int, int]]
) -> list[int]:
    """Offsets of every occurrence of ``search``, overlapping ones included, inside a span."""
    found = []
    for start, end in spans:
        at = source.find(search, start)
        while at != -1 and at + len(search) <= end:
            found.append(at)
            at = source.find(search, at + 1)
    return found


def apply_edits(source: str, edits: Sequence[Mapping[str, Any]]) -> str:
    """Apply SEARCH/REPLACE ``edits`` in order; raise ValueError if any is not allowed.

    Each search text must occur exactly once inside the fenced blocks (it may
    also occur elsewhere; only the fenced occurrence is edited). The edits
    must leave the fence lines unchanged and every block non-empty.
    """
    if not edits:
        raise ValueError("a patch needs at least one edit")
    fences = _fence_lines(source)
    changed = 0
    for index, edit in enumerate(edits):
        search, replace = edit.get("search"), edit.get("replace")
        if not isinstance(search, str) or not search:
            raise ValueError(f"edit {index}: 'search' must be non-empty text")
        if not isinstance(replace, str):
            raise ValueError(f"edit {index}: 'replace' must be text")
        if FENCE_START in replace or FENCE_END in replace:
            raise ValueError(f"edit {index}: the replacement may not contain fences")
        found = _fenced_matches(source, search, fenced_spans(source))
        if not found:
            if search in source:
                raise ValueError(f"edit {index} is not inside an EVOLVE-BLOCK region")
            raise ValueError(f"edit {index}: the search text does not occur")
        if len(found) > 1:
            raise ValueError(
                f"edit {index}: the search text occurs {len(found)} times inside "
                "the EVOLVE-BLOCK regions; it must occur exactly once"
            )
        at = found[0]
        changed += max(search.count("\n"), replace.count("\n")) + 1
        source = source[:at] + replace + source[at + len(search) :]
    try:
        spans = fenced_spans(source)
    except ValueError as error:
        raise ValueError(f"the edits break the EVOLVE-BLOCK fences: {error}") from None
    if _fence_lines(source) != fences:
        raise ValueError("the edits change an EVOLVE-BLOCK fence line")
    if any(not source[start:end].strip() for start, end in spans):
        raise ValueError("the edits empty an EVOLVE-BLOCK region")
    if changed > MAX_EDITED_LINES:
        raise ValueError(
            f"the patch changes about {changed} lines; at most {MAX_EDITED_LINES}"
        )
    return source


def check_source(source: str, filename: str) -> None:
    """Refuse a patched Python file that does not compile."""
    try:
        compile(source, filename, "exec")
    except SyntaxError as error:
        raise ValueError(f"the patched file does not compile: {error}") from None


class PatchStore:
    """Patched sources of one session, stored under their SHA-256."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory) / "patches"

    def save(self, source: str) -> str:
        sha = hashlib.sha256(source.encode()).hexdigest()
        path = self.path(sha)
        if not path.is_file():
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(source)
            temporary.replace(path)
        return sha

    def path(self, sha: str) -> Path:
        if not is_patch_value(sha):
            raise ValueError(f"not a patch digest: {sha!r}")
        return self.directory / f"{sha}.py"

    def source(self, slot: PatchSlot, recipe: Mapping[str, Any]) -> str:
        """The slot's source under ``recipe``: its patch, or the original file."""
        sha = recipe.get(slot.key)
        return self.path(sha).read_text() if sha else Path(slot.path).read_text()


def editable_blocks(source: str) -> list[str]:
    return [source[start:end] for start, end in fenced_spans(source)]


def load_module(path: str | Path, name: str) -> ModuleType:
    """Import the Python file at ``path`` as a fresh module named ``name``."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
