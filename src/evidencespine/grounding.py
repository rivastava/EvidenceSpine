"""Span grounding: turn ``file:line`` references into checksummed evidence items.

A grounded evidence item binds a claim to an exact excerpt of a source file:
``source_id`` + ``line_start``/``line_end`` + the excerpt text + a sha256
checksum of the excerpt. Checksums make citations checkable (see the
drift-checker) and make ``verified`` a verifiable status.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_MAX_EVIDENCE_LINES = 500

_REF_RE = re.compile(
    r"^(?P<path>.+?)"
    r"(?:"
    r"#L(?P<a>\d+)(?:-L?(?P<b>\d+))?"
    r"|:(?P<c>\d+)-(?P<d>\d+)"
    r")$"
)


def parse_ref(ref: str) -> Optional[Tuple[str, int, int]]:
    """Parse ``path#L10-L20`` / ``path#L10`` / ``path:10-20`` into (path, start, end)."""
    text = str(ref or "").strip()
    if not text:
        return None
    match = _REF_RE.match(text)
    if not match:
        return None
    path = match.group("path").strip()
    if not path:
        return None
    start = int(match.group("a") or match.group("c") or 0)
    end = int(match.group("b") or match.group("d") or start)
    if start < 1:
        return None
    return path, start, max(end, start)


def excerpt_checksum(excerpt: str) -> str:
    digest = hashlib.sha256(excerpt.encode("utf-8", errors="ignore")).hexdigest()
    return f"sha256:{digest}"


def read_lines(path: str, line_start: int, line_end: int) -> Optional[List[str]]:
    """Return the exact lines at a 1-indexed range, or None when unreadable."""
    if line_end < line_start or line_end - line_start > _MAX_EVIDENCE_LINES:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    if line_start > len(lines):
        return None
    end = min(line_end, len(lines))
    return [line.rstrip("\n").rstrip("\r") for line in lines[line_start - 1 : end]]


def ground_file(
    path: str,
    line_start: int,
    line_end: int,
    *,
    source_root: str = ".",
    allow_absolute: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build a grounded evidence item for a file/line range.

    ``path`` may be absolute or relative to ``source_root``. Relative paths must
    resolve inside ``source_root`` (no ``../`` escape). Absolute paths are
    honored as explicit references only when ``allow_absolute`` is True; when
    False (server-side grounding), absolute paths are rejected outright so a
    caller cannot read arbitrary host files. Returns None when the file or
    range cannot be read.
    """
    if os.path.isabs(path):
        if not allow_absolute:
            return None
        full = path
    else:
        root_real = os.path.realpath(str(source_root or "."))
        full = os.path.realpath(os.path.join(root_real, path))
        if not full.startswith(root_real + os.sep) and full != root_real:
            return None
    lines = read_lines(full, line_start, line_end)
    if lines is None or not lines:
        return None
    excerpt = "\n".join(lines)
    return {
        "source_id": path,
        "line_start": int(line_start),
        "line_end": int(line_end),
        "excerpt": excerpt,
        "checksum": excerpt_checksum(excerpt),
    }


def ground_ref(ref: str, *, source_root: str = ".", allow_absolute: bool = True) -> Optional[Dict[str, Any]]:
    """Build a grounded evidence item from a ``path#L1-L5`` style reference."""
    parsed = parse_ref(ref)
    if parsed is None:
        return None
    path, start, end = parsed
    return ground_file(path, start, end, source_root=source_root, allow_absolute=allow_absolute)


def ground_claim_refs(
    refs: Any,
    *,
    source_root: str = ".",
    limit: int = 5,
    allow_absolute: bool = True,
) -> List[Dict[str, Any]]:
    """Best-effort grounding for a list of refs; ungroundable refs are skipped."""
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ref in list(refs or []):
        if not isinstance(ref, str) or not ref.strip():
            continue
        item = ground_ref(ref, source_root=source_root, allow_absolute=allow_absolute)
        if item is None:
            continue
        key = f"{item['source_id']}#L{item['line_start']}-{item['line_end']}"
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= max(1, int(limit)):
            break
    return items
