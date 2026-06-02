from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from prism.types import Action


TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|-?\d+|[<>=!]=|[+\-*/]")


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def state_feature_names(state: str) -> list[str]:
    lower = state.lower()
    names = ["bias"]
    names.extend(f"tok:{token}" for token in tokenize(lower))

    if "steps taken: 0" in lower:
        names.append("kw:first_step")
    if "tests=passed" in lower or "all tests passed" in lower:
        names.append("kw:tests_passed")
    if "tests=failed" in lower or "assertionerror" in lower or "traceback" in lower:
        names.append("kw:tests_failed")
    if "patch=correct" in lower or "patch=wrong" in lower:
        names.append("kw:patch_applied")
    if "src/module.py" in lower and "return" in lower:
        names.append("kw:has_source")

    if "return a - b" in lower or "add(2, 3)" in lower:
        names.append("bug:plus_minus")
    if "range(len(items) + 1)" in lower or "indexerror" in lower:
        names.append("bug:off_by_one")
    if "price - tax" in lower or "nameerror" in lower:
        names.append("bug:wrong_variable")
    if "sum(values) / len(values)" in lower or "empty list" in lower:
        names.append("bug:empty_case")
    if "age > 18" in lower or "18 should be adult" in lower:
        names.append("bug:comparison")

    return names


def feature_index(name: str, num_features: int) -> int:
    return stable_hash(name, num_features)


def sparse_features(names: Iterable[str], num_features: int) -> dict[int, float]:
    features: dict[int, float] = {}
    for name in names:
        idx = feature_index(name, num_features)
        features[idx] = features.get(idx, 0.0) + 1.0
    return features


def state_features(state: str, num_features: int) -> dict[int, float]:
    return sparse_features(state_feature_names(state), num_features)


def prm_features(state: str, action: Action, num_features: int) -> dict[int, float]:
    names = [f"state:{name}" for name in state_feature_names(state)]
    names.append(f"action:{action.key}")
    names.append(f"action_name:{action.name}")
    if action.argument:
        names.append(f"action_arg:{action.argument}")

    for state_name in state_feature_names(state):
        if state_name.startswith("bug:") or state_name.startswith("kw:"):
            names.append(f"cross:{state_name}|{action.key}")

    return sparse_features(names, num_features)

