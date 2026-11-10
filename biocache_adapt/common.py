"""Dependency-free common contract for BioCache proxy adapters."""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, os, subprocess, sys, time
from contextlib import contextmanager
from typing import Any, Mapping

SCHEMA_VERSION = "biocache-proxy-v1"
BUDGET_RATIOS = (0.125, 0.25, 0.5)
PHASES = ("compress", "recover", "generate", "network")


def recursive_nbytes(obj: Any, seen: set[int] | None = None) -> int:
    if obj is None:
        return 0
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return obj.nbytes if isinstance(obj, memoryview) else len(obj)
    if isinstance(obj, (str, int, float, bool)):
        return 0
    if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
        return int(obj.element_size()) * int(obj.nelement())
    if hasattr(obj, "nbytes") and not isinstance(obj, Mapping):
        try:
            return int(obj.nbytes)
        except (TypeError, ValueError):
            pass
    if isinstance(obj, Mapping):
        return sum(recursive_nbytes(k, seen) + recursive_nbytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(recursive_nbytes(v, seen) for v in obj)
    if dataclasses.is_dataclass(obj):
        return sum(recursive_nbytes(getattr(obj, f.name), seen) for f in dataclasses.fields(obj))
    if hasattr(obj, "__dict__"):
        return recursive_nbytes(vars(obj), seen)
    return 0


class PhaseTimer:
    def __init__(self): self.ms = {name: 0.0 for name in PHASES}
    @contextmanager
    def measure(self, phase: str):
        start = time.perf_counter()
        try: yield
        finally: self.ms[phase] += (time.perf_counter() - start) * 1000
    def reset(self):
        for phase in self.ms: self.ms[phase] = 0.0


def normalize_budget_ratio(value: float | str) -> float:
    text = str(value).strip()
    ratio = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    ratio = ratio / 100 if ratio > 1 else ratio
    if ratio not in BUDGET_RATIOS:
        raise ValueError("budget must be one of 12.5%, 25%, or 50%")
    return ratio


def budget_report(ratio, dense_state, persistent_state):
    ratio = normalize_budget_ratio(ratio)
    dense, actual = recursive_nbytes(dense_state), recursive_nbytes(persistent_state)
    budget = int(dense * ratio)
    return {"budget_ratio": ratio, "dense_reference_bytes": dense, "budget_bytes": budget,
            "actual_persistent_bytes": actual, "within_budget": actual <= budget}


def result_record(*, method, suite, sample_id, adapter, response="", target="",
                  context_length=0, needle_position=None, metadata=None):
    if suite not in {"needle", "length_scan"}: raise ValueError("invalid suite")
    return {
        "schema_version": SCHEMA_VERSION, "method": method, "suite": suite,
        "sample_id": str(sample_id), "context_length": int(context_length),
        "needle_position": needle_position, "budget_ratio": adapter.budget_ratio,
        "budget_bytes": adapter.last_budget.get("budget_bytes", 0),
        "actual_persistent_bytes": recursive_nbytes(adapter.persistent_state()),
        "actual_static_extra_bytes": recursive_nbytes(adapter.static_extra_state()),
        "actual_temporary_bytes": recursive_nbytes(adapter.temporary_state()),
        "peak_temporary_bytes": int(adapter.peak_temporary_bytes),
        "timings_ms": {p: float(adapter.timings_ms.get(p, 0)) for p in PHASES},
        "response": response, "target": target,
        "exact_match": response.strip() == target.strip() if target else None,
        "metadata": dict(metadata or {}),
    }


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def collect_manifest(repo_root, config):
    def git(*args):
        try:
            return subprocess.run(["git", "-C", repo_root, *args], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
        except OSError: return None
    hashes = {}
    for name in sorted(os.listdir(os.path.dirname(__file__))):
        path = os.path.join(os.path.dirname(__file__), name)
        if name.endswith(".py") and os.path.isfile(path):
            with open(path, "rb") as f: hashes[name] = hashlib.sha256(f.read()).hexdigest()
    return {"schema_version": SCHEMA_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv, "python": sys.version, "config": dict(config),
            "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                    "dirty": bool(git("status", "--porcelain"))}, "source_sha256": hashes}


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(dict(manifest), f, ensure_ascii=False, indent=2)


def dry_run_cli(adapter_cls, method, argv=None):
    p = argparse.ArgumentParser(description=f"{method} BioCache proxy dry-run")
    p.add_argument("--dry-run", action="store_true", required=True)
    p.add_argument("--budget", choices=("12.5%", "25%", "50%"), default="25%")
    p.add_argument("--suite", choices=("needle", "length_scan"), default="needle")
    p.add_argument("--output"); p.add_argument("--manifest")
    args = p.parse_args(argv)
    adapter = adapter_cls(budget_ratio=args.budget, dry_run=True)
    adapter.reset(); adapter.prefill_archive("prefix needle=314159 suffix", segment_id="dry-0")
    response = str(adapter.query("What is the needle?"))
    record = result_record(method=method, suite=args.suite, sample_id="dry-0", adapter=adapter,
                           response=response, target="314159", context_length=27, needle_position=7,
                           metadata={"dry_run": True})
    if args.output: append_jsonl(args.output, record)
    if args.manifest: write_manifest(args.manifest, collect_manifest(os.path.dirname(os.path.dirname(__file__)), vars(args)))
    print(json.dumps(record, ensure_ascii=False)); return record
