"""xKV core proxy; deliberately independent of NeMo and CUTLASS."""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from .common import PhaseTimer, budget_report, normalize_budget_ratio, recursive_nbytes


def enumerate_xkv_state(cache: Any) -> dict[str, Any]:
    """Enumerate all U/SV factors from cache objects, mappings, or layer groups."""
    found: dict[str, Any] = {}
    seen: set[int] = set()

    def visit(obj: Any, path: str) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key).lower() in {"u", "sv"}:
                    found[child] = value
                else:
                    visit(value, child)
            return
        for name in ("U", "SV", "u", "sv"):
            value = getattr(obj, name, None)
            if value is not None:
                found[f"{path}.{name}" if path else name] = value
        for name in ("layer_groups", "groups", "caches", "layers"):
            value = getattr(obj, name, None)
            if value is not None:
                visit(value, f"{path}.{name}" if path else name)
        if isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                visit(value, f"{path}[{index}]")

    visit(cache, "cache")
    return found


class XKVAdapter:
    def __init__(self, *, budget_ratio: float | str = 0.25,
                 prefill: Callable[[str], Any] | None = None,
                 reconstruct: Callable[[Any], Any] | None = None,
                 generate: Callable[[str, Any], Any] | None = None,
                 static_state: Any = None, dry_run: bool = False,
                 strict_budget: bool = False) -> None:
        self.budget_ratio = normalize_budget_ratio(budget_ratio)
        self._prefill, self._reconstruct, self._generate = prefill, reconstruct, generate
        self._static = static_state
        self.dry_run, self.strict_budget = dry_run, strict_budget
        self._timer = PhaseTimer()
        self.reset()

    @property
    def timings_ms(self): return dict(self._timer.ms)

    def reset(self) -> None:
        self._factors: dict[str, Any] = {}
        self._layer_groups = b""
        self._dense_reference = None
        self._dense_workspace = None
        self.last_budget: dict[str, Any] = {}
        self.peak_temporary_bytes = 0
        self._timer.reset()

    def prefill_archive(self, text: str, segment_id: str | None = None) -> dict[str, Any]:
        if self.dry_run:
            size = max(8, len(text.encode()) * 16)
            self._dense_reference = bytearray(size)
            groups = [{"layers": [0, 1, 2, 3], "segment_id": segment_id}]
            group_bytes = json.dumps(
                groups, separators=(",", ":"), sort_keys=True
            ).encode()
            archive = bytearray(max(0, int(size * self.budget_ratio) - len(group_bytes)))
            self._factors = {"group[0].U": archive[:len(archive)//2],
                             "group[0].SV": archive[len(archive)//2:]}
        else:
            if self._prefill is None: raise RuntimeError("prefill callback is required")
            with self._timer.measure("compress"):
                result = self._prefill(text)
            if not isinstance(result, Mapping):
                result = {"cache": result}
            cache = result.get("cache", result)
            self._factors = enumerate_xkv_state(cache)
            if not self._factors:
                raise ValueError("xKV archive exposed no U/SV factors")
            self._dense_reference = result.get("dense_reference", result.get("dense_kv"))
            groups = result.get("layer_groups", [])
        self._layer_groups = json.dumps(groups, separators=(",", ":"), sort_keys=True).encode()
        persistent = {"factors": self._factors, "layer_group_metadata": self._layer_groups}
        self.last_budget = budget_report(self.budget_ratio, self._dense_reference, persistent)
        if self.strict_budget and not self.last_budget["within_budget"]:
            raise MemoryError(f"xKV factors exceed byte budget: {self.last_budget}")
        return dict(self.last_budget)

    def query(self, text: str, **kwargs: Any) -> Any:
        if not self._factors: raise RuntimeError("prefill_archive must be called first")
        if self.dry_run: return "314159"
        if self._reconstruct is None or self._generate is None:
            raise RuntimeError("reconstruct and generate callbacks are required")
        with self._timer.measure("recover"):
            self._dense_workspace = self._reconstruct(self._factors)
        self.peak_temporary_bytes = max(self.peak_temporary_bytes,
                                        recursive_nbytes(self._dense_workspace))
        try:
            with self._timer.measure("generate"):
                return self._generate(text, self._dense_workspace, **kwargs)
        finally:
            self._dense_workspace = None

    def persistent_state(self) -> Mapping[str, Any]:
        return {"U_SV": self._factors, "layer_group_metadata": self._layer_groups}

    def static_extra_state(self) -> Mapping[str, Any]:
        return {"cross_layer_projection": self._static}

    def temporary_state(self) -> Mapping[str, Any]:
        return {"dense_reconstruction": self._dense_workspace}
