"""
reid_resolver.py
================
Bridges the offline global Re-ID pipeline (src/reid.py) to the live server.

Responsibilities:
  * compute a stable `raw_id -> canonical_id (1..4)` mapping for a given source
    by running src.reid.run_reid once,
  * cache the result to disk so restarts are instant,
  * load a cached mapping automatically (no compute),
  * expose a tiny query API (ReIDResolver.canonical).

The mapping is only valid when the live tracker uses the SAME BoT-SORT config,
model, imgsz and conf/iou as the offline pass — the analyzer guarantees that
(see src/analyzer.py: both paths use configs/botsort_reid.yaml, 0.3/0.5/640).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "reid_cache"


class ReIDResolver:
    """Holds a raw_id -> canonical_id mapping and answers queries."""

    def __init__(self, mapping: dict[int, int], meta: Optional[dict] = None):
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.meta = meta or {}

    def canonical(self, raw_id) -> Optional[int]:
        return self.mapping.get(int(raw_id))

    def __bool__(self) -> bool:
        return bool(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    # ---- caching ----------------------------------------------------------
    @staticmethod
    def _cache_path(source: str, player_weights: str, k: int, tracker: str) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = Path(source)
        size = p.stat().st_size if p.exists() else 0
        wstem = Path(player_weights).stem if player_weights else "auto"
        tstem = Path(tracker).stem
        key = f"{p.stem}_{size}_{wstem}_{tstem}_k{k}"
        return CACHE_DIR / f"{key}.json"

    @classmethod
    def load_cached(
        cls, source: str, player_weights: str, k: int = 4,
        tracker: str = "configs/botsort_reid.yaml",
    ) -> Optional["ReIDResolver"]:
        cp = cls._cache_path(source, player_weights, k, tracker)
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
            resolver = cls(data.get("mapping", {}), data.get("config"))
            print(f"[reid] loaded cached mapping ({len(resolver)} ids) from {cp.name}")
            return resolver
        except Exception as e:
            print(f"[reid] cache read failed ({e}), recomputing")
            return None

    # ---- compute ----------------------------------------------------------
    @classmethod
    def compute(
        cls, source: str, player_weights: str, k: int = 4,
        backend: str = "auto", device: str = "auto",
        tracker: str = "configs/botsort_reid.yaml",
    ) -> "ReIDResolver":
        from src.reid import run_reid

        print(f"[reid] computing stable IDs for {source} (this runs once) ...")
        result = run_reid(
            video_path=source,
            player_weights=player_weights,
            tracker_cfg=tracker,
            k=k,
            backend=backend,
            device=device,
            output_dir=str(CACHE_DIR),
        )
        mapping = {int(rk): int(rv) for rk, rv in result.get("mapping", {}).items()}
        resolver = cls(mapping, result.get("config"))

        cp = cls._cache_path(source, player_weights, k, tracker)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(
            json.dumps({"mapping": mapping, "config": result.get("config")}, indent=2)
        )
        print(f"[reid] done -> {len(mapping)} track ids mapped to "
              f"{len(set(mapping.values()))} players; cached at {cp.name}")
        return resolver


class ReIDRunner:
    """Background compute wrapper with thread-safe status reporting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "idle"        # idle | computing | ready | error
        self._detail = ""
        self._thread: Optional[threading.Thread] = None

    @property
    def status(self) -> dict:
        with self._lock:
            return {"status": self._status, "detail": self._detail}

    def is_running(self) -> bool:
        with self._lock:
            return self._status == "computing"

    def start(
        self, source: str, player_weights: str, device: str = "0", k: int = 4,
        on_done=None,
    ) -> bool:
        """Kick off a background compute. Returns False if one is already running."""
        with self._lock:
            if self._status == "computing":
                return False
            self._status = "computing"
            self._detail = "starting"

        def _work():
            try:
                resolver = ReIDResolver.compute(
                    source, player_weights, k=k, device=device
                )
                with self._lock:
                    self._status = "ready"
                    self._detail = f"{len(resolver.mapping)} ids -> {len(set(resolver.mapping.values()))} players"
                if on_done:
                    on_done()
            except Exception as e:  # pragma: no cover
                with self._lock:
                    self._status = "error"
                    self._detail = str(e)
                print(f"[reid] compute failed: {e}")

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()
        return True
