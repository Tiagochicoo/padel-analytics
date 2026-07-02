"""
src/match_report.py
===================
Post-match report assembler — the definitive, exportable summary of a match.

Consumes the pipeline's accumulated LIVE state (per-slot stats, court positions,
score, shot log, rules-engine totals, match lifecycle) and produces a single
``match_report.json`` plus final heatmap PNGs under
``data/reports/<source_id>/``.

The canonical slots P1..P4 are already stable (locked by the online Re-ID
resolver at match-start, Phase 4) and team-assigned (Phase 5), so the report
just aggregates by slot/team — no re-clustering needed for v1. (A post-match
full re-cluster to re-validate IDs is a future refinement; it would not change
the per-slot stats, which are already keyed by canonical slot.)

Report shape:
    {
      "meta":    source, generated_at, match lifecycle, duration,
      "score":   [team0_points, team1_points],
      "rallies": int, "total_shots": int,
      "per_player": [ {slot, name, team, time_on_court, distance_m,
                       shots, shots_by_type, heatmap}, ... ],
      "per_team":   { "0": {players, shots, points}, "1": {...} },
      "shot_log":   [ {slot, team, type, frame}, ... ]
    }
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.heatmap import render_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "data" / "reports"


def build_report(pipeline) -> dict:
    """Assemble the report dict from the pipeline's current LIVE state."""
    snap = pipeline.stats_acc.snapshot()
    score = list(getattr(pipeline, "_score", [0, 0]))
    rallies = getattr(getattr(pipeline, "_rules", None), "total_rallies", 0)
    match = pipeline.match_state.status

    fps = float(pipeline.stats.get("fps") or 0.0) or 30.0
    live_at = match.get("live_started_at")
    ended_at = match.get("ended_at")
    cur = pipeline._frame_idx
    end = ended_at if ended_at is not None else cur
    duration_frames = (end - live_at) if (live_at is not None and end is not None) else 0

    per_player = []
    per_team = {0: {"players": 0, "shots": 0, "points": score[0] if len(score) > 0 else 0},
                1: {"players": 0, "shots": 0, "points": score[1] if len(score) > 1 else 0}}
    total_shots = 0
    for p in snap:
        slot = p["slot"]
        team = p.get("team")
        shots = p.get("shots", 0)
        total_shots += shots
        per_player.append({
            "slot": slot,
            "name": p.get("name", f"Player {slot}"),
            "team": team,
            "active": p.get("active", False),
            "time_on_court": p.get("time_on_court", 0.0),
            "distance_m": p.get("distance_m", 0.0),
            "shots": shots,
            "shots_by_type": p.get("shots_by_type", {}),
            "heatmap": f"heat_P{slot}.png",
        })
        if team in per_team:
            per_team[team]["players"] += 1
            per_team[team]["shots"] += shots

    return {
        "meta": {
            "source": pipeline.stats.get("source", ""),
            "source_id": pipeline.current_source_id(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "match": match,
            "duration_frames": int(duration_frames),
            "duration_sec": round(duration_frames / fps, 1) if duration_frames > 0 else 0.0,
            "fps": round(fps, 1),
            "model": pipeline.stats.get("model", ""),
        },
        "score": score,
        "rallies": int(rallies),
        "total_shots": int(total_shots),
        "per_player": per_player,
        "per_team": {str(t): v for t, v in per_team.items()},
        "shot_log": list(getattr(pipeline, "_shot_log", [])),
    }


def save_report(report: dict, source_id: str,
                pipeline=None, render_heatmaps: bool = True) -> tuple:
    """Write match_report.json (+ final heatmap PNGs) and return (path, match_id).

    Each match is stored under ``data/reports/<match_id>/`` where ``match_id``
    is a sortable timestamp (``YYYYMMDD_HHMMSS``).  Multiple matches from the
    same source each get their own directory.
    """
    match_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_ROOT / match_id
    # handle the rare case of two saves within the same second
    i = 2
    while out_dir.exists() and (out_dir / "match_report.json").exists():
        out_dir = REPORTS_ROOT / f"{match_id}_{i}"
        i += 1
    match_id = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if render_heatmaps and pipeline is not None:
        for p in report["per_player"]:
            slot = p["slot"]
            if not p.get("active"):
                continue
            pts = pipeline.positions(slot)
            if not pts:
                continue
            if len(pts) > 6000:                       # cap KDE cost on long matches
                pts = pts[:: len(pts) // 6000]
            png = render_heatmap(pts, title=p["name"])
            (out_dir / f"heat_P{slot}.png").write_bytes(png)

    report.setdefault("meta", {})["match_id"] = match_id
    path = out_dir / "match_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path, match_id


def load_report(source_id: str) -> Optional[dict]:
    """Legacy: load the report for *source_id* (single-report era)."""
    path = REPORTS_ROOT / source_id / "match_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_report_by_id(match_id: str) -> Optional[dict]:
    """Load a saved report by its match_id directory name."""
    path = REPORTS_ROOT / match_id / "match_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_reports() -> list[dict]:
    """Return a lightweight metadata list of all saved match reports, newest first."""
    results = []
    if not REPORTS_ROOT.exists():
        return results
    for d in REPORTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        rp = d / "match_report.json"
        if not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        meta = r.get("meta", {})
        score = r.get("score", [0, 0])
        results.append({
            "match_id": d.name,
            "generated_at": meta.get("generated_at", ""),
            "source": meta.get("source", ""),
            "source_id": meta.get("source_id", ""),
            "score": score,
            "rallies": r.get("rallies", 0),
            "total_shots": r.get("total_shots", 0),
            "duration_sec": meta.get("duration_sec", 0),
            "match_state": (meta.get("match", {}) or {}).get("state", ""),
        })
    results.sort(key=lambda x: x["match_id"], reverse=True)
    return results


def report_dir(source_id: str) -> Path:
    """Legacy: directory for a source_id's report."""
    return REPORTS_ROOT / source_id


def report_match_dir(match_id: str) -> Path:
    """Directory for a specific saved match report."""
    return REPORTS_ROOT / match_id


def delete_report(match_id: str) -> bool:
    """Permanently delete a saved match report directory. Returns True if deleted."""
    import shutil
    d = REPORTS_ROOT / match_id
    if not d.exists() or not d.is_dir():
        return False
    shutil.rmtree(d)
    return True


def archive_report(match_id: str) -> bool:
    """Archive a report by renaming its directory with an _archived suffix."""
    d = REPORTS_ROOT / match_id
    if not d.exists() or not d.is_dir():
        return False
    archived = REPORTS_ROOT / (match_id + "_archived")
    d.rename(archived)
    return True


__all__ = ["build_report", "save_report", "load_report", "load_report_by_id",
           "list_reports", "report_dir", "report_match_dir", "delete_report",
           "archive_report", "REPORTS_ROOT"]
