"""
modelutil.py
============
Shared helpers used by both the live inference pipeline (src/server/pipeline.py)
and the offline analysis jobs (src/server/analyze_job.py):

  * resolve the best available player weights (trained padel model or the
    COCO person fallback so things work while training runs),
  * build a PadelAnalyzer wired with the manual court calibration for a source,
  * resolve a media source to a local file path (download via yt-dlp when the
    source is a YouTube URL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_player_weights(requested_weights: Optional[str] = None):
    """Return (weights_path, classes, label).

    classes is None for the padel model, [0] (person) for the COCO fallback.
    """
    if requested_weights and Path(requested_weights).exists():
        return requested_weights, None, str(requested_weights)
    padel = PROJECT_ROOT / "data" / "models" / "player_best.pt"
    if padel.exists():
        return str(padel), None, str(padel)
    return "yolo11n.pt", [0], "yolo11n.pt (person fallback)"


def build_analyzer(source: str, requested_weights: Optional[str] = None,
                   device: str = "0", online_reid: bool = False):
    """Build a PadelAnalyzer with the manual calibration for `source`.

    Automatically applies a cached global Re-ID mapping (raw_id -> canonical
    1..4) if one exists for this source, so the live stream shows stable player
    IDs without any extra compute. Returns (analyzer, label, calibrated).

    If ``online_reid`` is True, an OnlineReIDResolver is attached so P1..P4 are
    produced live during the warmup/match (Phase 4) instead of needing a cached
    offline mapping.
    """
    from src.analyzer import PadelAnalyzer
    from src.utils.calibration import load_for_source
    from src.server.reid_resolver import ReIDResolver

    weights, classes, label = resolve_player_weights(requested_weights)
    cal = load_for_source(source)

    resolver = None
    reid_mapping = None
    if online_reid:
        from src.reid_online import OnlineReIDResolver
        resolver = OnlineReIDResolver(device=device)
        label += " + onlineReID"
    else:
        # autoload a precomputed global Re-ID mapping for this source (instant)
        reid_resolver = ReIDResolver.load_cached(source, weights)
        reid_mapping = reid_resolver.mapping if reid_resolver else None
        if reid_mapping:
            label += " + reID"

    analyzer = PadelAnalyzer(
        player_weights=weights,
        device=device,
        manual_calibration=cal,
        player_classes=classes,
        reid_mapping=reid_mapping,
        online_reid=resolver,
    )
    return analyzer, label, cal is not None


# ── media resolution ────────────────────────────────────────────────────────
def resolve_media(source: str, youtube_url: Optional[str] = None) -> str:
    """Resolve a source to a local playable path (download via yt-dlp if needed)."""
    s = str(source)
    if s and Path(s).exists():
        return s
    if youtube_url or ("youtube" in s.lower() or "youtu.be" in s.lower()):
        url = youtube_url or s
        out = download_youtube(url)
        if out:
            return out
    if s.isdigit():
        return s
    if s.startswith(("rtsp://", "http://", "https://")):
        return s
    raise RuntimeError(f"Cannot resolve media source: {s!r}")


def download_youtube(url: str) -> Optional[str]:
    out_dir = PROJECT_ROOT / "data" / "sample_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(out_dir / "yt_%(id)s.%(ext)s")
    try:
        import yt_dlp
    except ImportError:  # pragma: no cover
        print("[media] yt-dlp not installed")
        return None
    opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": out_tpl,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    print(f"[media] downloading YouTube: {url}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = ydl.prepare_filename(info)
            p = Path(downloaded)
            if not p.exists():
                cand = list(out_dir.glob(f"yt_{info.get('id')}.*"))
                if cand:
                    p = cand[0]
            print(f"[media] ready: {p}")
            return str(p)
    except Exception as e:  # pragma: no cover
        print(f"[media] yt-dlp failed: {e}")
        return None
