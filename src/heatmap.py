"""
src/heatmap.py
==============
Render per-player position heatmaps on the 20x10 m padel court template.

Input: per-slot lists of ``(x, y)`` court-plane positions in METRES, as produced
by projecting each player's feet through the homography (``Player.court_xy``).
Output: PNG bytes for each player — a KDE density field overlaid on a drawn
court (boundary, net, service lines), suitable for the live dashboard and the
post-match ``match_report``.

Court coordinate system matches ``src/utils/homography.py``:
    X: 0 (left) -> 10 (right) metres (court width)
    Y: 0 (near/camera) -> 20 (far) metres (court length)
"""

from __future__ import annotations

import io
from typing import Iterable

# Object-oriented Agg backend — avoids matplotlib.use()/pyplot entirely, which
# races with partial pyplot init under the web server (no display needed).
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import gaussian_kde

from src.utils.homography import COURT_LENGTH_M, COURT_WIDTH_M, SERVICE_LINE_M

NET_Y = COURT_LENGTH_M / 2.0              # 10.0 m
SERVICE_FAR_Y = COURT_LENGTH_M - SERVICE_LINE_M   # 13.05 m

_COURT_X = (0.0, COURT_WIDTH_M)           # 0..10
_COURT_Y = (0.0, COURT_LENGTH_M)          # 0..20
_VALID_X = (-1.0, COURT_WIDTH_M + 1.0)    # tolerate slight projection noise
_VALID_Y = (-1.0, COURT_LENGTH_M + 1.0)


def _filter(positions: Iterable[tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(list(positions), dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    mask = (
        np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= _VALID_X[0]) & (pts[:, 0] <= _VALID_X[1])
        & (pts[:, 1] >= _VALID_Y[0]) & (pts[:, 1] <= _VALID_Y[1])
    )
    return pts[mask]


def _draw_court(ax) -> None:
    """Draw an accurate padel court top-down (20m × 10m)."""
    W, L = COURT_WIDTH_M, COURT_LENGTH_M       # 10, 20
    CX = W / 2                                  # centre X = 5
    SL_NEAR = SERVICE_LINE_M                    # 6.95
    SL_FAR = COURT_LENGTH_M - SERVICE_LINE_M    # 13.05
    NET = COURT_LENGTH_M / 2                    # 10.0

    # Service box background fill (subtle green tint)
    ax.add_patch(Rectangle((0, 0), W, SL_NEAR, fill=True, facecolor="#0e2a1e",
                           edgecolor="none", zorder=0))   # near service area
    ax.add_patch(Rectangle((0, SL_FAR), W, L - SL_FAR, fill=True, facecolor="#0e2a1e",
                           edgecolor="none", zorder=0))   # far service area

    # Outer court boundary
    ax.add_patch(Rectangle((0, 0), W, L, fill=False, lw=2.0, ec="#e6edf3", zorder=3))

    # Net (thick dashed line with a slightly different style)
    ax.plot([0, W], [NET, NET], color="#f0f0f0", lw=2.5, ls="-", alpha=0.6, zorder=2)
    ax.plot([0, W], [NET, NET], color="#e6edf3", lw=1.0, ls="--", zorder=3)

    # Service lines (full width)
    ax.plot([0, W], [SL_NEAR, SL_NEAR], color="#94a3b8", lw=1.4, zorder=3)
    ax.plot([0, W], [SL_FAR, SL_FAR], color="#94a3b8", lw=1.4, zorder=3)

    # Centre service line — two segments, service line → net on each side
    ax.plot([CX, CX], [SL_NEAR, NET], color="#94a3b8", lw=1.0, ls=":", zorder=3)  # near
    ax.plot([CX, CX], [NET, SL_FAR], color="#94a3b8", lw=1.0, ls=":", zorder=3)   # far

    ax.set_xlim(0, W)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_facecolor("#0a3d2a")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_heatmap(positions: Iterable[tuple[float, float]], title: str = "",
                   cmap: str = "magma") -> bytes:
    """Render one KDE heatmap to PNG bytes. Returns PNG bytes (always non-empty)."""
    pts = _filter(positions)
    fig = Figure(figsize=(4, 8), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _draw_court(ax)

    if len(pts) >= 2:
        try:
            kde = gaussian_kde(pts.T, bw_method=0.15)
            xs = np.linspace(0, 10, 120)
            ys = np.linspace(0, 20, 240)
            xx, yy = np.meshgrid(xs, ys)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            zz[zz < zz.max() * 0.05] = np.nan  # fade tails for a clean overlay
            ax.imshow(zz, extent=(0, COURT_WIDTH_M, 0, COURT_LENGTH_M),
                      origin="lower", cmap=cmap, alpha=0.75, aspect="auto",
                      interpolation="bilinear")
        except np.linalg.LinAlgError:
            ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#f97316", alpha=0.5)
    elif len(pts) == 1:
        ax.scatter(pts[:, 0], pts[:, 1], s=30, c="#f97316", alpha=0.8)

    if title:
        ax.set_title(title, color="#e6edf3", fontsize=11, pad=8)
    fig.patch.set_facecolor("#0d1117")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()


def render_player_heatmaps(positions_by_slot: dict[int, list[tuple[float, float]]],
                           slot_names: dict[int, str] | None = None) -> dict[int, bytes]:
    """Render a heatmap PNG per slot. Skips slots with no positions."""
    out: dict[int, bytes] = {}
    for slot, pts in positions_by_slot.items():
        if not pts:
            continue
        name = (slot_names or {}).get(slot, f"Player {slot}")
        out[slot] = render_heatmap(pts, title=f"{name} — position heat")
    return out


__all__ = ["render_heatmap", "render_player_heatmaps"]
