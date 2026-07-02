# Match lifecycle & live Re-ID (Phase 4)

How a padel session flows from camera setup to a scored match, and how player
IDs (P1..P4) stay stable **during** the match.

Related: [`ROADMAP.md`](ROADMAP.md) (Component 5 / production), the online
resolver `src/reid_online.py`, the state machine `src/match_state.py`.

---

## 1. Session flow

```
 camera setup (Jetson)        WARMUP                         LIVE                     ENDED
 ─────────────────────       ─────────────────────────      ──────────────────        ────────
 place camera                capture running                match started             user stops
 click 4 court corners  →    Re-ID clusters P1..P4     →    IDs LOCKED, stats    →    final report
 (manual calibration)        (stats NOT counted)            count from here           (Phase 7)
```

**Court calibration is a one-time setup step on the Jetson**, performed when the
camera is placed: open the dashboard, click the 4 back-wall court corners
(`POST /api/calibrate`), and the homography is persisted per camera source under
`data/calibrations/`. It is NOT part of the match — Phase 4 assumes it is done
before LIVE.

## 2. Warmup is the Re-ID calibration window

Recreational players warm up 5–10 minutes before the real match. That footage
does not count, so we use it as the calibration window: the
`OnlineReIDResolver` accumulates BoT-SORT tracklets and periodically re-clusters
them (reusing the batch `src/reid.py` internals) to refine the `raw_id → P1..P4`
mapping. By the time the match starts, IDs are already stable — the dashboard
never shows wrong IDs during actual play.

The expensive appearance extraction (DINOv2/OSNet) runs **only during warmup**.
In LIVE the resolver is frozen and the per-frame cost is just a position-EMA
update plus a nearest-slot lookup, so live throughput is unaffected.

## 3. Match start — button OR auto-detect, button authoritative

A match transitions `WARMUP → LIVE` on the first of:

* the user clicks **Start Match** (`POST /api/match/start`), or
* serve auto-detect fires (`src/match_state.detect_serve_start`).

**Whichever comes first starts LIVE — but the button is authoritative:** if
auto-detect already started LIVE and the user later clicks Start, the stats
counted since auto-detect are **discarded as warmup** and LIVE re-anchors to the
button moment. If the user never clicks, auto-detect's time stands.

This makes the button a safety override against auto-detect false positives.

> **Auto-detect is currently a STUB** (`detect_serve_start` returns `False`).
> Reliable serve detection needs the ball model (Phase 1c, TrackNet), so for now
> **only the button can start the match**. Wire the real detector in once
> `data/models/ball_best.pt` exists.

## 4. Position-based P1..P4 labels

The resolver canonicalises the 4 clusters by court position
(`mean_y, mean_x` — same key as the batch `_canonicalize`), so the labels are
deterministic: e.g. P1 = near-left, etc. (Decision: position-based labeling is
accepted; no per-match "who is P1" UI pick.)

## 5. Automatic team assignment (Phase 5)

Padel doubles: each pair occupies one half of the court (split by the net at
Y = 10 m). `src/team_assigner.py::TeamAssigner` tracks a per-slot EMA of
court-Y and, once it has stabilised over ~240 frames (the warmup window), locks
each slot to **team 0** (near half) or **team 1** (far half) — the classic
**P1+P2 vs P3+P4** split, driven by actual mean court-Y so it stays correct even
if the canonical ordering is noisy. `Player.team` is set each frame and surfaces
in the `/stats` snapshot (`team` per player + a `teams` block); the rules
engine's coarse point winner (`rules_engine._team_of`) consumes it.

## 6. Warmup data is discarded from stats

`StatsAccumulator.update(...)` is only called while `MatchState.counting`
(`state == LIVE`). On `Start Match`, `stats_acc.reset()` discards anything
accumulated during warmup (or during a prior auto-start that the button
overrides). `MatchState.discarded_warmup_frames` records how many frames were
thrown away for reporting.

## 7. API & UI

| Endpoint | Effect |
|----------|--------|
| `POST /api/match/start` | authoritative LIVE start; locks Re-ID; resets stats |
| `POST /api/match/end` | `LIVE → ENDED` |
| `POST /api/match/reset` | back to `WARMUP` for a new match on the same stream |
| `GET /api/heatmap/{slot}` | per-player position heatmap PNG (KDE over LIVE court positions) |
| `GET /stats` | `match`, `online_reid`, `teams`, `score`, `shot_log`, `rallies`, `positions_slots` |

Dashboard (`templates/index.html`): a `match: warmup|live|ended` pill (showing
`mapped/tracklets` IDs while warming up) plus the **Start Match** button, a live
**score** (Team A vs B) + per-team shot aggregate, a scrolling **shot log**
(player · team · shot type · frame), and a **position heatmaps** grid (one KDE
court overlay per player, refreshed every 5 s). All Phase 6 panels populate only
during LIVE (warmup is discarded).

## 8. Post-match report (Phase 7)

`src/match_report.py` assembles the definitive, exportable summary from the
pipeline's accumulated LIVE state and writes it to
`data/reports/<source_id>/match_report.json` (+ final `heat_P{slot}.png`
overlays). Shape: `meta` (source, duration, lifecycle), `score`, `rallies`,
`total_shots`, `per_player` (time on court, distance, shots/by-type, heatmap),
`per_team` (players, shots, points), `shot_log`. The canonical slots are already
stable (locked at match-start), so no re-clustering is needed for v1.

| Endpoint | Effect |
|----------|--------|
| `POST /api/match/report` | build + save the report for the current source |
| `GET /api/match/report` | fetch the last saved report (404 if none) |
| `GET /api/match/report/asset/{name}` | serve a saved heatmap PNG |
| `GET /report` | the report view (`templates/report.html`) |
