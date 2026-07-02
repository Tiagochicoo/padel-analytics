# Scoring spec — full padel match scoring

Scope of the **v1 rules engine** (`src/rules_engine.py`) and what is required to
upgrade it to **full official padel scoring** (FIP / World Padel Tour rules).

The v1 engine only segments rallies/points and attributes shots to players. It
does **not** call a score. This document defines the additional detection
primitives, court zones, state machines, and inputs needed to compute an
official score end-to-end.

---

## 1. What v1 provides (baseline)

`src/rules_engine.py` (v1) consumes the analyzer outputs and produces:

- **Rally segmentation** — IDLE → SERVING → RALLY → POINT_SCORED transitions,
  keyed on ball motion + player ball-proximity events.
- **Shot attribution** — each player hit (`{slot, type, frame}` from
  `src/shot_classifier.py`) is bound to a player and a timestamp.
- **Point winner side** — coarse (which team last touched the ball before the
  rally ended), but **without bounce/in/out reasoning**, so it is not
  official.

v1 does **not** know: serve box landing, faults, double faults, bounce count,
in/out, deuce/golden-point, game/set/match state.

---

## 2. Padal scoring rules (official)

Padel uses tennis-style scoring.

### 2.1 Points in a game
`0 (love) → 15 → 30 → 40 → game`.

### 2.2 Deuce / Golden Point
- **Golden Point** (modern FIP rule, since 2020): at 40–40 the **next point
  decides the game** — no advantage. This is the current pro standard.
- (Legacy / some amateur formats still use tennis Advantage: deuce → need two
  consecutive points. Configurable.)

### 2.3 Games in a set
First to **6 games, win by 2**. At **6–6** a **tie-break** is played.

### 2.4 Tie-break
- Points counted **1, 2, 3, …** to **7**, win by 2.
- Serve rotation during tie-break: **A→B→B→A→A→B…** (1 serve, then 2 each).
- The tie-break winner takes the set **7–6**.

### 2.5 Sets / match
- Pro padel: **best of 3 sets**.
- Some formats replace a deciding 3rd set with a **super tie-break to 10**
  (match tie-break). Configurable.

### 2.6 Serve rules (the detection-hard part)
- **Diagonal serve** into the opponent's service box.
- The server gets **two serves** (1st + 2nd). A serve is a **fault** if the
  ball hits the net or lands outside the correct service box.
- **Double fault** (both serves fault) → point to the receiving team.
- **Let** (serve clips net, lands in box) → serve retaken.
- **Serve rotation:** players alternate as server each game in the order
  Server₁ → Server₂ → Server₃ → Server₄ (across both teams). Serve **side**
  (deuce/right vs advantage/left) **alternates every point** within a game.

---

## 3. Detection primitives required for full scoring

These are the missing CV/perception signals. Each must exist before the rules
engine can call an official score.

| # | Primitive | Source | Status |
|---|-----------|--------|--------|
| P1 | **Bounce detection** (ball ground contact + frame) | TrackNetV3 trajectory + velocity-discontinuity / physics model | **NOT IMPLEMENTED** — depends on Component 3 (TrackNetV3) |
| P2 | **Bounce court-zone classification** (which zone a bounce lands in) | homography (`src/utils/homography.py`) + court keypoints | homography math ready; zone predicates to add |
| P3 | **Serve-motion detection** (ball toss + first contact) | body pose (`configs/bodypose.yaml`) + ball trajectory | needs pose + shot classifier |
| P4 | **Last-hitter identity** (who hit last before a terminal event) | shot attribution (v1) | v1 provides |
| P5 | **Net-touch detection** (ball enters net plane) | ball trajectory + net keypoints (`NET_INDICES`) | NOT IMPLEMENTED |
| P6 | **Serve box landing** (1st/2nd serve bounce in correct diagonal box) | P1 + P2 + serve-side state | NOT IMPLEMENTED |

### Court zones (top-down, metres) — derived from court keypoints
Using the 20×10 m template in `src/utils/homography.py`:

- **Service boxes (near half):** X∈[0,5]∪[5,10], Y∈[0, 6.95] → left/right near.
- **Service boxes (far half):** X∈[0,5]∪[5,10], Y∈[13.05, 20] → left/right far.
- **In-play area:** inside the 4 back-wall corners + service lines (poly via
  `COURT_POLYGON_INDICES`).
- **Net plane:** Y = 10 (the `NET_INDICES` keypoints).
- **Out:** outside the in-play polygon.

A serve is "in the correct box" iff its bounce (P1) projected to court metres
(P2) lands in the diagonal service box for the current server + side.

---

## 4. Rules-engine state machine (v2 — full scoring)

The v2 engine extends v1's IDLE→SERVING→RALLY→POINT_SCORED with scoring state.

### 4.1 Match-level state (persisted across the whole match)
```
match = {
  sets: [ {games_a, games_b, points: [...], tiebreak: bool}, ... ],  # best of 3
  current_game: { points_a, points_b },                              # 0/15/30/40
  server_order: [p1, p2, p3, p4],                                    # set once per set
  server_idx, serve_side,                                            # advance each game / point
  first_or_second_serve,                                             # 1 or 2
  format: {golden_point: true, deciding_set: 'super_tb'|'set'},
}
```

### 4.2 Per-point decision (the core)
On each terminal event (rally end), classify the **point-ending event**:

| Event | Detection | Point goes to |
|-------|-----------|---------------|
| Double fault | two serve faults (P6) | receiving team |
| Serve fault (1st) then 2nd in | P6 | continue rally |
| Bounce twice on team A's side | P1 (two consecutive bounces, no hit between) | team B |
| Ball bounces out of court | P1 + P2 (out polygon) | team not-last-hitting |
| Ball in net | P5 | team not-last-hitting |
| Winner (forced) | last hit by team X, then unplayable bounce on team Y | team X |

Then apply point → game (15/30/40/golden-point) → set (6 / tie-break) → match.

### 4.3 Serve-side & rotation tracking
- **Server** advances one slot per game (4-player rotation).
- **Serve side** alternates every point within a game (deuce ↔ advantage).
- First serve of each point; on fault → second serve; on second fault → point.

---

## 5. Inputs the engine needs (dependencies)

| Input | From | v1 has it? |
|-------|------|-----------|
| Per-frame ball xy + trail | `src/ball_tracker.py` (TrackNetV3) | **NO** (Component 3) |
| Bounce events (frame, xy) | ball trajectory post-processing | **NO** |
| Court homography (live) | `PadelAnalyzer` + court keypoints / manual cal | yes (manual now) |
| Shot events (player, type, frame) | `src/shot_classifier.py` | **NO** (Component 4) |
| Player slots P1..P4 | online Re-ID resolver | **NO** (Phase 4) |
| Team of each player | court-side assignment | **NO** (Phase 5) |
| Serve-order / side setup | UI at match start (or inferred from first points) | **NO** (UX) |

**Conclusion:** full scoring is gated on **Components 3 + 4** (ball + shots),
the **online Re-ID** and **team assignment** phases, plus a new **bounce
detector**. It is the last feature, not the first.

---

## 6. Build order for scoring (v1 → v2)

1. **v1 (this roadmap, Phase 2):** rally/point segmentation + shot attribution.
   Output: `events.jsonl` with rallies, points (winner side coarse), shots.
2. **Bounce detector** (new module, after Component 3 TrackNetV3 lands):
   `src/bounce_detector.py` — frame + court-metre xy of each bounce.
3. **Zone predicates** in `src/utils/homography.py`: `is_in_service_box(xy, side)`,
   `is_in_play(xy)`, `is_out(xy)`.
4. **Serve detector** (after Component 4 body pose): identify serve motion +
   1st/2nd serve + serve-side state.
5. **v2 rules engine:** point-ending classification + game/set/match state
   machine + serve rotation + golden point + tie-break/super-tie-break.
6. **Match-setup UI:** pick server order + starting side + format flags
   (golden point on/off, deciding-set format).

---

## 7. Open product questions (resolve before v2)

- **Golden point** default on (current pro standard) or off (legacy advantage)?
- **Deciding set**: full 3rd set (WPT) or super tie-break to 10 (amateur)?
- **Serve-order source**: user sets it at match start (reliable), or infer from
  the first ~3 points (convenient but error-prone)?
- **Score correction UI**: allow the user to override a mis-called point? (A CV
  score will never be 100 % correct, so an override path is strongly advised.)
- **Live vs final score**: show the live best-estimate score during the match
  (may flip as late bounces resolve) and lock the definitive score post-match?
