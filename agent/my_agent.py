"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

WHAT WE KNOW (from hands-on exploration of the 25 public games):

GAME TYPES:
  1. Keyboard-slide (ar25, wa30, ls20, g50t, tr87, ...):
       - One or more movable pieces navigated with A1/A2/A3/A4.
       - Direction mapping varies per game — we learn it from the first moves.
       - Win = move piece onto target zone.

  2. Click-only (vc33, tn36, r11l, s5i5, su15, lf52, lp85, ...):
       - Only ACTION6 (click) is available or useful.
       - Every click advances internal sprite state; win condition is internal.
       - Strategy: repeatedly click every non-background cell systematically.
       - VC33 confirmed: won level 1 in ~14-40 random clicks on non-bg cells.

  3. Keyboard+click (most of the remaining 14 games):
       - Both movement and clicking are needed.
       - Fall back to hybrid: try movement toward target + click non-bg cells.

GENERAL PRINCIPLES:
  - Background = most frequent color.
  - Target zone = cyan (11) if present, else smallest stationary cluster.
  - Piece = color that moves most between frames.
  - Stall detection: if grid doesn't change for 3+ steps, try next strategy.
  - Learn direction mapping: probe A1-A4 and record which moves the piece.
  - After GAME_OVER: RESET and retry (each retry adds more randomness).
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class MyAgent(Agent):
    """
    Multi-strategy agent covering keyboard, click, and hybrid games.

    Phase 1 (first ~8 steps): probe each direction action to learn which
      one moves the piece and in what direction.
    Phase 2: navigate piece toward target using learned mapping.
    Click games: cycle through all non-bg cells with ACTION6.
    Stall recovery: try untried directions, then A5, then random.
    """

    MAX_ACTIONS = 200

    # The 4 direction actions
    DIR_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2,
                   GameAction.ACTION3, GameAction.ACTION4]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_episode()

    def _reset_episode(self) -> None:
        """Reset all per-episode tracking."""
        self._last_grid     = None   # previous frame grid
        self._stall_count   = 0      # consecutive steps with no grid change
        self._step          = 0      # total steps this episode
        self._retry         = 0      # how many GAME_OVERs we've had

        # Direction learning: maps GameAction → (delta_row, delta_col) for piece
        self._dir_map: dict[GameAction, tuple[float, float]] = {}
        self._probe_queue   = list(self.DIR_ACTIONS)  # actions left to probe
        self._probing       = True    # still in probe phase

        # Click game state: list of (col, row) cells to click through
        self._click_cells:  list[tuple[int,int]] = []
        self._click_idx     = 0
        self._is_click_game = False   # detected once on first frame

        # Stall recovery: rotate through untried directions
        self._last_dir      = None
        self._dir_tries     = 0      # consecutive tries of same direction

        random.seed(hash(self.game_id) % 99991 + 7)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ── grid helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_grid(frame: FrameData) -> np.ndarray:
        raw = frame.frame
        # frame.frame can be a list of layers or a single array
        layer = raw[0] if (isinstance(raw, (list, tuple)) and len(raw) > 0) else raw
        arr = np.array(layer, dtype=np.int16)
        arr[arr == -1] = 0
        return arr

    @staticmethod
    def _dominant(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    @staticmethod
    def _centroid(grid: np.ndarray, color: int):
        mask = grid == color
        if not mask.any(): return None
        rs, cs = np.where(mask)
        return float(rs.mean()), float(cs.mean())

    def _moving_color(self, g0: np.ndarray, g1: np.ndarray, bg: int) -> int | None:
        """Return the non-bg color whose centroid moved the most between frames."""
        if g0.shape != g1.shape or g0.size == 0 or g1.size == 0:
            return None
        diff_mask = g0 != g1
        if not diff_mask.any(): return None
        changed_colors = set(int(v) for v in np.unique(g0[diff_mask])) | \
                         set(int(v) for v in np.unique(g1[diff_mask]))
        changed_colors -= {bg, 0}
        best_color, best_dist = None, 0.0
        for color in changed_colors:
            c0 = self._centroid(g0, color)
            c1 = self._centroid(g1, color)
            if c0 and c1:
                dist = abs(c0[0]-c1[0]) + abs(c0[1]-c1[1])
                if dist > best_dist:
                    best_dist = dist
                    best_color = color
        return best_color

    def _build_click_cells(self, grid: np.ndarray, bg: int) -> list[tuple[int,int]]:
        """
        Build ordered list of (col, row) cells to click.
        Prioritises small clusters (likely interactive sprites).
        Densely samples every non-bg cell so cycling through them
        covers the whole interactive area.
        """
        vals, counts = np.unique(grid, return_counts=True)
        # Sort colors: small clusters first (more likely to be targets/buttons)
        color_order = [int(v) for v,c in sorted(zip(vals,counts), key=lambda x:x[1])
                       if int(v) not in (bg, 0)]
        cells = []
        seen = set()
        for color in color_order:
            mask = (grid == color)
            rs, cs = np.where(mask)
            # Take up to 16 evenly-spaced cells per color
            n = min(16, len(rs))
            idxs = np.linspace(0, len(rs)-1, n, dtype=int)
            for i in idxs:
                key = (int(cs[i]), int(rs[i]))
                if key not in seen:
                    cells.append(key)
                    seen.add(key)
        # Shuffle slightly so repeated cycles explore different orderings
        random.shuffle(cells)
        return cells

    # ── main decision ─────────────────────────────────────────────────────────

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:

        # ── RESET on game-over / not started ─────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._retry += 1
            self._last_grid   = None
            self._stall_count = 0
            self._step        = 0
            self._probing     = True
            self._probe_queue = list(self.DIR_ACTIONS)
            self._dir_map     = {}
            self._click_cells = []
            self._click_idx   = 0
            return GameAction.RESET

        self._step += 1
        grid  = self._to_grid(latest_frame)
        avail = list(latest_frame.available_actions or [])
        bg    = self._dominant(grid)

        # ── Detect click-only game on first real step ─────────────────────
        if self._step == 1:
            # Click-only game: ONLY action 6 available (or no 1-4)
            dir_avail_check = [a for a in (1,2,3,4) if a in avail]
            self._is_click_game = (len(dir_avail_check) == 0)
            # Build click cell list regardless (used as fallback too)
            self._click_cells = self._build_click_cells(grid, bg)
            self._click_idx   = 0
            # Pre-seed the AR25-verified direction mapping as default assumption.
            # A1=up(-row), A2=down(+row), A3=right(+col), A4=left(-col).
            # Probe phase will correct this if wrong.
            self._dir_map = {
                GameAction.ACTION1: (-3.0,  0.0),
                GameAction.ACTION2: ( 3.0,  0.0),
                GameAction.ACTION3: ( 0.0,  3.0),
                GameAction.ACTION4: ( 0.0, -3.0),
            }

        # ── Stall detection ───────────────────────────────────────────────
        changed = True
        if self._last_grid is not None:
            changed = bool((grid != self._last_grid).any())
        self._stall_count = 0 if changed else self._stall_count + 1
        self._last_grid = grid.copy()

        # ── CLICK-ONLY GAME STRATEGY ──────────────────────────────────────
        if self._is_click_game or (6 in avail and not any(a in avail for a in (1,2,3,4))):
            # Rebuild click list every 2 cycles (grid changes)
            if self._click_idx >= len(self._click_cells) or self._step == 1:
                self._click_cells = self._build_click_cells(grid, bg)
                self._click_idx   = 0

            if self._click_cells:
                col, row = self._click_cells[self._click_idx % len(self._click_cells)]
                self._click_idx += 1
                act = GameAction.ACTION6
                act.set_data({"x": col, "y": row})
                return act

        # ── KEYBOARD / HYBRID GAME STRATEGY ──────────────────────────────

        dir_avail = [a for a in self.DIR_ACTIONS if a.value in avail]

        # ── Phase 1: probe directions to learn mapping ────────────────────
        if self._probing and self._probe_queue and self._last_grid is not None:
            # We just executed an action — record what moved
            if len(frames) >= 2:
                try:
                    prev_frame = frames[-2]
                    prev_grid  = self._to_grid(prev_frame)
                    if prev_grid.shape == (64,64) and prev_grid.size > 0:
                        mc = self._moving_color(prev_grid, grid, bg)
                        if mc and len(self._probe_queue) < len(self.DIR_ACTIONS):
                            tried_idx = len(self.DIR_ACTIONS) - len(self._probe_queue) - 1
                            if 0 <= tried_idx < len(self.DIR_ACTIONS):
                                tried_act = self.DIR_ACTIONS[tried_idx]
                                c_prev = self._centroid(prev_grid, mc)
                                c_curr = self._centroid(grid, mc)
                                if c_prev and c_curr:
                                    dr = c_curr[0] - c_prev[0]
                                    dc = c_curr[1] - c_prev[1]
                                    if abs(dr) > 0.5 or abs(dc) > 0.5:
                                        self._dir_map[tried_act] = (dr, dc)
                except Exception:
                    pass

            # Fire next probe action
            if self._probe_queue:
                next_probe = self._probe_queue.pop(0)
                if next_probe in dir_avail:
                    return next_probe
            else:
                self._probing = False

        if not self._probe_queue:
            self._probing = False

        # ── Find piece and target ─────────────────────────────────────────
        # Piece = smallest moving color (prefer what dir_map found)
        piece_color = None
        if self._dir_map:
            # Use a color that was observed moving
            for act, (dr,dc) in self._dir_map.items():
                if abs(dr)+abs(dc) > 0.5:
                    # Find color that moved with this action
                    if len(frames) >= 2:
                        mc = self._moving_color(
                            self._to_grid(frames[-2]) if len(frames) >= 2 else grid,
                            grid, bg)
                        if mc:
                            piece_color = mc
                            break

        if piece_color is None:
            # Fallback: prefer small non-bg clusters not equal to 9,10,11 (those tend to be targets/UI)
            vals, counts = np.unique(grid, return_counts=True)
            candidates = [(int(v),int(c)) for v,c in zip(vals,counts)
                          if int(v) not in (bg,0,9,10,11) and int(c) < 200]
            if candidates:
                piece_color = min(candidates, key=lambda x:x[1])[0]

        # Target = cyan(11), else color9 (maroon), else smallest stationary cluster
        target_color = None
        if 11 in np.unique(grid) and 11 != bg:
            target_color = 11
        elif 9 in np.unique(grid) and 9 != bg:
            target_color = 9
        else:
            vals, counts = np.unique(grid, return_counts=True)
            # Pick smallest non-bg, non-piece cluster
            candidates = [(int(v),int(c)) for v,c in zip(vals,counts)
                          if int(v) not in (bg, 0, piece_color or -1)]
            if candidates:
                target_color = min(candidates, key=lambda x:x[1])[0]

        # ── Stall recovery ────────────────────────────────────────────────
        if self._stall_count >= 3:
            self._stall_count = 0
            # Try A5 (piece cycle / special action)
            if GameAction.ACTION5.value in avail:
                return GameAction.ACTION5
            # Try each direction we haven't tried recently
            if dir_avail:
                return random.choice(dir_avail)
            # Try click on a non-bg cell
            if 6 in avail and self._click_cells:
                col, row = self._click_cells[self._click_idx % len(self._click_cells)]
                self._click_idx += 1
                act = GameAction.ACTION6
                act.set_data({"x": col, "y": row})
                return act

        # ── Move piece toward target using learned or inferred mapping ────
        if piece_color and target_color:
            pc = self._centroid(grid, piece_color)
            tc = self._centroid(grid, target_color)

            if pc and tc:
                dr_needed = tc[0] - pc[0]  # positive = need to go down
                dc_needed = tc[1] - pc[1]  # positive = need to go right

                if self._dir_map:
                    # Use learned mapping: find action whose direction best aligns
                    best_act, best_score = None, -999.0
                    for act, (dr, dc) in self._dir_map.items():
                        if act not in dir_avail: continue
                        # Dot product: higher = more aligned with needed direction
                        score = dr * dr_needed + dc * dc_needed
                        if score > best_score:
                            best_score = score
                            best_act = act
                    if best_act and best_score > 0:
                        return best_act

                # No learned mapping or no good match — use AR25-verified fallback
                # (A1=up, A2=down, A3=right, A4=left confirmed for AR25)
                if abs(dr_needed) >= abs(dc_needed):
                    act = GameAction.ACTION1 if dr_needed < 0 else GameAction.ACTION2
                else:
                    act = GameAction.ACTION3 if dc_needed > 0 else GameAction.ACTION4
                if act in dir_avail:
                    return act

        # ── A5 if available (piece cycle in multi-piece games) ────────────
        if GameAction.ACTION5.value in avail and self._step % 8 == 0:
            return GameAction.ACTION5

        # ── Hybrid: also try clicking non-bg cells ────────────────────────
        if 6 in avail and self._step % 5 == 0 and self._click_cells:
            col, row = self._click_cells[self._click_idx % len(self._click_cells)]
            self._click_idx += 1
            act = GameAction.ACTION6
            act.set_data({"x": col, "y": row})
            return act

        # ── Last resort: random available direction ───────────────────────
        if dir_avail:
            # Add some retry-based noise
            if self._retry > 1:
                return random.choice(dir_avail)
            return dir_avail[self._step % len(dir_avail)]

        # Absolute fallback
        return GameAction.ACTION1
