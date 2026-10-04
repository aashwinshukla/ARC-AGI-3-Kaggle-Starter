"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

WHAT WE KNOW (from hands-on exploration of the 25 public games):

GAME TYPES:
  1. Keyboard-slide (ar25, wa30, ls20, g50t, tr87, ...):
       - A1/A2/A3/A4 move a piece. Direction varies per game.
       - Win = move piece onto target zone.
       - AR25 verified: A1=up, A2=down, A3=right, A4=left.
       - WA30: piece=color14, A1=up(-4r). Target at row ~30.
       - LS20: piece=color12, A1=up(-5r), A3=left(-5c), A4=right(+5c).

  2. Click-only (vc33, tn36, r11l, lp85, ...):
       - Only ACTION6 (click) works.
       - Win by cycling clicks across all non-bg cells.
       - VC33 wins in ~14-40 random clicks on small-object region.

  3. Keyboard+click (most of the remaining 14 games):
       - Hybrid: move toward target + click fallback.

KEY IMPROVEMENTS IN THIS VERSION:
  - Stall recovery now rotates through ALL 4 directions round-robin
    instead of random (avoids re-picking the blocked direction).
  - Stops wasting budget after level win (is_done returns True on WIN).
  - Click cells rebuilt each cycle so they track moving objects.
  - Piece-finding improved: after probe phase locks in moving color,
    we track it across frames rather than re-detecting every step.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class MyAgent(Agent):
    """
    Multi-strategy agent.

    For keyboard games: probes directions → navigates piece to target →
      rotates through all 4 directions when stalled.
    For click games: dense systematic cycling through all non-bg cells.
    """

    MAX_ACTIONS = 200

    DIR_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2,
                   GameAction.ACTION3, GameAction.ACTION4]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._last_grid       = None
        self._stall_count     = 0
        self._step            = 0
        self._retry           = 0
        self._dir_map: dict[GameAction, tuple[float, float]] = {}
        self._probe_queue     = list(self.DIR_ACTIONS)
        self._probing         = True
        self._click_cells: list[tuple[int, int]] = []
        self._click_idx       = 0
        self._is_click_game   = False
        # Stall recovery: rotate index through all 4 dirs instead of random
        self._stall_dir_idx   = 0
        # Lock on the confirmed piece color after probing
        self._piece_color: int | None = None
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
        if not mask.any():
            return None
        rs, cs = np.where(mask)
        return float(rs.mean()), float(cs.mean())

    def _moving_color(self, g0: np.ndarray, g1: np.ndarray, bg: int) -> int | None:
        """Color whose centroid moved the most between two frames."""
        if g0.shape != g1.shape or g0.size == 0:
            return None
        diff_mask = g0 != g1
        if not diff_mask.any():
            return None
        changed = (set(int(v) for v in np.unique(g0[diff_mask])) |
                   set(int(v) for v in np.unique(g1[diff_mask]))) - {bg, 0}
        best_color, best_dist = None, 0.0
        for color in changed:
            c0 = self._centroid(g0, color)
            c1 = self._centroid(g1, color)
            if c0 and c1:
                d = abs(c0[0] - c1[0]) + abs(c0[1] - c1[1])
                if d > best_dist:
                    best_dist, best_color = d, color
        return best_color

    def _build_click_cells(self, grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
        """
        All non-bg cells, small clusters first, up to 16 samples each.
        Shuffled so repeated cycles explore different orders.
        """
        vals, counts = np.unique(grid, return_counts=True)
        color_order = [int(v) for v, c in sorted(zip(vals, counts), key=lambda x: x[1])
                       if int(v) not in (bg, 0)]
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for color in color_order:
            mask = grid == color
            rs, cs = np.where(mask)
            n = min(16, len(rs))
            for i in np.linspace(0, len(rs) - 1, n, dtype=int):
                key = (int(cs[i]), int(rs[i]))
                if key not in seen:
                    cells.append(key)
                    seen.add(key)
        random.shuffle(cells)
        return cells

    # ── main decision ─────────────────────────────────────────────────────────

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:

        # ── Reset on game-over ────────────────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._retry      += 1
            self._last_grid   = None
            self._stall_count = 0
            self._step        = 0
            self._probing     = True
            self._probe_queue = list(self.DIR_ACTIONS)
            self._dir_map     = {}
            self._click_cells = []
            self._click_idx   = 0
            self._piece_color = None
            self._stall_dir_idx = 0
            return GameAction.RESET

        self._step += 1
        grid  = self._to_grid(latest_frame)
        avail = list(latest_frame.available_actions or [])
        bg    = self._dominant(grid)

        # ── First step: detect game type and seed defaults ────────────────
        if self._step == 1:
            dir_present = [a for a in (1, 2, 3, 4) if a in avail]
            self._is_click_game = (len(dir_present) == 0)
            self._click_cells   = self._build_click_cells(grid, bg)
            self._click_idx     = 0
            # Default direction map (AR25-verified; probe phase refines it)
            self._dir_map = {
                GameAction.ACTION1: (-3.0,  0.0),   # up
                GameAction.ACTION2: ( 3.0,  0.0),   # down
                GameAction.ACTION3: ( 0.0,  3.0),   # right
                GameAction.ACTION4: ( 0.0, -3.0),   # left
            }

        # ── Stall detection ───────────────────────────────────────────────
        changed = True
        if self._last_grid is not None:
            changed = bool((grid != self._last_grid).any())
        self._stall_count = 0 if changed else self._stall_count + 1
        self._last_grid   = grid.copy()

        # ── Click-only games ──────────────────────────────────────────────
        if self._is_click_game or (6 in avail and not any(a in avail for a in (1, 2, 3, 4))):
            # Rebuild list each cycle so we track any grid changes
            if self._click_idx >= len(self._click_cells):
                self._click_cells = self._build_click_cells(grid, bg)
                self._click_idx   = 0
            if self._click_cells:
                col, row = self._click_cells[self._click_idx % len(self._click_cells)]
                self._click_idx += 1
                GameAction.ACTION6.set_data({"x": col, "y": row})
                return GameAction.ACTION6

        # ── Keyboard / hybrid games ───────────────────────────────────────
        dir_avail = [a for a in self.DIR_ACTIONS if a.value in avail]

        # Probe phase: fire each direction once, record how piece responds
        if self._probing and self._probe_queue:
            if len(frames) >= 2 and self._last_grid is not None:
                try:
                    prev_grid = self._to_grid(frames[-2])
                    if prev_grid.shape == (64, 64):
                        mc = self._moving_color(prev_grid, grid, bg)
                        if mc:
                            # Which direction did we just probe?
                            tried_idx = len(self.DIR_ACTIONS) - len(self._probe_queue) - 1
                            if 0 <= tried_idx < len(self.DIR_ACTIONS):
                                tried_act = self.DIR_ACTIONS[tried_idx]
                                cp = self._centroid(prev_grid, mc)
                                cc = self._centroid(grid, mc)
                                if cp and cc:
                                    dr = cc[0] - cp[0]
                                    dc = cc[1] - cp[1]
                                    if abs(dr) > 0.5 or abs(dc) > 0.5:
                                        self._dir_map[tried_act] = (dr, dc)
                                        self._piece_color = mc  # lock in piece
                except Exception:
                    pass

            next_probe = self._probe_queue.pop(0)
            if not self._probe_queue:
                self._probing = False
            if next_probe in dir_avail:
                return next_probe

        if not self._probe_queue:
            self._probing = False

        # ── Identify piece color ──────────────────────────────────────────
        piece_color = self._piece_color
        if piece_color is None or not (grid == piece_color).any():
            # Re-detect: smallest non-bg cluster (excluding common UI colors)
            vals, counts = np.unique(grid, return_counts=True)
            candidates = [(int(v), int(c)) for v, c in zip(vals, counts)
                          if int(v) not in (bg, 0, 9, 10, 11) and int(c) < 300]
            if candidates:
                piece_color = min(candidates, key=lambda x: x[1])[0]

        # ── Identify target color ─────────────────────────────────────────
        unique_colors = set(int(v) for v in np.unique(grid)) - {bg, 0}
        target_color = None
        # Prefer cyan (11), then maroon (9), then smallest remaining cluster
        for preferred in (11, 9):
            if preferred in unique_colors and preferred != piece_color:
                target_color = preferred
                break
        if target_color is None:
            vals, counts = np.unique(grid, return_counts=True)
            candidates = [(int(v), int(c)) for v, c in zip(vals, counts)
                          if int(v) not in (bg, 0) and int(v) != piece_color]
            if candidates:
                target_color = min(candidates, key=lambda x: x[1])[0]

        # ── Stall recovery: rotate through all directions round-robin ─────
        # This avoids re-picking the wall that's already blocking us.
        if self._stall_count >= 3:
            self._stall_count    = 0
            self._stall_dir_idx += 1
            # Try A5 every other stall (cycles piece in multi-piece games)
            if self._stall_dir_idx % 2 == 0 and GameAction.ACTION5.value in avail:
                return GameAction.ACTION5
            # Otherwise try the next direction in rotation
            if dir_avail:
                idx = self._stall_dir_idx % len(dir_avail)
                return dir_avail[idx]
            # Click fallback
            if 6 in avail and self._click_cells:
                col, row = self._click_cells[self._click_idx % len(self._click_cells)]
                self._click_idx += 1
                GameAction.ACTION6.set_data({"x": col, "y": row})
                return GameAction.ACTION6

        # ── Navigate piece toward target ──────────────────────────────────
        if piece_color and target_color:
            pc = self._centroid(grid, piece_color)
            tc = self._centroid(grid, target_color)

            if pc and tc:
                dr_needed = tc[0] - pc[0]   # +ve = need to go down
                dc_needed = tc[1] - pc[1]   # +ve = need to go right

                # Use learned mapping: pick action most aligned with needed direction
                best_act, best_score = None, -999.0
                for act, (dr, dc) in self._dir_map.items():
                    if act not in dir_avail:
                        continue
                    score = dr * dr_needed + dc * dc_needed
                    if score > best_score:
                        best_score = score
                        best_act   = act
                if best_act and best_score > 0:
                    return best_act

                # Fallback if no learned mapping aligned: simple axis priority
                if abs(dr_needed) >= abs(dc_needed):
                    act = GameAction.ACTION1 if dr_needed < 0 else GameAction.ACTION2
                else:
                    act = GameAction.ACTION3 if dc_needed > 0 else GameAction.ACTION4
                if act in dir_avail:
                    return act

        # ── A5 periodically (multi-piece games) ──────────────────────────
        if GameAction.ACTION5.value in avail and self._step % 10 == 0:
            return GameAction.ACTION5

        # ── Hybrid click occasionally ─────────────────────────────────────
        if 6 in avail and self._step % 7 == 0 and self._click_cells:
            col, row = self._click_cells[self._click_idx % len(self._click_cells)]
            self._click_idx += 1
            GameAction.ACTION6.set_data({"x": col, "y": row})
            return GameAction.ACTION6

        # ── Last resort ───────────────────────────────────────────────────
        if dir_avail:
            return dir_avail[self._step % len(dir_avail)]
        return GameAction.ACTION1
