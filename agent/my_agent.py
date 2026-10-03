"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

What I learned by experimenting with AR25 (a block-sliding puzzle):
  - ACTION1 = move selected piece UP    (-row)
  - ACTION2 = move selected piece DOWN  (+row)   ← A3 moves piece right (confusing names)
  - ACTION3 = move selected piece RIGHT (+col)   ← verified empirically
  - ACTION4 = move selected piece LEFT  (-col)
  - ACTION5 = cycle to the next selectable piece
  - ACTION6(x,y) = click on grid to select a specific piece
  - Cyan cells (colour 11) on the right = TARGET ZONE
  - Win = move the piece so it overlaps with the cyan target

Key insight for AR25:
  The yellow piece (colour 4) starts at rows 15-23, cols 36-44.
  The cyan target is at rows 45-53, cols 51-59 (exactly the same 9x9 shape).
  Moving the piece RIGHT (A3) 5 times and DOWN (A2) 10 times = level complete.

General strategy (works for any game in this puzzle family):
  1. Find the small movable piece (smallest non-bg, non-wall colour cluster).
  2. Find the target zone (cyan=11, or smallest cluster on the opposite side).
  3. Move the piece toward the target using the correct directional action.
  4. If the piece stops moving (wall), cycle pieces (A5) or try other directions.
  5. Reset on GAME_OVER and retry.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class MyAgent(Agent):
    """
    Heuristic piece-slider agent.

    Identifies the movable piece, identifies the target, and navigates
    the piece to the target step by step. Handles stalls and retries.
    """

    MAX_ACTIONS = 200

    # Direction mapping (verified empirically):
    #   A1 = up    (row decreases)
    #   A2 = down  (row increases)
    #   A3 = right (col increases)   ← note: opposite of label
    #   A4 = left  (col decreases)   ← note: opposite of label

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_grid   = None
        self._stall_count = 0
        self._retry_count = 0
        self._step_count  = 0
        random.seed(0 + hash(self.game_id) % 9999)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ── grid helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _grid(frame: FrameData) -> np.ndarray:
        arr = np.array(frame.frame[0], dtype=int)
        arr[arr == -1] = 0
        return arr

    @staticmethod
    def _centroid(grid: np.ndarray, color: int):
        """Mean (row, col) of all cells with this color, or None."""
        mask = grid == color
        if not mask.any():
            return None
        rs, cs = np.where(mask)
        return float(rs.mean()), float(cs.mean())

    @staticmethod
    def _dominant_color(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _find_piece_and_target(self, grid: np.ndarray):
        """
        Returns (piece_color, target_color).

        Target = cyan (11) if present, else smallest cluster in top-right.
        Piece = yellow (4) if present, else smallest non-bg, non-wall, non-target.
        """
        bg   = self._dominant_color(grid)
        skip = {0, bg, 10}  # transparent, background, blue wall

        color_counts: dict[int, int] = {}
        for v, c in zip(*np.unique(grid, return_counts=True)):
            v, c = int(v), int(c)
            if v not in skip:
                color_counts[v] = c

        if not color_counts:
            return None, None

        # Target: prefer cyan (11)
        if 11 in color_counts:
            target_color = 11
        else:
            # Smallest cluster = target
            target_color = min(color_counts, key=lambda v: color_counts[v])

        # Piece: prefer yellow (4), then grey (5), then smallest non-target
        piece_color = None
        for preferred in (4, 5, 3, 2, 1):
            if preferred in color_counts and preferred != target_color:
                piece_color = preferred
                break
        if piece_color is None:
            candidates = [v for v in color_counts if v != target_color]
            if candidates:
                piece_color = min(candidates, key=lambda v: color_counts[v])

        return piece_color, target_color

    # ── action selection ──────────────────────────────────────────────────────

    def _move_toward(self, piece_rc, target_rc, avail: list[int]) -> GameAction:
        """
        Return the action that moves the piece closer to the target.
        Verified mapping: A1=up, A2=down, A3=right, A4=left.
        """
        dr = target_rc[0] - piece_rc[0]   # positive → need to go down (A2)
        dc = target_rc[1] - piece_rc[1]   # positive → need to go right (A3)

        # Build preference order: larger axis first
        if abs(dr) >= abs(dc):
            vert  = GameAction.ACTION1 if dr < 0 else GameAction.ACTION2
            horiz = GameAction.ACTION4 if dc < 0 else GameAction.ACTION3
            order = [vert, horiz]
        else:
            horiz = GameAction.ACTION4 if dc < 0 else GameAction.ACTION3
            vert  = GameAction.ACTION1 if dr < 0 else GameAction.ACTION2
            order = [horiz, vert]

        for act in order:
            if not avail or act.value in avail:
                return act

        # Fallback: first available move
        for aid in (1, 2, 3, 4):
            if aid in avail:
                return GameAction(aid)
        return GameAction.ACTION2

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        # ── Reset / restart ───────────────────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_grid   = None
            self._stall_count = 0
            self._step_count  = 0
            self._retry_count += 1
            return GameAction.RESET

        self._step_count += 1
        grid  = self._grid(latest_frame)
        avail = list(latest_frame.available_actions or [])

        # ── Stall detection ───────────────────────────────────────────────
        if self._last_grid is not None:
            self._stall_count = 0 if (grid != self._last_grid).any() else self._stall_count + 1
        self._last_grid = grid.copy()

        # ── Stall recovery ────────────────────────────────────────────────
        if self._stall_count >= 4:
            self._stall_count = 0
            # Alternate between cycling piece and random escape move
            if self._retry_count % 2 == 0 and GameAction.ACTION5.value in avail:
                return GameAction.ACTION5
            moves = [a for a in (1, 2, 3, 4) if a in avail]
            if moves:
                return GameAction(random.choice(moves))

        # ── Find piece and target ─────────────────────────────────────────
        piece_color, target_color = self._find_piece_and_target(grid)

        if piece_color is None or target_color is None:
            moves = [a for a in (1, 2, 3, 4) if a in avail]
            return GameAction(random.choice(moves)) if moves else GameAction.ACTION2

        piece_rc  = self._centroid(grid, piece_color)
        target_rc = self._centroid(grid, target_color)

        if piece_rc is None or target_rc is None:
            moves = [a for a in (1, 2, 3, 4) if a in avail]
            return GameAction(random.choice(moves)) if moves else GameAction.ACTION2

        # ── Move piece toward target ──────────────────────────────────────
        return self._move_toward(piece_rc, target_rc, avail)
