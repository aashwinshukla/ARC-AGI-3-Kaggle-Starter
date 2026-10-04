---
inclusion: manual
---

# Next Session Plan (50 credits)

## Step 1 — Baseline all 25 games (~5 credits)
Run the current agent against every game and record which ones already score > 0.

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python scripts\play_local.py --max-steps 200
```

Note which games score 0 vs > 0.

---

## Step 2 — Attack by priority order

### Tier 1: Click-only games (7 games) — easiest wins
These likely just need "find the object, click it."
- `lf52`, `lp85`, `r11l`, `s5i5`, `su15`, `tn36`, `vc33`

For each: open in visualizer, click around, observe what triggers a level.
Then write a general click strategy (find non-bg non-wall cluster, click its center).

### Tier 2: Keyboard-only games (4 games) — same mechanic as AR25
Current agent might already score here. If not, same fix as AR25.
- `g50t`, `ls20`, `tr87`, `wa30`

### Tier 3: AR25 levels 2-5 — already cracked the pattern
Same piece-to-target logic, just needs better multi-piece handling and stall recovery.
- Improve `_find_piece_and_target` to handle multiple pieces per level.
- Increase stall tolerance and try all four directions before cycling.

### Tier 4: keyboard+click games — 14 games, leave for last
- `ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`, `ka59`, `m0r0`,
  `re86`, `sb26`, `sc25`, `sk48`, `sp80`, `tu93`

---

## Key things we know going in

- AR25 control mapping (verified): A1=up, A2=down, A3=right, A4=left
- ACTION5 cycles selected piece, ACTION6(x,y) clicks to select
- Cyan (colour 11) = target zone in AR25. Other games may use different colours.
- Background is the dominant colour (most frequent cell value).
- Step budget is shown as the rightmost column shrinking — exceeding it = GAME_OVER.
- 8 levels in AR25. Level budgets: 64 (lvl 1-2), 128 (lvl 3-5), 320 (lvl 6-8).

---

## Current agent score
- AR25: ~0.154 (1 level)
- All others: unknown (to check in Step 1)
