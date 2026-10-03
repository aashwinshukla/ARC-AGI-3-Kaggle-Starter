# ARC Prize 2026 — Project Context

## What this project is
Kaggle code competition: ARC Prize 2026 — ARC-AGI-3.
The submission is a Python agent that plays interactive puzzle environments.
The only file the user edits is `agent/my_agent.py`.
Local test command: `.venv\Scripts\python scripts\play_local.py --max-steps 200`
(equivalent to `make play-local` on Linux/Mac).

## The game environment
- 64×64 grids, cell values 0–15
- No instructions are given; the agent must infer the rules purely by interacting
- Games are unseen — the agent must be general, not tuned to any specific game
- Each step: agent receives the current frame and all previous frames, returns one action
- Actions are integers (e.g. ACTION1–ACTION7); available actions vary per game

## Hard constraints — never violate these
- No internet access inside the agent (blocked during Kaggle evaluation)
- No LLM or external API calls inside the agent
- Never hard-code logic for a specific named game (ls20, vc33, etc.) — keep behaviour general
- Never read, print, log, or commit `.kaggle/access_token`

## User preferences
- The user is a beginner: always provide complete, runnable files (no partial snippets)
- Keep code simple and well-commented — explain the "why", not just the "what"
- After every code change, summarise what changed in 3 lines or fewer
- Prefer clarity over cleverness; avoid abstractions the user hasn't seen before

## Project layout (reference)
```
agent/my_agent.py          ← the only file the user edits
scripts/play_local.py      ← runs the agent locally
scripts/build_notebook.py  ← packages agent into a Kaggle notebook
scripts/slim_framework.py  ← trims vendor deps
vendor/ARC-AGI-3-Agents/   ← cloned framework (do not edit)
.venv/                     ← Python venv (do not commit)
.kaggle/access_token       ← Kaggle API token (never read or commit)
notebooks/kernel-metadata.json  ← edit once: set your Kaggle username
```
