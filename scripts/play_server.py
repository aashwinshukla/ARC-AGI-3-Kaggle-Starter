"""
play_server.py — Local web server so you can play any ARC-AGI-3 game in your browser.

How to run:
    .venv/Scripts/python scripts/play_server.py

Then open: http://localhost:5001
"""

import json
import logging
import os
import sys

# ── Make the vendor framework importable ─────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR    = os.path.join(REPO_ROOT, "vendor", "ARC-AGI-3-Agents")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from flask import Flask, jsonify, request, send_from_directory
from arc_agi import LocalEnvironmentWrapper, EnvironmentInfo
from arcengine import GameAction

app = Flask(__name__, static_folder=os.path.join(REPO_ROOT, "visualizer"))

logger = logging.getLogger("play_server")
logging.basicConfig(level=logging.WARNING)   # keep the console quiet

# ── One active game session (one game at a time is fine for local play) ───────
session: dict = {
    "env":     None,   # LocalEnvironmentWrapper instance
    "game_id": None,
    "steps":   0,
}


def _find_env_dir(short_id: str) -> str | None:
    """Return the versioned directory for a game, e.g. environment_files/ls20/9607627b."""
    base = os.path.join(REPO_ROOT, "environment_files", short_id)
    if not os.path.isdir(base):
        return None
    versions = sorted(os.listdir(base))
    for ver in versions:
        candidate = os.path.join(base, ver)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "metadata.json")):
            return candidate
    return None


def _raw_to_dict(raw, steps: int = 0) -> dict:
    """
    Convert a FrameDataRaw into a JSON-friendly dict.

    raw.frame is a list with one numpy array of shape (64, 64).
    Cell values are 0-15 (colours) or -1 (transparent / background = 0).
    """
    # Flatten the single layer; replace -1 transparency with 0 (background colour)
    layer = raw.frame[0]  # numpy array (64, 64)
    grid = []
    for row in layer:
        grid.append([int(v) if int(v) != -1 else 0 for v in row])

    # available_actions is a list of ints (e.g. [1,2,3,4])
    available = list(raw.available_actions) if raw.available_actions else list(range(1, 8))

    return {
        "grid":               grid,                     # 64×64 list-of-lists
        "state":              str(raw.state),           # "GameState.NOT_FINISHED" etc.
        "levels_completed":   int(raw.levels_completed),
        "available_actions":  available,                # list of ints
        "steps":              steps,
    }


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/games")
def list_games():
    """Return all locally cached games."""
    env_base = os.path.join(REPO_ROOT, "environment_files")
    games = []
    if os.path.isdir(env_base):
        for name in sorted(os.listdir(env_base)):
            env_dir = _find_env_dir(name)
            if env_dir:
                meta_path = os.path.join(env_dir, "metadata.json")
                with open(meta_path) as f:
                    meta = json.load(f)
                games.append({
                    "id":    name,
                    "title": meta.get("title", name.upper()),
                    "tags":  meta.get("tags", []),
                })
    return jsonify(games)


@app.route("/api/start", methods=["POST"])
def start_game():
    """Load and reset a game.  Body: {"game_id": "ls20"}"""
    data = request.get_json(force=True) or {}
    short_id = data.get("game_id", "ls20")

    env_dir = _find_env_dir(short_id)
    if env_dir is None:
        return jsonify({"error": f"Game '{short_id}' not found locally."}), 404

    # Close any existing env cleanly
    if session["env"] is not None:
        try:
            session["env"].close()
        except Exception:
            pass

    # Load metadata and build EnvironmentInfo
    with open(os.path.join(env_dir, "metadata.json")) as f:
        meta = json.load(f)
    meta["local_dir"] = os.path.abspath(env_dir)

    info = EnvironmentInfo(**meta)
    env  = LocalEnvironmentWrapper(
        environment_info=info,
        logger=logger,
        scorecard_id="visualizer",
    )
    raw = env.reset()

    session["env"]     = env
    session["game_id"] = short_id
    session["steps"]   = 0

    return jsonify({"status": "ok", "frame": _raw_to_dict(raw, steps=0)})


@app.route("/api/step", methods=["POST"])
def step():
    """
    Send one action to the running game.
    Body: {"action": 1}   (integer action id, 0=RESET, 1-7=ACTION1-7)
    For ACTION6 (click): {"action": 6, "x": 32, "y": 15}
    """
    if session["env"] is None:
        return jsonify({"error": "No game running. Call /api/start first."}), 400

    data      = request.get_json(force=True) or {}
    action_id = int(data.get("action", 0))
    x         = data.get("x", None)
    y         = data.get("y", None)

    # Find the GameAction by its integer value
    action = None
    for a in GameAction:
        if a.value == action_id:
            action = a
            break
    if action is None:
        return jsonify({"error": f"Unknown action id {action_id}"}), 400

    # ACTION6 is a click — it needs x,y coordinates
    if action.is_complex():
        cx = int(x) if x is not None else 32
        cy = int(y) if y is not None else 32
        action.set_data({"x": cx, "y": cy})

    raw = session["env"].step(action)
    session["steps"] += 1

    return jsonify(_raw_to_dict(raw, steps=session["steps"]))


@app.route("/api/reset", methods=["POST"])
def reset_game():
    """Reset the current game to the start without reloading."""
    if session["env"] is None:
        return jsonify({"error": "No game running."}), 400
    raw = session["env"].reset()
    session["steps"] = 0
    return jsonify({"status": "ok", "frame": _raw_to_dict(raw, steps=0)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Open your browser at:  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
