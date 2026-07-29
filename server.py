"""
Live web server for the swarm search simulation.

Runs SwarmCoverageEnv continuously in the background (using the BFS
coverage policy -- see movement_policy.py) and streams drone positions,
battery, and coverage progress to any connected browser over a WebSocket.
The browser renders it on a real Leaflet map (OpenStreetMap tiles, no API
key needed) instead of the matplotlib window used by visualize_swarm.py.

This file doesn't replace visualize_swarm.py -- it's a second, web-based
way to view the exact same simulation, useful for a nicer-looking live demo
in a browser tab instead of a desktop plot window.

Run with:
    PYTHONPATH=./swarm-env ./swarm-env/bin/python3.14 -m uvicorn server:app --reload
Then open http://127.0.0.1:8000 in a browser.

SECURITY NOTE: this starts an unauthenticated local web server. It's fine
for local development/demos on your own machine, but don't expose this
port to the public internet as-is -- there's no login, and anyone who can
reach it can watch (and, if you extend it, potentially control) the sim.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm-env"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from swarm import SwarmCoverageEnv
from movement_policy import BFSCoveragePolicy

# Same downtown San Jose slice used in visualize_swarm.py's demo_san_jose().
SAN_JOSE_BBOX = (-121.94, 37.32, -121.87, 37.37)  # west, south, east, north

GRID_SIZE = 25
N_DRONES = 4
MAX_STEPS = 150
TICK_SECONDS = 0.25  # simulated time between broadcasts -- lower = faster playback

# Grid movement delta -> real-world compass bearing in degrees (clockwise
# from north), matching how a map marker's rotation is normally expressed.
# NOTE: this is a different convention from HEADING_ANGLES in
# visualize_swarm.py, which is tuned for matplotlib's on-screen axes -- map
# bearings and screen-rotation angles aren't the same thing.
HEADING_BEARINGS = {
    (-1, 0): 0,     # moved up/north
    (1, 0): 180,    # moved down/south
    (0, -1): 270,   # moved left/west
    (0, 1): 90,     # moved right/east
}

app = FastAPI()


class ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts JSON to all of
    them, quietly dropping any that have disconnected."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


class SimulationRunner:
    """Owns one long-running SwarmCoverageEnv episode loop. When an episode
    truncates, it resets and keeps going -- this is meant to run forever as
    a live demo, not a single one-shot script."""

    def __init__(self):
        self.env = SwarmCoverageEnv(
            grid_size=GRID_SIZE, n_drones=N_DRONES, max_steps=MAX_STEPS,
            bbox=SAN_JOSE_BBOX,
        )
        self.policy = None
        self.headings = {}
        self._reset_episode()

    def _reset_episode(self):
        self.env.reset()
        self.policy = BFSCoveragePolicy()
        self.headings = {agent: 0 for agent in self.env.agents}

    def obstacles_payload(self):
        """[west, south, east, north] bounds for every obstacle cell --
        sent once so the frontend can draw them as map rectangles."""
        return [list(self.env._cell_bounds(r, c)) for (r, c) in self.env.obstacles]

    def drone_payload(self):
        payload = {}
        for agent, pos in self.env.positions.items():
            lat, lon = self.env.cell_to_latlon(*pos)
            payload[agent] = {
                "lat": lat,
                "lon": lon,
                "heading": self.headings.get(agent, 0),
                "battery": round(self.env.battery[agent], 1),
            }
        return payload

    def step(self):
        """Advance one tick. Returns (tick_message, just_reset)."""
        prev_positions = self.env.positions
        actions = self.policy.actions(self.env)
        prev_covered = set(self.env.covered)
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        for agent, pos in self.env.positions.items():
            prev = prev_positions[agent]
            delta = (pos[0] - prev[0], pos[1] - prev[1])
            if delta in HEADING_BEARINGS:
                self.headings[agent] = HEADING_BEARINGS[delta]

        new_cells = set(self.env.covered) - prev_covered
        new_covered_bounds = [list(self.env._cell_bounds(r, c)) for (r, c) in new_cells]

        coverage = list(infos.values())[0]["coverage"] if infos else 0.0

        message = {
            "type": "tick",
            "step": self.env.steps,
            "max_steps": self.env.max_steps,
            "coverage": coverage,
            "drones": self.drone_payload(),
            "new_covered": new_covered_bounds,
        }

        just_reset = False
        if not self.env.agents:  # episode truncated
            self._reset_episode()
            just_reset = True

        return message, just_reset


runner = SimulationRunner()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # New client gets the current obstacle layout immediately -- obstacles
    # are fixed for this bbox, so this only needs to happen once per client,
    # not every tick.
    await ws.send_json({
        "type": "init",
        "grid_size": runner.env.grid_size,
        "obstacles": runner.obstacles_payload(),
        "center": {
            "lat": (SAN_JOSE_BBOX[1] + SAN_JOSE_BBOX[3]) / 2,
            "lon": (SAN_JOSE_BBOX[0] + SAN_JOSE_BBOX[2]) / 2,
        },
    })
    try:
        while True:
            # Keep the connection open; the actual simulation loop
            # (below) is what pushes ticks. We just need to detect
            # disconnects here.
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


async def simulation_loop():
    """Runs forever in the background: step the sim, broadcast the result,
    wait, repeat. Starts as soon as the server starts, independent of
    whether any browser is currently connected."""
    while True:
        message, just_reset = runner.step()
        if just_reset:
            await manager.broadcast({"type": "reset"})
        await manager.broadcast(message)
        await asyncio.sleep(TICK_SECONDS)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(simulation_loop())


_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(_WEB_DIR, "index.html"), "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")
