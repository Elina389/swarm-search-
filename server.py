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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from swarm import SwarmCoverageEnv
from movement_policy import BFSCoveragePolicy
from unknown_terrain import UnknownTerrainPolicy, FREE, OCCUPIED
from damage import (DamageModel, priority_density, top_priority_zones,
                    SEVERITY_LABELS)

# Two operating MODES, each with its own set of selectable sites:
#
#   known_map      -- the swarm is handed an accurate map up front (real
#                     OpenStreetMap obstacles) and plans routes around known
#                     obstacles with the BFS coverage policy. This is the
#                     original behaviour, unchanged.
#
#   unknown_terrain -- NO map is given. Each drone discovers obstacles live
#                     from limited-range sensing and remembers them in a
#                     shared occupancy grid (see unknown_terrain.py). This is
#                     the search-and-rescue-realistic version, demoed over
#                     real places that have actually experienced disasters.
#
# All sites are real named places used as stand-in search areas; there is no
# live incident data feed. bbox = (west, south, east, north) lon/lat degrees.
KNOWN_LOCATIONS = {
    "san_jose": {
        "label": "Downtown San Jose, CA",
        "bbox": (-121.94, 37.32, -121.87, 37.37),
        "grid_size": 25,
    },
    "berkeley": {
        "label": "UC Berkeley campus area",
        "bbox": (-122.259, 37.870, -122.253, 37.875),
        "grid_size": 20,
    },
    "golden_gate_park": {
        "label": "Golden Gate Park, SF",
        "bbox": (-122.511, 37.765, -122.454, 37.775),
        "grid_size": 25,
    },
}

DISASTER_LOCATIONS = {
    "amatrice": {
        "label": "Amatrice, Italy (2016 earthquake)",
        "bbox": (13.280, 42.620, 13.310, 42.640),
        "grid_size": 20,
    },
    "lahaina": {
        "label": "Lahaina, Maui (2023 wildfire)",
        "bbox": (-156.695, 20.868, -156.665, 20.892),
        "grid_size": 22,
    },
    "christchurch": {
        "label": "Christchurch, NZ (2011 earthquake)",
        "bbox": (172.620, -43.540, 172.645, -43.525),
        "grid_size": 22,
    },
    "kahramanmaras": {
        "label": "Kahramanmaras, Turkey (2023 earthquake)",
        "bbox": (36.910, 37.565, 36.935, 37.585),
        "grid_size": 22,
    },
}

MODES = {
    "known_map": {
        "label": "Known map -- plan around known obstacles",
        "locations": KNOWN_LOCATIONS,
        "default_location": "san_jose",
    },
    "unknown_terrain": {
        "label": "Unknown terrain -- discover obstacles live (disaster zones)",
        "locations": DISASTER_LOCATIONS,
        "default_location": "amatrice",
    },
}

DEFAULT_MODE = "known_map"
SENSOR_RADIUS = 3  # drone sensing range (cells) in unknown_terrain mode
MIN_DRONES = 1
MAX_DRONES = 10
DEFAULT_N_DRONES = 4

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
    a live demo, not a single one-shot script.

    Also supports switching location or drone count live: rebuild_env()
    tears down the current env and builds a fresh one, which necessarily
    means a full episode reset (obstacles, coverage, and battery for the
    new area/swarm size don't carry over from the old one -- there's no
    sensible way to "resume" a search of one place with a swarm sized for
    a different one).
    """

    def __init__(self, mode=DEFAULT_MODE, location_key=None, n_drones=DEFAULT_N_DRONES):
        self.mode = None
        self.location_key = None
        self.n_drones = None
        self.env = None
        self.policy = None
        self.headings = {}
        self._prev_occ = None  # snapshot of occupancy grid for tick-diffing
        # When True (unknown_terrain only), the swarm biases its routing
        # toward discovered high-severity damage. Toggle off to see the
        # "before" behaviour -- pure exploration ignoring damage -- so you
        # can compare how the drones fly with vs without prioritization.
        self.priority_routing = True
        # guards the sim loop from reading self.env mid-rebuild
        self.lock = asyncio.Lock()
        self.rebuild_env(mode, location_key, n_drones)

    def rebuild_env(self, mode, location_key, n_drones):
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode!r}")
        locations = MODES[mode]["locations"]
        if location_key is None:
            location_key = MODES[mode]["default_location"]
        if location_key not in locations:
            raise ValueError(f"unknown location {location_key!r} for mode {mode!r}")
        n_drones = max(MIN_DRONES, min(MAX_DRONES, int(n_drones)))

        loc = locations[location_key]
        self.mode = mode
        self.location_key = location_key
        self.n_drones = n_drones
        self.env = SwarmCoverageEnv(
            grid_size=loc["grid_size"], n_drones=n_drones, max_steps=MAX_STEPS,
            bbox=loc["bbox"],
        )
        self._reset_episode()

    def _reset_episode(self):
        self.env.reset()
        if self.mode == "unknown_terrain":
            self.policy = UnknownTerrainPolicy(sensor_radius=SENSOR_RADIUS)
            self.policy.reset(self.env)
            self._prev_occ = self.policy.occ.state.copy()
            # ground-truth damage for this disaster site (synthetic stand-in
            # for a real damage classifier -- see damage.py). Drones only
            # LEARN a cell's severity once they sense it; self.sensed_damage
            # is the discovered-so-far subset, exactly like the occupancy
            # grid is discovered-so-far obstacles.
            self.damage_model = DamageModel(
                self.env.grid_size, self.env.obstacles,
                seed=self.env.rng.integers(1_000_000), n_epicenters=2,
            )
            self.sensed_damage = {}  # (r,c) -> severity, only where sensed
        else:
            self.policy = BFSCoveragePolicy()
            self._prev_occ = None
            self.damage_model = None
            self.sensed_damage = {}
        self.headings = {agent: 0 for agent in self.env.agents}

    def _priority_grid(self):
        """Gaussian priority-density grid built from damage discovered so
        far (0 everywhere not-yet-sensed-as-damaged). Used both to bias the
        swarm's routing and to rank zones for the UI."""
        sev = np.zeros((self.env.grid_size, self.env.grid_size), dtype=float)
        for (r, c), s in self.sensed_damage.items():
            sev[r, c] = s
        if not self.sensed_damage:
            return None
        return priority_density(sev, sigma=2.0)

    def _damage_counts(self):
        """Running tally of discovered damage by severity label, for the UI."""
        counts = {label: 0 for label in SEVERITY_LABELS.values() if label != "intact"}
        for s in self.sensed_damage.values():
            counts[SEVERITY_LABELS[s]] += 1
        return counts

    def _locations(self):
        return MODES[self.mode]["locations"]

    def location_center(self):
        west, south, east, north = self._locations()[self.location_key]["bbox"]
        return {"lat": (south + north) / 2, "lon": (west + east) / 2}

    def obstacles_payload(self):
        """[west, south, east, north] bounds for every obstacle cell. Only
        used in known_map mode -- in unknown_terrain mode obstacles are
        hidden and revealed incrementally through sensing instead."""
        return [list(self.env._cell_bounds(r, c)) for (r, c) in self.env.obstacles]

    def drone_payload(self):
        payload = {}
        reasons = getattr(self.policy, "last_reason", {})
        for agent, pos in self.env.positions.items():
            lat, lon = self.env.cell_to_latlon(*pos)
            payload[agent] = {
                "lat": lat,
                "lon": lon,
                "heading": self.headings.get(agent, 0),
                "battery": round(self.env.battery[agent], 1),
                "status": reasons.get(agent, ""),
            }
        return payload

    def step(self):
        """Advance one tick. Returns (tick_message, just_reset)."""
        prev_positions = self.env.positions
        prev_covered = set(self.env.covered)

        # In unknown_terrain mode the policy senses (updating its occupancy
        # grid) as part of choosing actions. Pass the current damage-priority
        # grid so the swarm biases toward discovered high-severity clusters.
        if self.mode == "unknown_terrain":
            pri = self._priority_grid() if self.priority_routing else None
            actions = self.policy.actions(self.env, priority=pri)
        else:
            actions = self.policy.actions(self.env)
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        for agent, pos in self.env.positions.items():
            prev = prev_positions[agent]
            delta = (pos[0] - prev[0], pos[1] - prev[1])
            if delta in HEADING_BEARINGS:
                self.headings[agent] = HEADING_BEARINGS[delta]

        coverage = list(infos.values())[0]["coverage"] if infos else 0.0

        message = {
            "type": "tick",
            "step": self.env.steps,
            "max_steps": self.env.max_steps,
            "coverage": coverage,
            "drones": self.drone_payload(),
        }

        if self.mode == "unknown_terrain":
            # Report cells whose occupancy state changed since last tick, so
            # the frontend can lift the "fog" incrementally and draw newly
            # discovered obstacles. explored = fraction of the map no longer
            # unknown (the real progress metric in this mode).
            occ = self.policy.occ.state
            newly_free, newly_occ = [], []
            damage_events = []
            changed = (occ != self._prev_occ)
            for (r, c) in zip(*changed.nonzero()):
                r, c = int(r), int(c)
                bounds = list(self.env._cell_bounds(r, c))
                if occ[r, c] == FREE:
                    newly_free.append(bounds)
                elif occ[r, c] == OCCUPIED:
                    newly_occ.append(bounds)
                    # first time we sense this obstacle -> run the "damage
                    # classifier" (severity_of seam) and, if damaged, emit an
                    # event carrying the severity + a plain-language reason
                    # for the map popup.
                    sev = self.damage_model.severity_of((r, c))
                    if sev > 0 and (r, c) not in self.sensed_damage:
                        self.sensed_damage[(r, c)] = sev
                        lat, lon = self.env.cell_to_latlon(r, c)
                        damage_events.append({
                            "bounds": bounds,
                            "lat": lat, "lon": lon,
                            "severity": sev,
                            "label": SEVERITY_LABELS[sev],
                            "description": self.damage_model.describe((r, c)),
                        })
            self._prev_occ = occ.copy()

            # rank discovered damage into distinct priority zones for the UI
            zones = []
            pri = self._priority_grid()
            if pri is not None:
                for (zr, zc, score) in top_priority_zones(pri, k=3):
                    lat, lon = self.env.cell_to_latlon(int(zr), int(zc))
                    zones.append({"lat": lat, "lon": lon, "score": round(float(score), 3)})

            message["new_sensed_free"] = newly_free
            message["new_sensed_obstacle"] = newly_occ
            message["explored"] = self.policy.occ.explored_fraction()
            message["damage_events"] = damage_events
            message["damage_counts"] = self._damage_counts()
            message["priority_zones"] = zones
        else:
            new_cells = set(self.env.covered) - prev_covered
            message["new_covered"] = [list(self.env._cell_bounds(r, c)) for (r, c) in new_cells]

        just_reset = False
        if not self.env.agents:  # episode truncated
            self._reset_episode()
            just_reset = True

        return message, just_reset


runner = SimulationRunner()


def init_payload():
    return {
        "type": "init",
        "mode": runner.mode,
        "modes": {k: v["label"] for k, v in MODES.items()},
        "location_key": runner.location_key,
        # locations available for the CURRENT mode (frontend swaps the site
        # dropdown when the mode changes)
        "locations": {k: v["label"] for k, v in runner._locations().items()},
        "n_drones": runner.n_drones,
        "min_drones": MIN_DRONES,
        "max_drones": MAX_DRONES,
        "priority_routing": runner.priority_routing,
        "grid_size": runner.env.grid_size,
        # obstacles are only revealed up-front in known_map mode; in
        # unknown_terrain they start hidden and are sensed live
        "obstacles": runner.obstacles_payload() if runner.mode == "known_map" else [],
        "center": runner.location_center(),
        "bbox": list(runner._locations()[runner.location_key]["bbox"]),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # New client gets the current location/obstacle layout immediately --
    # obstacles are fixed per bbox, so this only needs to happen once per
    # client (or again after a rebuild, see /api/configure below), not
    # every tick.
    async with runner.lock:
        await ws.send_json(init_payload())
    try:
        while True:
            # Keep the connection open; the actual simulation loop
            # (below) is what pushes ticks. We just need to detect
            # disconnects here.
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


class ConfigureRequest(BaseModel):
    mode: str
    location: str | None = None
    n_drones: int
    priority_routing: bool | None = None


@app.post("/api/configure")
async def configure(req: ConfigureRequest):
    """Switch the live simulation to a different mode, search site, and/or
    drone count -- the 'radar: pick a mode + zone and send N drones' control.
    Rebuilds the environment (a fresh episode; see rebuild_env's docstring
    for why state can't carry over) and re-broadcasts a fresh init payload
    to every connected browser so their overlay/sidebar update to match the
    new configuration immediately, instead of waiting for the next tick.
    """
    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode!r}")
    locations = MODES[req.mode]["locations"]
    if req.location is not None and req.location not in locations:
        raise HTTPException(
            status_code=400,
            detail=f"unknown location {req.location!r} for mode {req.mode!r}",
        )
    async with runner.lock:
        if req.priority_routing is not None:
            runner.priority_routing = bool(req.priority_routing)
        runner.rebuild_env(req.mode, req.location, req.n_drones)
        await manager.broadcast(init_payload())
        await manager.broadcast({"type": "reset"})
    return {"ok": True, "mode": runner.mode, "location": runner.location_key,
            "n_drones": runner.n_drones, "priority_routing": runner.priority_routing}


async def simulation_loop():
    """Runs forever in the background: step the sim, broadcast the result,
    wait, repeat. Starts as soon as the server starts, independent of
    whether any browser is currently connected."""
    while True:
        async with runner.lock:
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
