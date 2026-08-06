/*
 * Live map frontend for the swarm search simulation.
 * Connects to server.py's WebSocket, draws obstacles/coverage/drones on a
 * Leaflet map (OpenStreetMap tiles, no API key needed), and keeps the
 * sidebar stats in sync with each tick broadcast from the backend.
 *
 * Two modes, chosen from the sidebar:
 *   known_map       -- obstacles are known up front and drawn immediately;
 *                      searched ground fills in green.
 *   unknown_terrain -- the map starts as dark "fog"; cells are revealed as
 *                      the swarm senses them (free = light, obstacle = red),
 *                      modelling drones discovering a disaster zone live.
 */

const map = L.map("map", { zoomControl: true });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Layers (drawn bottom to top):
//   fogLayer      -- unknown_terrain only: a dark blanket the size of the
//                    search area, representing not-yet-sensed space.
//   revealedLayer -- cells sensed as free (unknown mode) -- "lifts the fog".
//   obstacleLayer -- known obstacles (known mode) OR discovered obstacles
//                    (unknown mode).
//   coverageLayer -- searched/visited ground (known mode green trail).
//   droneLayer    -- the drones themselves, on top.
const fogLayer = L.layerGroup().addTo(map);
const revealedLayer = L.layerGroup().addTo(map);
const obstacleLayer = L.layerGroup().addTo(map);
const coverageLayer = L.layerGroup().addTo(map);
const damageLayer = L.layerGroup().addTo(map);   // sensed damaged structures
const priorityLayer = L.layerGroup().addTo(map); // ranked priority-zone rings
const trailLayer = L.layerGroup().addTo(map);    // per-drone flight trajectories
const droneLayer = L.layerGroup().addTo(map);

const droneMarkers = {};     // agent -> L.marker
const droneTrails = {};      // agent -> [ [lat,lon], ... ] recent path
const TRAIL_MAX = 30;        // how many recent positions to keep per drone
const DRONE_COLOR = "#e74c3c";
let currentMode = "known_map";

// Severity 1-3 -> colour (matches the sidebar legend swatches).
const SEVERITY_COLOR = { 1: "#f1c40f", 2: "#e67e22", 3: "#c0392b" };

function planeIcon(headingDeg) {
  return L.divIcon({
    className: "",
    html: `<div style="
        width:0; height:0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        border-bottom: 16px solid ${DRONE_COLOR};
        filter: drop-shadow(0 0 1px #000);
        transform: rotate(${headingDeg}deg);
        transform-origin: 50% 60%;
      "></div>`,
    iconSize: [14, 16],
    iconAnchor: [7, 8],
  });
}

function boundsToLatLngBounds(b) {
  const [west, south, east, north] = b; // [west, south, east, north]
  return [[south, west], [north, east]];
}

function setStatus(connected, text) {
  document.getElementById("status-dot").classList.toggle("connected", connected);
  document.getElementById("status-text").textContent = text;
}

function renderDroneSidebar(drones) {
  const list = document.getElementById("drone-list");
  list.innerHTML = "";
  for (const [name, d] of Object.entries(drones)) {
    const pct = Math.max(0, Math.min(100, d.battery));
    const color = pct > 50 ? "#2ecc71" : pct > 20 ? "#f1c40f" : "#e74c3c";
    const card = document.createElement("div");
    card.className = "drone-card";
    card.innerHTML = `
      <div class="name">${name}</div>
      <div class="battery-track"><div class="battery-fill" style="width:${pct}%; background:${color};"></div></div>
      <div class="battery-label">battery ${pct.toFixed(0)}%</div>
    `;
    list.appendChild(card);
  }
}

function fillSelect(select, options, selectedKey) {
  // Rebuild an option list only when its contents actually change (e.g. the
  // site list changes when the mode changes). Avoids clobbering the user's
  // in-progress selection on every tick.
  const wanted = JSON.stringify(Object.keys(options));
  if (select.dataset.keys !== wanted) {
    select.innerHTML = "";
    for (const [key, label] of Object.entries(options)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = label;
      select.appendChild(opt);
    }
    select.dataset.keys = wanted;
  }
  select.value = selectedKey;
}

function populateControls(msg) {
  fillSelect(document.getElementById("mode-select"), msg.modes, msg.mode);
  fillSelect(document.getElementById("location-select"), msg.locations, msg.location_key);

  const slider = document.getElementById("drone-slider");
  slider.min = msg.min_drones;
  slider.max = msg.max_drones;
  slider.value = msg.n_drones;
  document.getElementById("drone-count-label").textContent = msg.n_drones;

  const isUnknown = msg.mode === "unknown_terrain";
  document.getElementById("progress-label").textContent =
    isUnknown ? "Explored" : "Coverage";
  document.getElementById("subtitle").textContent =
    (isUnknown ? "Discovering unknown terrain: " : "Live coverage over ") +
    msg.locations[msg.location_key];

  // priority toggle only applies to unknown_terrain mode
  document.getElementById("priority-row").classList.toggle("hidden", !isUnknown);
  if (typeof msg.priority_routing === "boolean") {
    document.getElementById("priority-toggle").checked = msg.priority_routing;
  }
}

function clearAllLayers() {
  fogLayer.clearLayers();
  revealedLayer.clearLayers();
  obstacleLayer.clearLayers();
  coverageLayer.clearLayers();
  damageLayer.clearLayers();
  priorityLayer.clearLayers();
  trailLayer.clearLayers();
  droneLayer.clearLayers();
  for (const key of Object.keys(droneMarkers)) delete droneMarkers[key];
  for (const key of Object.keys(droneTrails)) delete droneTrails[key];
}

// Plain-language explanation of the active policy + its "functions" + a
// legend for the trajectory/colours, shown in the right-side panel.
function updateExplanation(mode) {
  const el = document.getElementById("explain-panel");
  if (mode === "unknown_terrain") {
    el.innerHTML = `
      <h3>How the swarm is flying</h3>
      No map is given. Each drone senses a small radius around itself and only
      ever moves into cells it has already confirmed <b>free</b>, so it can't
      crash into something it hasn't seen.
      <div style="margin-top:6px">Target choice minimizes a cost function:</div>
      <div class="fn">cost(cell) = distance &minus; w &times; priority(cell)</div>
      <div style="margin-top:4px" class="muted">With "Prioritize damage" on, the
      damage-priority field (a Gaussian blur of discovered severity) pulls drones
      toward severe clusters; off, it's pure nearest-frontier exploration.</div>
      <h3 style="margin-top:10px">Trajectories &amp; colours</h3>
      <div class="legend-row"><span class="legend-swatch" style="background:#5da9ff"></span>flight trail (recent path)</div>
      <div class="legend-row"><span class="legend-swatch" style="background:#cfeede"></span>sensed free (fog lifted)</div>
      <div class="legend-row"><span class="legend-swatch" style="background:#2b2f36"></span>intact structure</div>
      <div class="legend-row"><span class="legend-swatch" style="background:#f1c40f"></span>minor / <span class="legend-swatch" style="background:#e67e22"></span>major / <span class="legend-swatch" style="background:#c0392b"></span>destroyed</div>
      <div style="margin-top:6px" class="muted">Hover a drone to see what it's
      doing right now; click a damaged block for details.</div>`;
  } else {
    el.innerHTML = `
      <h3>How the swarm is flying</h3>
      The map (real OpenStreetMap buildings/water) is known up front. Each drone
      runs a breadth-first search over open cells and walks the <b>shortest path</b>
      to its nearest unsearched cell; drones claim different targets so they split
      up the area.
      <div style="margin-top:6px">Coverage grows as:</div>
      <div class="fn">coverage = visited free cells / total free cells</div>
      <h3 style="margin-top:10px">Trajectories &amp; colours</h3>
      <div class="legend-row"><span class="legend-swatch" style="background:#5da9ff"></span>flight trail (recent path)</div>
      <div class="legend-row"><span class="legend-swatch" style="background:#2ecc71"></span>searched ground</div>
      <div class="legend-row"><span class="legend-swatch" style="background:#1a1a1a"></span>known obstacle</div>
      <div style="margin-top:6px" class="muted">Hover a drone to see what it's
      doing right now.</div>`;
  }
}

// Re-draw all drone flight trails from their stored recent positions.
function redrawTrails() {
  trailLayer.clearLayers();
  for (const [name, pts] of Object.entries(droneTrails)) {
    if (pts.length >= 2) {
      L.polyline(pts, { color: "#5da9ff", weight: 2, opacity: 0.55 }).addTo(trailLayer);
    }
  }
}

function showDamagePanel(show) {
  document.getElementById("damage-panel").classList.toggle("hidden", !show);
}

function renderDamageEvents(events) {
  // Draw each newly-detected damaged structure coloured by severity, with a
  // click popup explaining what is wrong -- this is the "describe what's
  // wrong on the map" piece.
  for (const e of events) {
    const color = SEVERITY_COLOR[e.severity] || "#e74c3c";
    const popup =
      `<div class="damage-popup"><b>${e.label.toUpperCase()} damage</b>${e.description}</div>`;
    L.rectangle(boundsToLatLngBounds(e.bounds), {
      color: "#000", weight: 1, fillColor: color, fillOpacity: 0.85,
    }).bindPopup(popup).addTo(damageLayer);
  }
}

function renderDamageSidebar(counts, zones) {
  if (counts) {
    document.getElementById("cnt-minor").textContent = counts.minor ?? 0;
    document.getElementById("cnt-major").textContent = counts.major ?? 0;
    document.getElementById("cnt-destroyed").textContent = counts.destroyed ?? 0;
  }
  // priority-zone rings on the map + clickable list in the sidebar
  priorityLayer.clearLayers();
  const list = document.getElementById("zone-list");
  list.innerHTML = "";
  (zones || []).forEach((z, i) => {
    L.circle([z.lat, z.lon], {
      radius: 60, color: "#ff4d4d", weight: 2, fill: false, dashArray: "4 4",
    }).addTo(priorityLayer);
    const row = document.createElement("div");
    row.className = "zone-row";
    row.textContent = `#${i + 1} priority zone -- score ${z.score.toFixed(2)}`;
    row.addEventListener("click", () => map.setView([z.lat, z.lon], 17));
    list.appendChild(row);
  });
  if (!zones || zones.length === 0) {
    list.innerHTML = '<div style="font-size:11px;color:#777">none detected yet</div>';
  }
}

function handleInit(msg) {
  currentMode = msg.mode;
  map.setView([msg.center.lat, msg.center.lon], 15);
  populateControls(msg);
  clearAllLayers();
  showDamagePanel(msg.mode === "unknown_terrain");
  renderDamageSidebar({ minor: 0, major: 0, destroyed: 0 }, []);
  updateExplanation(msg.mode);

  if (msg.mode === "unknown_terrain" && msg.bbox) {
    // Lay a dark "fog of war" blanket over the whole search area. Sensed
    // free cells (revealedLayer) and discovered obstacles are drawn on top,
    // so the fog visibly lifts as the swarm explores.
    L.rectangle(boundsToLatLngBounds(msg.bbox), {
      color: "#000", weight: 1, fillColor: "#0a0c10", fillOpacity: 0.82,
    }).addTo(fogLayer);
  }

  // known_map: draw the known obstacles right away
  for (const bounds of msg.obstacles) {
    L.rectangle(boundsToLatLngBounds(bounds), {
      color: "#000", weight: 0, fillColor: "#1a1a1a", fillOpacity: 0.55,
    }).addTo(obstacleLayer);
  }
}

async function deploySwarm() {
  const mode = document.getElementById("mode-select").value;
  const location = document.getElementById("location-select").value;
  const n_drones = parseInt(document.getElementById("drone-slider").value, 10);
  const btn = document.getElementById("deploy-btn");

  const priority_routing = document.getElementById("priority-toggle").checked;

  btn.disabled = true;
  btn.textContent = "Deploying...";
  try {
    const resp = await fetch("/api/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, location, n_drones, priority_routing }),
    });
    if (!resp.ok) throw new Error(`server returned ${resp.status}`);
    // Server broadcasts a fresh "init" + "reset" once rebuilt; handleInit /
    // handleReset pick it up automatically.
  } catch (err) {
    console.error("Failed to configure simulation:", err);
    alert("Couldn't reconfigure -- check that the server is running.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Deploy swarm";
  }
}

function handleTick(msg) {
  document.getElementById("stat-step").textContent = `${msg.step} / ${msg.max_steps}`;

  // Progress metric differs by mode: coverage (visited) vs explored (sensed).
  const progress = currentMode === "unknown_terrain"
    ? (msg.explored ?? 0)
    : msg.coverage;
  document.getElementById("stat-coverage").textContent = `${(progress * 100).toFixed(0)}%`;
  document.getElementById("coverage-bar-fill").style.width = `${progress * 100}%`;

  if (currentMode === "unknown_terrain") {
    // reveal newly-sensed free cells (lift the fog): a pale fill opaque
    // enough to visually cut through the dark fog blanket beneath it
    for (const bounds of (msg.new_sensed_free || [])) {
      L.rectangle(boundsToLatLngBounds(bounds), {
        color: "#a8e6c0", weight: 0, fillColor: "#cfeede", fillOpacity: 0.55,
      }).addTo(revealedLayer);
    }
    // draw newly-discovered obstacles as neutral dark blocks (intact
    // structures). Damaged ones are drawn separately, coloured by severity,
    // on top via the damageLayer so they stand out.
    for (const bounds of (msg.new_sensed_obstacle || [])) {
      L.rectangle(boundsToLatLngBounds(bounds), {
        color: "#000", weight: 0, fillColor: "#2b2f36", fillOpacity: 0.75,
      }).addTo(obstacleLayer);
    }
    renderDamageEvents(msg.damage_events || []);
    renderDamageSidebar(msg.damage_counts, msg.priority_zones);
  } else {
    for (const bounds of (msg.new_covered || [])) {
      L.rectangle(boundsToLatLngBounds(bounds), {
        color: "#2ecc71", weight: 0, fillColor: "#2ecc71", fillOpacity: 0.35,
      }).addTo(coverageLayer);
    }
  }

  for (const [name, d] of Object.entries(msg.drones)) {
    const latlng = [d.lat, d.lon];
    // accumulate the flight trail
    (droneTrails[name] = droneTrails[name] || []).push(latlng);
    if (droneTrails[name].length > TRAIL_MAX) droneTrails[name].shift();

    // hover tooltip explaining what this drone is doing right now
    const tip =
      `<div class="drone-tooltip"><b>${name}</b>` +
      `battery ${d.battery.toFixed(0)}%<br>${d.status || ""}</div>`;

    if (droneMarkers[name]) {
      droneMarkers[name].setLatLng(latlng);
      droneMarkers[name].setIcon(planeIcon(d.heading));
      droneMarkers[name].setTooltipContent(tip);
    } else {
      droneMarkers[name] = L.marker(latlng, { icon: planeIcon(d.heading) })
        .bindTooltip(tip, { direction: "top", offset: [0, -8], sticky: true })
        .addTo(droneLayer);
    }
  }

  redrawTrails();
  renderDroneSidebar(msg.drones);
}

function handleReset() {
  // A new episode started (same mode/site) -- clear the progressive layers
  // so the fog/coverage rebuilds from scratch, but leave known obstacles.
  revealedLayer.clearLayers();
  coverageLayer.clearLayers();
  trailLayer.clearLayers();
  for (const key of Object.keys(droneTrails)) delete droneTrails[key];
  if (currentMode === "unknown_terrain") {
    obstacleLayer.clearLayers();  // discovered obstacles reset too in unknown mode
    damageLayer.clearLayers();    // discovered damage resets with the new episode
    priorityLayer.clearLayers();
    renderDamageSidebar({ minor: 0, major: 0, destroyed: 0 }, []);
  }
}

function connect() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws`);

  ws.onopen = () => setStatus(true, "connected");
  ws.onclose = () => {
    setStatus(false, "disconnected -- retrying in 2s");
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "init") handleInit(msg);
    else if (msg.type === "tick") handleTick(msg);
    else if (msg.type === "reset") handleReset();
  };
}

document.getElementById("drone-slider").addEventListener("input", (e) => {
  document.getElementById("drone-count-label").textContent = e.target.value;
});
// When the mode changes, immediately swap the site dropdown to that mode's
// sites so the user can't pick an invalid mode/site combination. We fetch a
// fresh site list by asking the server to switch mode with its default site.
document.getElementById("mode-select").addEventListener("change", () => {
  // Deploy is what actually applies the change; but to keep the site list
  // valid we trigger a deploy immediately on mode change using that mode's
  // default site (the server picks the default when location is omitted).
  deploySwarmModeChange();
});
document.getElementById("deploy-btn").addEventListener("click", deploySwarm);

async function deploySwarmModeChange() {
  const mode = document.getElementById("mode-select").value;
  const n_drones = parseInt(document.getElementById("drone-slider").value, 10);
  const priority_routing = document.getElementById("priority-toggle").checked;
  try {
    await fetch("/api/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, location: null, n_drones, priority_routing }),
    });
  } catch (err) {
    console.error("mode switch failed:", err);
  }
}

connect();
