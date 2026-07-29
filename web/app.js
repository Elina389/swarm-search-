/*
 * Live map frontend for the swarm search simulation.
 * Connects to server.py's WebSocket, draws obstacles/coverage/drones on a
 * Leaflet map (OpenStreetMap tiles, no API key needed), and keeps the
 * sidebar stats in sync with each tick broadcast from the backend.
 */

const map = L.map("map", { zoomControl: true });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Layers: obstacles (fixed, drawn once), coverage (grows over time, cleared
// on reset), drones (redrawn every tick).
const obstacleLayer = L.layerGroup().addTo(map);
const coverageLayer = L.layerGroup().addTo(map);
const droneLayer = L.layerGroup().addTo(map);

const droneMarkers = {};     // agent -> L.marker
const DRONE_COLOR = "#e74c3c";

// Small triangular "plane" icon as a divIcon, rotated via CSS transform to
// match the drone's current heading (a real <img> would work the same way
// if you wanted a nicer icon later -- this keeps things dependency-free).
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
  // b = [west, south, east, north]
  const [west, south, east, north] = b;
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

function handleInit(msg) {
  map.setView([msg.center.lat, msg.center.lon], 15);

  obstacleLayer.clearLayers();
  for (const bounds of msg.obstacles) {
    L.rectangle(boundsToLatLngBounds(bounds), {
      color: "#000",
      weight: 0,
      fillColor: "#1a1a1a",
      fillOpacity: 0.55,
    }).addTo(obstacleLayer);
  }
}

function handleTick(msg) {
  document.getElementById("stat-step").textContent = `${msg.step} / ${msg.max_steps}`;
  document.getElementById("stat-coverage").textContent = `${(msg.coverage * 100).toFixed(0)}%`;
  document.getElementById("coverage-bar-fill").style.width = `${msg.coverage * 100}%`;

  for (const bounds of msg.new_covered) {
    L.rectangle(boundsToLatLngBounds(bounds), {
      color: "#2ecc71",
      weight: 0,
      fillColor: "#2ecc71",
      fillOpacity: 0.35,
    }).addTo(coverageLayer);
  }

  for (const [name, d] of Object.entries(msg.drones)) {
    const latlng = [d.lat, d.lon];
    if (droneMarkers[name]) {
      droneMarkers[name].setLatLng(latlng);
      droneMarkers[name].setIcon(planeIcon(d.heading));
    } else {
      droneMarkers[name] = L.marker(latlng, { icon: planeIcon(d.heading) })
        .bindPopup(name)
        .addTo(droneLayer);
    }
  }

  renderDroneSidebar(msg.drones);
}

function handleReset() {
  coverageLayer.clearLayers();
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

connect();
