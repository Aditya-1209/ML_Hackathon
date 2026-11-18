const el = id => document.getElementById(id);
const query = new URLSearchParams(location.search);
const api = (query.get("api") || "").replace(/\/$/, "");
if (api && !/^https?:\/\//i.test(api)) throw new Error("The API address must be HTTP or HTTPS.");
const select = el("trip-select");
const inputs = Object.fromEntries([...document.querySelectorAll("[data-layer]")]
  .map(input => [input.dataset.layer, input]));
const percent = value => value == null ? "—" : (100 * value).toFixed(2) + "%";
const integer = value => new Intl.NumberFormat().format(value);
const map = L.map("map", {preferCanvas: true}).setView([48.7, 9.1], 5);
L.control.scale({imperial: false, position: "bottomleft"}).addTo(map);
map.createPane("gps-points").style.zIndex = "410";
map.createPane("matched-route").style.zIndex = "420";
const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
});
const overlays = {roads: L.layerGroup(), gps: L.layerGroup(), runs: L.layerGroup()};
let bounds, pending = false, hasRoute = false, tileFailed = false;

function syncLayers() {
  for (const [key, group] of Object.entries(overlays)) {
    if (inputs[key].checked || (key === "roads" && (tileFailed || !inputs.basemap.checked)))
      group.addTo(map);
    else map.removeLayer(group);
  }
}
function syncBasemap() {
  if (inputs.basemap.checked) tiles.addTo(map);
  else { map.removeLayer(tiles); el("basemap-notice").hidden = true; }
  syncLayers();
}
tiles.on("tileerror", () => {
  tileFailed = true;
  el("basemap-notice").hidden = !inputs.basemap.checked;
  syncLayers();
});
tiles.on("tileload", () => {
  if (!tileFailed) return;
  tileFailed = false;
  el("basemap-notice").hidden = true;
  syncLayers();
});
inputs.basemap.checked = query.get("basemap") !== "off";
inputs.basemap.addEventListener("change", syncBasemap);
for (const key of Object.keys(overlays)) inputs[key].addEventListener("change", syncLayers);
syncBasemap();

async function request(path, options) {
  const response = await fetch(api + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The API could not complete this request.");
  return data;
}

function message(text) {
  el("map-message").textContent = text;
  el("map-message").hidden = !text;
}

function fitRoute() {
  if (bounds?.isValid()) map.fitBounds(bounds, {padding: [35, 35], maxZoom: 17, animate: false});
}
el("fit").onclick = fitRoute;
new ResizeObserver(() => map.invalidateSize({pan:false})).observe(el("map"));

function showRoute(data) {
  const coordinates = data.map;
  for (const group of Object.values(overlays)) group.clearLayers();
  const polyline = (edges, style) => L.polyline(edges, {interactive: false, ...style});
  polyline(coordinates.roads, {color: "#697a72", weight: 1, opacity: .65}).addTo(overlays.roads);
  for (const run of coordinates.runs) {
    polyline(run, {pane: "matched-route", color: "#fff", weight: 7, opacity: .87}).addTo(overlays.runs);
    polyline(run, {pane: "matched-route", color: "#1764c0", weight: 4, opacity: .98}).addTo(overlays.runs);
  }
  const stride = Math.max(1, Math.ceil(coordinates.gps.length / 3000));
  for (let i = 0; i < coordinates.gps.length; i += stride) {
    L.circleMarker(coordinates.gps[i], {pane:"gps-points", radius:2, stroke:false, fillColor:"#e68a18",
      fillOpacity:.75, interactive:false}).addTo(overlays.gps);
  }
  bounds = L.latLngBounds(coordinates.gps);
  for (const run of coordinates.runs) for (const [a,b] of run) bounds.extend(a).extend(b);
  if (!bounds.isValid()) for (const [a,b] of coordinates.roads) bounds.extend(a).extend(b);
  hasRoute = bounds.isValid();
  el("fit").disabled = !hasRoute;
  fitRoute();
}

async function evaluate() {
  if (!select.value || pending) return;
  pending = true;
  select.disabled = el("evaluate").disabled = true;
  const trip = select.value;
  message("Evaluating trip " + trip + "…");
  try {
    const data = await request("/api/evaluate", {method:"POST", headers:{"Content-Type":"application/json"},
                                               body:JSON.stringify({trip_id:trip})});
    const metrics = data.metrics;
    el("result-title").textContent = "Trip " + data.trip_id;
    el("map-meta").textContent = integer(metrics.valid)+" GPS points · "+
      (data.decoder === "hmm" ? "HMM / Viterbi" : "Geometry preview");
    el("accuracy").textContent = percent(metrics.accuracy);
    el("continuity").textContent = percent(metrics.continuity);
    el("coverage").textContent = percent(metrics.valid ? metrics.assigned/metrics.valid : null);
    el("accuracy-detail").textContent = integer(metrics.correct)+" of "+integer(metrics.assigned)+" assigned points";
    el("coverage-detail").textContent = integer(metrics.assigned)+" of "+integer(metrics.valid)+" valid points";
    el("notice").textContent = data.mode === "placeholder"
      ? "Development preview with distance and heading scores."
      : "Accuracy compares the decoded road to assigned reference-derived labels.";
    showRoute(data);
    message(hasRoute ? "" : "No route geometry available.");
  } catch (error) {
    message(error.message);
  } finally {
    pending = false;
    select.disabled = el("evaluate").disabled = false;
  }
}

el("evaluate").onclick = evaluate;
async function initialize() {
  try {
    const data = await request("/api/trips");
    el("connection").textContent = "API connected";
    el("connection").className = "online";
    el("mode-badge").hidden = data.mode !== "placeholder";
    select.replaceChildren(...data.trips.map(({trip_id}) => new Option(trip_id, trip_id)));
    if (!data.trips.length) { message("No trips are available."); return; }
    select.disabled = el("evaluate").disabled = false;
    message("Choose a trip to view its route.");
    if (data.mode === "placeholder" && data.partition === "student_test") await evaluate();
  } catch {
    el("connection").textContent = "API offline";
    el("connection").className = "offline";
    message("Unable to reach the PathSnap API.");
  }
}
initialize();
