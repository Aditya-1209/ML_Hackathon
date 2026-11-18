const el = id => document.getElementById(id);
const query = new URLSearchParams(location.search);
const api = (query.get("api") || "").replace(/\/$/, "");
if (api && !/^https?:\/\//i.test(api)) throw new Error("The API address must be HTTP or HTTPS.");
const canvas = el("canvas");
const ctx = canvas.getContext("2d");
let selected, pending = false, result, camera, drag;
const layers = Object.fromEntries([...document.querySelectorAll("[data-layer]")]
  .map(input => [input.dataset.layer, input]));
const percentage = number => number == null ? "â€”" : (100 * number).toFixed(2) + "%";
const integer = number => new Intl.NumberFormat("en").format(number);

async function request(path, options) {
  const response = await fetch(api + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The API could not complete this request.");
  return data;
}

function setConnection(label, status) {
  const connection = el("connection");
  connection.className = "connection " + status;
  connection.replaceChildren(document.createElement("i"), " " + label);
}

function message(text) {
  el("map-message").textContent = text;
  el("map-message").hidden = !text;
}

function setSelection(id) {
  selected = id;
  for (const button of el("trip-list").children)
    button.setAttribute("aria-pressed", String(button.dataset.id === id));
  el("evaluate").disabled = pending;
}

function resetCamera() {
  if (!result) return;
  const xs = [], ys = [];
  for (const [x,y] of result.map.gps) { xs.push(x); ys.push(y); }
  for (const run of result.map.runs) for (const [x,y,u,v] of run) { xs.push(x,u); ys.push(y,v); }
  if (!xs.length) for (const [x,y,u,v] of result.map.roads) { xs.push(x,u); ys.push(y,v); }
  const [xmin, xmax] = xs.reduce(([lo,hi], x) => [Math.min(lo,x),Math.max(hi,x)],[Infinity,-Infinity]);
  const [ymin, ymax] = ys.reduce(([lo,hi], y) => [Math.min(lo,y),Math.max(hi,y)],[Infinity,-Infinity]);
  if (!xs.length) { camera = undefined; draw(); return; }
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const scale = Math.max(.0001, Math.min((w - 105) / Math.max(xmax-xmin, 50),
                                       (h - 100) / Math.max(ymax-ymin, 50)));
  camera = {x:(xmin+xmax)/2, y:(ymin+ymax)/2, scale};
  draw();
}

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight, ratio = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
  }
  ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.fillStyle = "#11251e"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#1c352b"; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 8; x < w; x += 28) { ctx.moveTo(x,0); ctx.lineTo(x,h); }
  for (let y = 8; y < h; y += 28) { ctx.moveTo(0,y); ctx.lineTo(w,y); }
  ctx.stroke();
  el("map-scale").hidden = !camera;
  if (!camera || !result) return;
  const scale = camera.scale;
  const x = value => (value-camera.x)*scale + w/2;
  const y = value => h/2 - (value-camera.y)*scale;
  const lines = (segments,color,width) => {
    ctx.beginPath();
    for (const [ax,ay,bx,by] of segments) { ctx.moveTo(x(ax),y(ay)); ctx.lineTo(x(bx),y(by)); }
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.lineCap="round"; ctx.lineJoin="round"; ctx.stroke();
  };
  if (layers.roads.checked) lines(result.map.roads,"#567565",1.3);
  if (layers.runs.checked) for (const run of result.map.runs) {
    lines(run,"#285b48",8); lines(run,"#73efb5",3.7);
  }
  if (layers.gps.checked) {
    ctx.fillStyle="#f7bd66c9"; ctx.beginPath();
    const step = Math.max(1,Math.ceil(result.map.gps.length/3000));
    for (let i=0; i<result.map.gps.length; i+=step) {
      const [a,b]=result.map.gps[i]; const px=x(a), py=y(b);
      ctx.moveTo(px+1.6,py);ctx.arc(px,py,1.6,0,2*Math.PI);
    }
    ctx.fill();
  }
  const metres = Math.max(1,Math.round(85/scale));
  el("map-scale").textContent = metres >= 1000 ? (metres/1000).toFixed(1)+" km" : metres+" m";
}

function zoom(factor) {
  if (!camera) return;
  camera.scale = Math.min(40,Math.max(.0001,camera.scale*factor));
  draw();
}
el("zoom-in").onclick=()=>zoom(1.35);
el("zoom-out").onclick=()=>zoom(1/1.35);
el("reset-view").onclick=resetCamera;
for (const input of Object.values(layers)) input.addEventListener("change",draw);
canvas.addEventListener("wheel",event=>{if(camera){event.preventDefault();zoom(event.deltaY<0?1.18:1/1.18);}},{passive:false});
canvas.addEventListener("pointerdown",event=>{
  if (!camera) return;
  drag = {pointer:event.pointerId,x:event.clientX,y:event.clientY};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove",event=>{
  if (!drag || drag.pointer!==event.pointerId) return;
  camera.x-=(event.clientX-drag.x)/camera.scale;
  camera.y+=(event.clientY-drag.y)/camera.scale;
  drag.x=event.clientX;drag.y=event.clientY;draw();
});
for (const event of ["pointerup","pointercancel","lostpointercapture"])
  canvas.addEventListener(event,()=>{drag=undefined;});
new ResizeObserver(draw).observe(canvas);

async function evaluate() {
  if (!selected || pending) return;
  const trip = selected;
  pending = true;
  for (const button of el("trip-list").children) button.disabled = true;
  el("evaluate").disabled = true;
  result = undefined; camera = undefined; draw();
  for (const id of ["accuracy","continuity","coverage"]) el(id).textContent = "â€”";
  el("accuracy-detail").textContent = "Assigned reference labels";
  el("coverage-detail").textContent = "Assigned GPS points";
  el("map-title").textContent = "Trip " + trip;
  el("map-meta").textContent = "Waiting for the API";
  message("Evaluating trip " + trip + "â€¦");
  try {
    result = await request("/api/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},
                                           body:JSON.stringify({trip_id:trip})});
    const m = result.metrics;
    el("map-meta").textContent = integer(m.valid)+" GPS points Â· "+integer(result.route_runs)+" route runs Â· "+
      (result.decoder==="hmm"?"HMM / Viterbi":"Geometry preview");
    el("accuracy").textContent = percentage(m.accuracy);
    el("continuity").textContent = percentage(m.continuity);
    el("coverage").textContent = percentage(m.valid ? m.assigned/m.valid : null);
    el("accuracy-detail").textContent = integer(m.correct)+" of "+integer(m.assigned)+" assigned points";
    el("coverage-detail").textContent = integer(m.assigned)+" of "+integer(m.valid)+" valid points";
    el("notice").textContent = result.mode==="placeholder"
      ? "Placeholder scores use distance and heading. Measurements are for the demo."
      : "Accuracy compares the decoded road to assigned reference-derived labels.";
    message(""); resetCamera();
  } catch (error) {
    el("map-meta").textContent = "Evaluation unavailable";
    message(error.message);
  } finally {
    pending = false;
    for (const button of el("trip-list").children) button.disabled = false;
    el("evaluate").disabled = !selected;
  }
}

el("evaluate").onclick = evaluate;
async function initialize() {
  try {
    const data = await request("/api/trips");
    setConnection("API connected","online");
    el("mode-badge").hidden = data.mode !== "placeholder";
    el("trip-count").textContent = data.trip_count+" available";
    for (const {trip_id} of data.trips) {
      const button = document.createElement("button");
      button.type="button"; button.dataset.id=trip_id;button.textContent="Trip "+trip_id;
      button.onclick=()=>setSelection(trip_id);
      el("trip-list").append(button);
    }
    if (!data.trips.length) { message("No trips are available."); return; }
    setSelection(data.trips[0].trip_id);
    message("Select a trip and run evaluation.");
    if (data.mode === "placeholder" && data.partition === "student_test") await evaluate();
  } catch {
    setConnection("API offline","offline");
    message("Unable to reach the PathSnap API.");
    el("notice").textContent = "Start the API or check the API address for this page.";
  }
}
initialize();
