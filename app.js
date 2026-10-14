// Flock Spotter: on-device Flock camera detection with GPS logging. Nothing leaves the phone.
'use strict';

const MODEL_URL = 'model.onnx';
const SIZE = 640;              // model input size
const CONFIRM_HITS = 2;        // detections needed...
const CONFIRM_WINDOW = 3000;   // ...within this many ms to count as a sighting
const DEDUPE_METERS = 80;      // same camera if logged within this distance
const KNOWN_METERS = 120;      // "on map" if a mapped camera is this close

const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
const work = document.createElement('canvas'); work.width = work.height = SIZE;
const wctx = work.getContext('2d', { willReadFrequently: true });

let session = null, running = false, stream = null, wakeLock = null;
let pos = null, watchId = null, soundOn = true, audioCtx = null;
let threshold = parseFloat($('thresh').value);
let hits = [], regionIdx = 0, frames = 0, lastFpsT = performance.now();
let spots = [];

// ---------- storage (IndexedDB) ----------
const dbp = new Promise((res, rej) => {
  const r = indexedDB.open('flockspotter', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('spots', { keyPath: 'id' });
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
});
async function dbAll() {
  const db = await dbp;
  return new Promise((res) => { const q = db.transaction('spots').objectStore('spots').getAll(); q.onsuccess = () => res(q.result || []); q.onerror = () => res([]); });
}
async function dbPut(s) { const db = await dbp; db.transaction('spots', 'readwrite').objectStore('spots').put(s); }
async function dbClear() { const db = await dbp; db.transaction('spots', 'readwrite').objectStore('spots').clear(); }

// ---------- geo ----------
function meters(a, b) {
  const R = 6371000, toR = Math.PI / 180;
  const dLat = (b[0] - a[0]) * toR, dLon = (b[1] - a[1]) * toR;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * toR) * Math.cos(b[0] * toR) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
function nearKnown(p) {
  const list = window.KNOWN_FLOCK || [];
  for (const k of list) {
    if (Math.abs(k[0] - p[0]) > 0.002 || Math.abs(k[1] - p[1]) > 0.003) continue;
    if (meters(k, p) <= KNOWN_METERS) return true;
  }
  return false;
}
function startGps() {
  if (!('geolocation' in navigator)) { $('gps').textContent = 'No GPS'; return; }
  watchId = navigator.geolocation.watchPosition((p) => {
    pos = { lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy, heading: p.coords.heading, t: p.timestamp };
    $('gps').textContent = 'GPS ±' + Math.round(p.coords.accuracy) + 'm';
    $('gps').className = 'pill on';
  }, () => { $('gps').textContent = 'GPS blocked'; $('gps').className = 'pill warn'; },
  { enableHighAccuracy: true, maximumAge: 1000 });
}

// ---------- model ----------
async function loadModel() {
  ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  $('fps').textContent = 'Loading AI…';
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}

// Regions to scan, in video pixel coords. The full frame catches close cameras;
// the two zoomed upper halves catch small, far-away ones on poles.
function regions(vw, vh) {
  const zw = vw * 0.55, zh = Math.min(vh * 0.75, zw);
  return [
    { x: 0, y: 0, w: vw, h: vh },
    { x: 0, y: 0, w: zw, h: zh },
    { x: vw - zw, y: 0, w: zw, h: zh },
  ];
}

async function detect(region) {
  const scale = Math.min(SIZE / region.w, SIZE / region.h);
  const dw = region.w * scale, dh = region.h * scale;
  const ox = (SIZE - dw) / 2, oy = (SIZE - dh) / 2;
  wctx.fillStyle = 'rgb(114,114,114)'; wctx.fillRect(0, 0, SIZE, SIZE);
  wctx.drawImage(video, region.x, region.y, region.w, region.h, ox, oy, dw, dh);
  const px = wctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, data = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    data[i] = px[j] / 255; data[i + n] = px[j + 1] / 255; data[i + 2 * n] = px[j + 2] / 255;
  }
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]) });
  const o = out[session.outputNames[0]];
  const [, rows, cols] = o.dims; const d = o.data;
  const boxes = [];
  for (let c = 0; c < cols; c++) {
    let best = 0;
    for (let k = 4; k < rows; k++) best = Math.max(best, d[k * cols + c]);
    if (best < threshold) continue;
    const cx = d[c], cy = d[cols + c], w = d[2 * cols + c], h = d[3 * cols + c];
    boxes.push({
      score: best,
      x: region.x + (cx - w / 2 - ox) / scale, y: region.y + (cy - h / 2 - oy) / scale,
      w: w / scale, h: h / scale,
    });
  }
  return nms(boxes, 0.45);
}
function nms(b, iouT) {
  b.sort((a, c) => c.score - a.score);
  const keep = [];
  for (const x of b) {
    if (keep.every((k) => iou(k, x) < iouT)) keep.push(x);
    if (keep.length >= 10) break;
  }
  return keep;
}
function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return i / (a.w * a.h + b.w * b.h - i + 1e-6);
}

// ---------- loop ----------
async function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.videoWidth) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (overlay.width !== vw) { overlay.width = vw; overlay.height = vh; }
    const regs = regions(vw, vh);
    const reg = regs[regionIdx++ % regs.length];
    let found = [];
    try { found = await detect(reg); } catch (e) { console.error(e); }
    draw(found, reg);
    const now = Date.now();
    if (found.length) {
      hits.push({ t: now, box: found[0] });
      hits = hits.filter((h) => now - h.t < CONFIRM_WINDOW);
      if (hits.length >= CONFIRM_HITS) { const best = hits.reduce((a, b) => (b.box.score > a.box.score ? b : a)); hits = []; sighting(best.box); }
    }
    frames++;
    const t = performance.now();
    if (t - lastFpsT > 2000) { $('fps').textContent = 'AI ' + (frames * 1000 / (t - lastFpsT)).toFixed(1) + '/s'; $('fps').className = 'pill on'; frames = 0; lastFpsT = t; }
  }
  requestAnimationFrame(loop);
}
function draw(found, reg) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.lineWidth = Math.max(2, overlay.width / 300);
  octx.strokeStyle = 'rgba(255,255,255,0.15)'; octx.strokeRect(reg.x, reg.y, reg.w, reg.h);
  octx.strokeStyle = '#ffb020'; octx.fillStyle = '#ffb020'; octx.font = `${Math.round(overlay.width / 40)}px system-ui`;
  for (const b of found) {
    octx.strokeRect(b.x, b.y, b.w, b.h);
    octx.fillText('Flock ' + Math.round(b.score * 100) + '%', b.x, Math.max(20, b.y - 6));
  }
}

async function sighting(box) {
  const here = pos ? [pos.lat, pos.lon] : null;
  const now = Date.now();
  // dedupe: same place (or, without GPS, the last 20 seconds)
  const dup = spots.find((s) => (here && s.lat != null ? meters(here, [s.lat, s.lon]) < DEDUPE_METERS : now - s.t < 20000));
  if (dup) return;
  const s = {
    id: now + '-' + Math.random().toString(36).slice(2, 7), t: now,
    lat: here ? here[0] : null, lon: here ? here[1] : null, acc: pos ? Math.round(pos.acc) : null,
    score: Math.round(box.score * 100) / 100, onMap: here ? nearKnown(here) : null, thumb: thumb(box),
  };
  spots.unshift(s); await dbPut(s); render(); alertUser();
}
function thumb(b) {
  const c = document.createElement('canvas'); const pad = Math.max(b.w, b.h) * 1.5;
  const x = Math.max(0, b.x - pad), y = Math.max(0, b.y - pad);
  const w = Math.min(video.videoWidth - x, b.w + 2 * pad), h = Math.min(video.videoHeight - y, b.h + 2 * pad);
  c.width = 240; c.height = Math.round(240 * h / w) || 180;
  c.getContext('2d').drawImage(video, x, y, w, h, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.7);
}
function alertUser() {
  const b = $('banner'); b.style.display = 'block'; setTimeout(() => (b.style.display = 'none'), 2500);
  if (navigator.vibrate) navigator.vibrate([120, 60, 120]);
  if (!soundOn) return;
  try {
    audioCtx = audioCtx || new AudioContext();
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.frequency.value = 880; g.gain.setValueAtTime(0.25, audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
    o.connect(g).connect(audioCtx.destination); o.start(); o.stop(audioCtx.currentTime + 0.35);
  } catch (e) { /* sound is optional */ }
}

// ---------- UI ----------
function render() {
  const ul = $('log'); ul.textContent = '';
  for (const s of spots) {
    const li = document.createElement('li');
    const img = document.createElement('img'); img.src = s.thumb; img.alt = 'Spotted camera';
    const m = document.createElement('div'); m.className = 'meta';
    const when = new Date(s.t);
    const b = document.createElement('b'); b.textContent = when.toLocaleDateString() + ' ' + when.toLocaleTimeString();
    m.append(b);
    if (s.onMap != null) { const tg = document.createElement('span'); tg.className = 'tag' + (s.onMap ? '' : ' new'); tg.textContent = s.onMap ? 'on map' : 'new'; m.append(tg); }
    m.append(document.createElement('br'));
    if (s.lat != null) {
      const a = document.createElement('a'); a.href = `https://www.openstreetmap.org/?mlat=${s.lat}&mlon=${s.lon}#map=19/${s.lat}/${s.lon}`;
      a.target = '_blank'; a.rel = 'noopener'; a.textContent = s.lat.toFixed(5) + ', ' + s.lon.toFixed(5);
      m.append(a, document.createTextNode(` ±${s.acc}m · ${Math.round(s.score * 100)}%`));
    } else m.append(document.createTextNode(`No GPS fix · ${Math.round(s.score * 100)}%`));
    li.append(img, m); ul.append(li);
  }
  $('count').textContent = spots.length;
  $('empty').style.display = spots.length ? 'none' : 'block';
}

async function start() {
  if (running) return stop();
  $('start').disabled = true;
  try {
    if (!session) await loadModel();
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
    video.srcObject = stream; await video.play();
    $('idle').style.display = 'none';
    startGps();
    try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (e) { /* optional */ }
    running = true; $('start').textContent = 'Stop'; loop();
  } catch (e) {
    $('idle').textContent = 'Could not start: ' + (e.message || e) + (/Permission|NotAllowed/.test(String(e)) ? ' Allow camera and location for this page, then try again.' : '');
  }
  $('start').disabled = false;
}
function stop() {
  running = false; stream?.getTracks().forEach((t) => t.stop()); stream = null;
  if (watchId != null) navigator.geolocation.clearWatch(watchId); watchId = null;
  wakeLock?.release?.(); wakeLock = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $('start').textContent = 'Start'; $('fps').textContent = 'AI off'; $('fps').className = 'pill';
  $('idle').style.display = 'grid'; $('idle').textContent = 'Stopped. Tap Start to scan again.';
}
async function exportSpots() {
  const fc = { type: 'FeatureCollection', features: spots.filter((s) => s.lat != null).map((s) => ({
    type: 'Feature', geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
    properties: { time: new Date(s.t).toISOString(), accuracy_m: s.acc, confidence: s.score, on_map: s.onMap },
  })) };
  const file = new File([JSON.stringify(fc, null, 1)], 'flock-spots.geojson', { type: 'application/geo+json' });
  if (navigator.canShare?.({ files: [file] })) { try { await navigator.share({ files: [file], title: 'Flock spots' }); return; } catch (e) { /* fall back */ } }
  const a = document.createElement('a'); a.href = URL.createObjectURL(file); a.download = file.name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

$('start').onclick = start;
$('sound').onclick = () => { soundOn = !soundOn; $('sound').textContent = soundOn ? 'Sound on' : 'Sound off'; };
$('thresh').oninput = (e) => { threshold = parseFloat(e.target.value); $('threshv').textContent = threshold.toFixed(2); };
$('export').onclick = exportSpots;
$('clear').onclick = async () => { if (confirm('Delete all logged spots on this phone?')) { await dbClear(); spots = []; render(); } };
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && running && !wakeLock) { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (e) { /* optional */ } }
});
dbAll().then((all) => { spots = all.sort((a, b) => b.t - a.t); render(); });
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
