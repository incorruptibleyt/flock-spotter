// Cache the app, AI runtime and model so it keeps working with weak signal.
const CACHE = 'flockspotter-v2';
const FILES = ['./', 'index.html', 'app.js', 'known.js', 'model.onnx', 'manifest.webmanifest', 'icon.svg',
  'ort/ort.wasm.min.js', 'ort/ort-wasm-simd-threaded.wasm', 'ort/ort-wasm-simd-threaded.mjs'];
self.addEventListener('install', (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener('activate', (e) => e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return;
  // network first so updates arrive, cache when offline
  e.respondWith(fetch(e.request).then((r) => { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request, { ignoreSearch: true })));
});
