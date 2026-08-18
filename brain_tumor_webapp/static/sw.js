/* BDNeuro-MRI service worker.
   Strategy: network-first for navigations (always fresh page), stale-while-
   revalidate for same-origin static assets. The /api/ endpoint is never cached. */

const CACHE = "bdneuro-v1";
const CORE = [
  "./",
  "./static/css/style.css",
  "./static/js/script.js",
  "./static/favicon.svg",
  "./static/manifest.json"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE)
      .then(cache => cache.addAll(CORE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.indexOf("/api/") !== -1) return;

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then(response => {
          const copy = response.clone();
          caches.open(CACHE).then(cache => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match(request).then(m => m || caches.match("./")))
    );
    return;
  }

  event.respondWith(
    caches.match(request).then(cached => {
      const fetchAndCache = fetch(request).then(response => {
        if (response && response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then(cache => cache.put(request, copy));
        }
        return response;
      });
      return cached || fetchAndCache;
    })
  );
});
