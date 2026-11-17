// Network first, falling back to the cache when offline.
const CACHE = 'baby-companion-v2';
const SHELL = ['/', '/static/manifest.json', '/static/vendor/tailwindcss-3.4.17.js'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;  // uploads, logging and the agent stream go straight to the network
  e.respondWith(
    fetch(e.request)
      .then(res => {
        if (res.ok && SHELL.includes(new URL(e.request.url).pathname)) {
          const copy = res.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, copy));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
