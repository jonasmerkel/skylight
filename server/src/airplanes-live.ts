// Shared rate gate for airplanes.live's REST API, which allows one request per
// second per client. Several pollers hit that host independently — the main
// API poll, its radio supplement, and the SFO ground panel. Each stays under
// 1/s on its own, but their timers drift into the same second and trip a 429,
// which then forces a backoff that freezes the display. Route every
// airplanes.live request through this one gate so their combined rate is
// serialized to at most one request per second.

// Comfortably over 1s: we space requests by their *send* time, but the server
// enforces the limit by *arrival* time, and network-latency jitter between two
// requests can compress that gap. A 1050ms spacing left almost no margin and
// still tripped 429s; the extra headroom absorbs the jitter. Tunable via env
// for slow or contended links. The display extrapolates between fixes, so a
// slightly slower poll is invisible.
const MIN_SPACING_MS = Number(process.env.AIRPLANES_LIVE_MIN_MS ?? 1250);

let chain: Promise<void> = Promise.resolve();

/**
 * Reserve the next airplanes.live request slot. Await the returned promise,
 * then fire the request. Consecutive acquisitions resolve at least
 * MIN_SPACING_MS apart, regardless of which poller asked, so the aggregate
 * request rate to the host never exceeds the limit. The first caller after an
 * idle period proceeds immediately.
 */
export function acquireAirplanesLive(): Promise<void> {
  const ready = chain;
  let release!: () => void;
  chain = new Promise<void>((r) => (release = r));
  return ready.then(() => {
    // Hold the next caller off until MIN_SPACING_MS after this one starts.
    setTimeout(release, MIN_SPACING_MS);
  });
}

/** Test-only: drop any pending pacing so each case starts from a clean gate. */
export function resetAirplanesLiveGate(): void {
  chain = Promise.resolve();
}
