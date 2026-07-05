import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG } from "@shared/index.js";
import { Poller, type PollerOptions } from "../src/datasource.js";
import type { RouteEnricher } from "../src/enrich/routes.js";
import { resetAirplanesLiveGate } from "../src/airplanes-live.js";

// Regression test for #15: polling airplanes.live must never exceed its 1 req/s
// limit — the double request rate trips the limit and makes aircraft flicker
// out and back. Two defenses: the supplement timer is torn down when the API is
// the *primary* source, and a shared rate gate serializes every airplanes.live
// request so their combined rate stays within the limit even if timers drift.

const stubEnricher = { enrichSync: () => ({}) } as unknown as RouteEnricher;

function makeOpts(over: Partial<PollerOptions>): PollerOptions {
  return {
    source: "api",
    apiUrlTemplate: "https://api.example/{lat}/{lon}/{r}",
    pollMs: 1000,
    supplementApi: true,
    apiPollMs: 4000,
    getConfig: () => DEFAULT_CONFIG,
    enricher: stubEnricher,
    onSnapshot: () => {},
    onStatus: () => {},
    ...over,
  };
}

describe("Poller supplement-timer lifecycle (#15)", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    resetAirplanesLiveGate();
    fetchSpy = vi.fn(async () => ({
      ok: true,
      json: async () => ({ aircraft: [] }),
    }));
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("paces the API to the rate limit when it is the primary source", async () => {
    const poller = new Poller(makeOpts({ source: "api" }));
    poller.start();
    await vi.advanceTimersByTimeAsync(4100);
    poller.stop();
    // The gate's ~1s spacing lets ticks through at 0, 1050, 2100, 3150 — four in
    // 4100ms, never the five a raw 1s poll interval would otherwise fire.
    expect(fetchSpy).toHaveBeenCalledTimes(4);
  });

  it("runs the supplement timer only while radio is primary", async () => {
    const poller = new Poller(makeOpts({ source: "radio" }));
    poller.start();
    await vi.advanceTimersByTimeAsync(1); // flush immediate radio + supplement polls
    const afterStart = fetchSpy.mock.calls.length;
    expect(afterStart).toBeGreaterThanOrEqual(2); // radio tick + supplement refresh

    // Switching to API should tear the supplement timer down; the gate then
    // caps the primary poll on its own.
    poller.setSource("api");
    resetAirplanesLiveGate(); // measure the post-switch window in isolation
    fetchSpy.mockClear();
    await vi.advanceTimersByTimeAsync(4100);
    poller.stop();
    // No supplement doubling, and the gate holds the API to at most one request
    // per ~1s over the window.
    expect(fetchSpy.mock.calls.length).toBeLessThanOrEqual(4);
    expect(fetchSpy.mock.calls.length).toBeGreaterThan(0);
  });
});
