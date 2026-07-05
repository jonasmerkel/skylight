// SFO surface traffic from the airplanes.live area API — feeds the "who's
// taxiing / who's next" panel on the TV and the Twitch stream. The local
// receiver rarely hears surface targets 13 mi away at ground level, so this
// comes from the aggregator instead.
//
// Polite polling: one request every POLL_MS (airplanes.live asks hobby users
// to stay around 1 req/s; we're far under). Failures skip the tick and keep
// the last snapshot — the panel just shows slightly stale dots.

import type { GroundAircraft } from "@shared/index.js";
import { acquireAirplanesLive } from "./airplanes-live.js";

const SFO_LAT = 37.6213;
const SFO_LON = -122.379;
const RADIUS_NM = 3;
const POLL_MS = 6000;
const URL = `https://api.airplanes.live/v2/point/${SFO_LAT}/${SFO_LON}/${RADIUS_NM}`;

/** Raw airplanes.live aircraft record (the fields we read). */
interface AlAircraft {
  hex?: string;
  flight?: string;
  r?: string;
  t?: string;
  category?: string;
  alt_baro?: number | "ground";
  gs?: number;
  track?: number;
  lat?: number;
  lon?: number;
}

export class SfoGroundPoller {
  private timer: ReturnType<typeof setInterval> | null = null;
  private last: { at: number; aircraft: GroundAircraft[] } | null = null;
  private lastErrorLogAt = 0;
  private polling = false;

  constructor(private onUpdate: (at: number, aircraft: GroundAircraft[]) => void) {}

  /** Latest snapshot for late-joining clients (null until first success). */
  getSnapshot(): { at: number; aircraft: GroundAircraft[] } | null {
    return this.last;
  }

  start(): void {
    if (this.timer) return;
    void this.poll();
    this.timer = setInterval(() => void this.poll(), POLL_MS);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private async poll(): Promise<void> {
    // A gated request can outlast the poll interval; don't stack a second run.
    if (this.polling) return;
    this.polling = true;
    try {
      // Share airplanes.live's 1 req/s budget with the main poller.
      await acquireAirplanesLive();
      const res = await fetch(URL, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as { ac?: AlAircraft[] };
      const aircraft: GroundAircraft[] = [];
      for (const a of body.ac ?? []) {
        if (a.alt_baro !== "ground") continue;
        if (a.lat == null || a.lon == null || !a.hex) continue;
        // Surface VEHICLES are ADS-B category C; TIS-B tracks with no
        // identity at all are almost always vehicles too. Keep aircraft.
        if (a.category?.startsWith("C")) continue;
        if (!a.t && !a.flight && !a.r) continue;
        aircraft.push({
          hex: a.hex,
          flight: a.flight?.trim() || undefined,
          reg: a.r,
          typeCode: a.t,
          lat: a.lat,
          lon: a.lon,
          trackDeg: a.track,
          gsKt: a.gs,
        });
      }
      const at = Date.now();
      this.last = { at, aircraft };
      this.onUpdate(at, aircraft);
    } catch (err) {
      // Quietly tolerant: log at most once a minute.
      const now = Date.now();
      if (now - this.lastErrorLogAt > 60_000) {
        this.lastErrorLogAt = now;
        console.warn(`[sfo-ground] poll failed: ${(err as Error).message}`);
      }
    } finally {
      this.polling = false;
    }
  }
}
