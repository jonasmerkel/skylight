// adsbdb.com enrichment: callsign -> route (origin/dest + airline) and
// hex -> aircraft type/registration. Cached aggressively and persisted to
// disk so a restart doesn't re-hammer the free API. One request per new key.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

const API = "https://api.adsbdb.com/v0";

interface RouteInfo {
  airline?: string;
  origin?: string;
  destination?: string;
  originName?: string;
  destName?: string;
  originLat?: number;
  originLon?: number;
  destLat?: number;
  destLon?: number;
}
interface AircraftInfo {
  typeName?: string;
  registration?: string;
}
interface CacheEntry<T> {
  data: T | null; // null = looked up, not found (negative cache)
  at: number; // ms epoch
}

interface CacheFile {
  routes: Record<string, CacheEntry<RouteInfo>>;
  aircraft: Record<string, CacheEntry<AircraftInfo>>;
}

export class RouteEnricher {
  private cache: CacheFile = { routes: {}, aircraft: {} };
  // Keys currently queued or in flight, so we never enqueue a duplicate.
  private inflight = new Set<string>();
  // Serialized, rate-limited fetch queue: at most one request every
  // `minIntervalMs` so a burst of new aircraft doesn't hammer the free API.
  private queue: Array<() => Promise<void>> = [];
  private pumping = false;
  private minIntervalMs: number;
  private dirty = false;
  private ttlMs: number;

  constructor(
    private cachePath: string,
    ttlHours = 12,
    minIntervalMs = 1000,
  ) {
    this.ttlMs = ttlHours * 3600_000;
    this.minIntervalMs = minIntervalMs;
  }

  /** Enqueue a fetch task; the pump runs them one at a time, ≥1s apart. */
  private enqueue(task: () => Promise<void>): void {
    this.queue.push(task);
    if (!this.pumping) void this.pump();
  }

  private async pump(): Promise<void> {
    this.pumping = true;
    try {
      while (this.queue.length) {
        const task = this.queue.shift()!;
        const start = Date.now();
        await task();
        const wait = this.minIntervalMs - (Date.now() - start);
        if (this.queue.length && wait > 0) {
          await new Promise((r) => setTimeout(r, wait));
        }
      }
    } finally {
      this.pumping = false;
    }
  }

  async load(): Promise<void> {
    try {
      const raw = await readFile(this.cachePath, "utf8");
      const parsed = JSON.parse(raw) as Partial<CacheFile>;
      this.cache = { routes: parsed.routes ?? {}, aircraft: parsed.aircraft ?? {} };
    } catch {
      // first run, no cache yet
    }
    // Persist periodically rather than on every write.
    setInterval(() => void this.flush(), 15_000).unref?.();
  }

  private fresh<T>(e: CacheEntry<T> | undefined, now: number): boolean {
    return !!e && now - e.at < this.ttlMs;
  }

  /** Synchronous read of whatever is cached; kicks off a fetch if missing. */
  enrichSync(
    hex: string,
    callsign: string | undefined,
    now: number,
  ): { route?: RouteInfo; aircraft?: AircraftInfo } {
    const out: { route?: RouteInfo; aircraft?: AircraftInfo } = {};

    const ac = this.cache.aircraft[hex];
    if (this.fresh(ac, now)) out.aircraft = ac!.data ?? undefined;
    else this.fetchAircraft(hex);

    if (callsign) {
      const cs = callsign.trim().toUpperCase();
      const r = this.cache.routes[cs];
      if (this.fresh(r, now)) out.route = r!.data ?? undefined;
      else this.fetchRoute(cs);
    }
    return out;
  }

  private fetchRoute(cs: string): void {
    const key = "r:" + cs;
    if (this.inflight.has(key)) return;
    this.inflight.add(key);
    this.enqueue(async () => {
      try {
        const res = await fetch(`${API}/callsign/${encodeURIComponent(cs)}`, {
          signal: AbortSignal.timeout(8000),
        });
        let data: RouteInfo | null = null;
        if (res.ok) {
          const json: any = await res.json();
          const fr = json?.response?.flightroute;
          if (fr) {
            data = {
              airline: fr.airline?.name,
              origin: fr.origin?.iata_code ?? fr.origin?.icao_code,
              destination: fr.destination?.iata_code ?? fr.destination?.icao_code,
              originName: fr.origin?.municipality,
              destName: fr.destination?.municipality,
              originLat: fr.origin?.latitude,
              originLon: fr.origin?.longitude,
              destLat: fr.destination?.latitude,
              destLon: fr.destination?.longitude,
            };
          }
        }
        this.cache.routes[cs] = { data, at: Date.now() };
        this.dirty = true;
      } catch {
        // leave uncached so we retry later
      } finally {
        this.inflight.delete(key);
      }
    });
  }

  private fetchAircraft(hex: string): void {
    const key = "a:" + hex;
    if (this.inflight.has(key)) return;
    this.inflight.add(key);
    this.enqueue(async () => {
      try {
        const res = await fetch(`${API}/aircraft/${encodeURIComponent(hex)}`, {
          signal: AbortSignal.timeout(8000),
        });
        let data: AircraftInfo | null = null;
        if (res.ok) {
          const json: any = await res.json();
          const a = json?.response?.aircraft;
          if (a) {
            data = {
              typeName: a.manufacturer && a.type ? `${a.manufacturer} ${a.type}` : a.type,
              registration: a.registration,
            };
          }
        }
        this.cache.aircraft[hex] = { data, at: Date.now() };
        this.dirty = true;
      } catch {
        // retry later
      } finally {
        this.inflight.delete(key);
      }
    });
  }

  private async flush(): Promise<void> {
    if (!this.dirty) return;
    this.dirty = false;
    try {
      await mkdir(dirname(this.cachePath), { recursive: true });
      await writeFile(this.cachePath, JSON.stringify(this.cache), "utf8");
    } catch {
      this.dirty = true; // try again next tick
    }
  }
}
