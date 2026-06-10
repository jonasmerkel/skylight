// Camera-motion-compensated temporal detection.
//
// The aircraft moves WITH the sky in world coordinates; the sky (clouds, haze
// gradient, lens dirt, hot pixels) is world-static. Between two frames the
// camera moved by a known amount, so if we REGISTER the previous frame onto the
// current one — cancelling the camera's motion — the static sky subtracts away
// and the plane is the one thing left that moved against it. That makes the
// detection robust at low contrast and straight through cloud, where a pure
// single-frame contrast detector gets fooled.
//
// The known pose delta (Δaz/Δel -> pixels) SEEDS the registration; a tiny local
// search refines it (lens distortion, pose lag, sub-pixel). Pure data + math,
// no I/O, so it is directly unit-testable; the real-frame polarity/sub-pixel
// tuning happens against recorded clips.

export interface Shift {
  dx: number;
  dy: number;
  /** Mean absolute residual at the chosen shift (lower = better alignment). */
  cost: number;
}

export interface MovingBlob {
  /** Centroid in frame fractions (0..1). */
  cx: number;
  cy: number;
  areaPx: number;
  /** Peak residual over the background sigma at that blob. */
  peakSigma: number;
}

const clampU8 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v);

/**
 * Estimate the integer pixel translation aligning `prev` onto `cur` (how the
 * sky moved), searching ±`searchPx` around the (`seedDx`,`seedDy`) pose-delta
 * seed. Coarse SAD on a subsampled grid — the small fast-moving plane is a tiny
 * fraction of the pixels, so it barely perturbs the background-dominated match.
 */
export function estimateShift(
  prev: Uint8Array,
  cur: Uint8Array,
  w: number,
  h: number,
  seedDx = 0,
  seedDy = 0,
  searchPx = 6,
  sample = 4,
): Shift {
  let best: Shift = { dx: seedDx, dy: seedDy, cost: Infinity };
  const m = Math.ceil(searchPx);
  for (let ddy = -m; ddy <= m; ddy++) {
    for (let ddx = -m; ddx <= m; ddx++) {
      const dx = seedDx + ddx;
      const dy = seedDy + ddy;
      let sad = 0;
      let n = 0;
      // Walk a subsampled grid; only count pixels where the shifted source is
      // in-bounds (border margin avoids edge garbage).
      for (let y = 4; y < h - 4; y += sample) {
        const sy = y - dy;
        if (sy < 0 || sy >= h) continue;
        for (let x = 4; x < w - 4; x += sample) {
          const sx = x - dx;
          if (sx < 0 || sx >= w) continue;
          const d = cur[y * w + x] - prev[sy * w + sx];
          sad += d < 0 ? -d : d;
          n++;
        }
      }
      const cost = n > 0 ? sad / n : Infinity;
      if (cost < best.cost) best = { dx, dy, cost };
    }
  }
  return best;
}

/**
 * Residual after compensating the camera motion: |cur - warp(prev, shift)|.
 * The static sky cancels to ~noise; the plane leaves a bright residual at its
 * current position (and a fainter ghost where it was — the positional prior in
 * findMovingBlob disambiguates). Out-of-bounds source pixels yield 0.
 */
export function compensatedResidual(
  prev: Uint8Array,
  cur: Uint8Array,
  w: number,
  h: number,
  shift: Shift,
): Uint8Array {
  const out = new Uint8Array(w * h);
  const { dx, dy } = shift;
  for (let y = 0; y < h; y++) {
    const sy = y - dy;
    const inRowY = sy >= 0 && sy < h;
    for (let x = 0; x < w; x++) {
      if (!inRowY) continue;
      const sx = x - dx;
      if (sx < 0 || sx >= w) continue;
      const d = cur[y * w + x] - prev[sy * w + sx];
      out[y * w + x] = clampU8(d < 0 ? -d : d);
    }
  }
  return out;
}

/** Mean and stddev of a map over its in-bounds (non-zero-able) interior. */
function meanStd(map: Uint8Array, w: number, h: number): { mean: number; std: number } {
  let sum = 0;
  let sum2 = 0;
  let n = 0;
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const v = map[y * w + x];
      sum += v;
      sum2 += v * v;
      n++;
    }
  }
  if (n === 0) return { mean: 0, std: 1 };
  const mean = sum / n;
  const varr = Math.max(1, sum2 / n - mean * mean);
  return { mean, std: Math.sqrt(varr) };
}

/**
 * Find the strongest compact residual blob, preferring those near the predicted
 * position (frame fractions). Returns null if nothing rises clearly above the
 * residual noise floor. A simple flood from the best-scoring peak — the plane
 * is small and bright in the residual, so this is enough; the world-velocity
 * tracker downstream still arbitrates between competing candidates.
 */
export interface FindOptions {
  expectedX?: number;
  expectedY?: number;
  /** Search radius around the expectation, frame fractions (default 0.25). */
  maxDistFrac?: number;
  /** Minimum peak height over the noise floor, sigmas (default 4). */
  minPeakSigma?: number;
}

/** Strongest single moving blob (best near the expectation), or null. */
export function findMovingBlob(
  residual: Uint8Array,
  w: number,
  h: number,
  opts: FindOptions = {},
): MovingBlob | null {
  return findMovingBlobs(residual, w, h, 1, opts)[0] ?? null;
}

/**
 * Up to `maxBlobs` moving blobs from a residual map, strongest first. Each found
 * blob is suppressed before the next search, so they're spatially distinct. The
 * downstream velocity tracker decides which is the plane — this just surfaces
 * the handful of things that actually moved against the (cancelled) sky.
 */
export function findMovingBlobs(
  residual: Uint8Array,
  w: number,
  h: number,
  maxBlobs = 4,
  opts: FindOptions = {},
): MovingBlob[] {
  const { mean, std } = meanStd(residual, w, h);
  const minPeakSigma = opts.minPeakSigma ?? 4;
  const thresh = mean + minPeakSigma * std;
  const ex = opts.expectedX != null ? opts.expectedX * w : null;
  const ey = opts.expectedY != null ? opts.expectedY * h : null;
  const maxDistPx = (opts.maxDistFrac ?? 0.25) * w;
  const maxDist2 = maxDistPx * maxDistPx;
  const work = residual.slice(); // mutable copy; found blobs get zeroed out
  const out: MovingBlob[] = [];

  for (let k = 0; k < maxBlobs; k++) {
    // Best peak: highest residual, tie-broken toward the expected position.
    let peak = -Infinity;
    let px = -1;
    let py = -1;
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const v = work[y * w + x];
        if (v < thresh) continue;
        if (ex != null && ey != null) {
          const dx = x - ex;
          const dy = y - ey;
          if (dx * dx + dy * dy > maxDist2) continue;
        }
        let scoreV = v;
        if (ex != null && ey != null) {
          const dx = x - ex;
          const dy = y - ey;
          scoreV = v - (0.5 * std * Math.sqrt(dx * dx + dy * dy)) / Math.max(1, maxDistPx);
        }
        if (scoreV > peak) {
          peak = scoreV;
          px = x;
          py = y;
        }
      }
    }
    if (px < 0) break;

    // Flood the connected bright region around the peak for area + centroid,
    // zeroing it in the working copy so the next iteration finds a new blob.
    const peakV = work[py * w + px];
    const floor = Math.max(mean + 2 * std, peakV * 0.4);
    let sx = 0;
    let sy = 0;
    let area = 0;
    const stack: number[] = [py * w + px];
    const seen = new Set<number>(stack);
    while (stack.length) {
      const idx = stack.pop()!;
      if (work[idx] < floor) continue;
      const cy = (idx / w) | 0;
      const cx = idx - cy * w;
      sx += cx;
      sy += cy;
      area++;
      work[idx] = 0;
      if (area > 4000) break;
      const nb = [idx - 1, idx + 1, idx - w, idx + w];
      for (const j of nb) {
        if (j < 0 || j >= w * h || seen.has(j)) continue;
        const jy = (j / w) | 0;
        const jx = j - jy * w;
        if (jx < 1 || jx >= w - 1 || jy < 1 || jy >= h - 1) continue;
        seen.add(j);
        if (work[j] >= floor) stack.push(j);
      }
    }
    out.push({
      cx: (area > 0 ? sx / area : px) / w,
      cy: (area > 0 ? sy / area : py) / h,
      areaPx: area,
      peakSigma: (peakV - mean) / std,
    });
  }
  return out;
}

/**
 * Zero the residual at/below the horizon (plus a margin for nearby ground
 * obstructions), computed from the KNOWN camera tilt — no image content needed.
 * The plane is always above the horizon; this drops rooftops/poles/trees for
 * free. `aimElDeg` is the frame-centre elevation, `vfovDeg` the vertical FOV.
 */
export function maskBelowHorizon(
  residual: Uint8Array,
  w: number,
  h: number,
  aimElDeg: number,
  vfovDeg: number,
  marginDeg = 2,
): void {
  if (vfovDeg <= 0) return;
  // el at frame row y (fraction): aimEl - (y-0.5)*vfov. Solve el = -margin.
  const yHorizon = 0.5 + (aimElDeg + marginDeg) / vfovDeg;
  const yStart = Math.max(0, Math.floor(yHorizon * h));
  for (let y = yStart; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) residual[row + x] = 0;
  }
}
