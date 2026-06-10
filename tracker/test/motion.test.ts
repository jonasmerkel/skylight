import { describe, expect, it } from "vitest";
import { estimateShift, compensatedResidual, findMovingBlob } from "../src/vision/motion.js";

// Camera-motion-compensated detection: a bright plane moving over a textured
// sky that the camera (and so the whole frame) panned between frames. After
// compensating the known/estimated background shift, the plane is the residual.

const W = 120;
const H = 80;

function makeSky(seed: number): Uint8Array {
  // Deterministic textured "sky": smooth gradient + fixed pseudo-random grain.
  const a = new Uint8Array(W * H);
  let s = seed >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 0xffffffff);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const grad = 60 + (x / W) * 40 + (y / H) * 20;
      a[y * W + x] = Math.min(200, grad + rnd() * 8);
    }
  }
  return a;
}

function shiftCopy(src: Uint8Array, dx: number, dy: number): Uint8Array {
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const sy = y - dy;
    for (let x = 0; x < W; x++) {
      const sx = x - dx;
      out[y * W + x] = sx >= 0 && sx < W && sy >= 0 && sy < H ? src[sy * W + sx] : 70;
    }
  }
  return out;
}

function drawDot(a: Uint8Array, cx: number, cy: number, val = 250, r = 2): void {
  for (let y = -r; y <= r; y++) {
    for (let x = -r; x <= r; x++) {
      const px = cx + x;
      const py = cy + y;
      if (px >= 0 && px < W && py >= 0 && py < H && x * x + y * y <= r * r) {
        a[py * W + px] = val;
      }
    }
  }
}

describe("estimateShift", () => {
  it("recovers the background pan even with a moving plane present", () => {
    const sky = makeSky(7);
    const prev = sky.slice();
    drawDot(prev, 30, 40); // plane in prev
    const cur = shiftCopy(sky, 3, -2); // sky panned (3,-2)
    drawDot(cur, 45, 38); // plane moved elsewhere (not sky+shift)
    const s = estimateShift(prev, cur, W, H, 0, 0, 6);
    expect(s.dx).toBe(3);
    expect(s.dy).toBe(-2);
  });
});

describe("compensatedResidual + findMovingBlob", () => {
  it("locates the plane at its current position after motion compensation", () => {
    const sky = makeSky(11);
    const prev = sky.slice();
    drawDot(prev, 28, 44);
    const cur = shiftCopy(sky, 2, 1);
    drawDot(cur, 70, 30); // current plane position
    const shift = estimateShift(prev, cur, W, H, 0, 0, 6);
    const resid = compensatedResidual(prev, cur, W, H, shift);
    const blob = findMovingBlob(resid, W, H, { expectedX: 70 / W, expectedY: 30 / H });
    expect(blob).not.toBeNull();
    expect(blob!.cx * W).toBeGreaterThan(64);
    expect(blob!.cx * W).toBeLessThan(76);
    expect(blob!.cy * H).toBeGreaterThan(24);
    expect(blob!.cy * H).toBeLessThan(36);
    expect(blob!.peakSigma).toBeGreaterThan(4);
  });

  it("returns null when only the static sky panned (no real mover)", () => {
    const sky = makeSky(3);
    const prev = sky.slice();
    const cur = shiftCopy(sky, 2, -1);
    const shift = estimateShift(prev, cur, W, H, 0, 0, 6);
    const resid = compensatedResidual(prev, cur, W, H, shift);
    const blob = findMovingBlob(resid, W, H, { minPeakSigma: 6 });
    expect(blob).toBeNull();
  });
});
