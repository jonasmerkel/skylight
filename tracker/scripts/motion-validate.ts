// Offline validation of camera-motion-compensated detection on REAL clip frames.
// Decodes consecutive frames, estimates the background shift, computes the
// residual, and reports/visualizes the moving blob (the plane) vs the static
// clutter (power lines, clouds, rooftops). Run: tsx tracker/scripts/motion-validate.ts <dir>

import sharp from "sharp";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { estimateShift, compensatedResidual, findMovingBlob } from "../src/vision/motion.js";

const W = 480;
const H = 270;
const dir = process.argv[2] ?? "/tmp/fr";

async function luma(path: string): Promise<Uint8Array> {
  const { data } = await sharp(path).greyscale().resize(W, H, { fit: "fill" }).raw().toBuffer({ resolveWithObject: true });
  return new Uint8Array(data.buffer, data.byteOffset, data.length);
}

async function main(): Promise<void> {
  const files = readdirSync(dir).filter((f) => f.endsWith(".png")).sort();
  let best: { i: number; blob: ReturnType<typeof findMovingBlob>; shift: ReturnType<typeof estimateShift> } | null = null;
  for (let i = 1; i < files.length; i++) {
    const prev = await luma(join(dir, files[i - 1]));
    const cur = await luma(join(dir, files[i]));
    const shift = estimateShift(prev, cur, W, H, 0, 0, 12);
    const resid = compensatedResidual(prev, cur, W, H, shift);
    const blob = findMovingBlob(resid, W, H, { minPeakSigma: 5 });
    const tag = blob ? `PLANE @ (${(blob.cx * W) | 0},${(blob.cy * H) | 0}) ${blob.peakSigma.toFixed(0)}σ area=${blob.areaPx}` : "—";
    console.log(`${files[i]}: shift(${shift.dx},${shift.dy}) cost=${shift.cost.toFixed(1)}  ${tag}`);
    if (blob && (!best || blob.peakSigma > (best.blob?.peakSigma ?? 0))) best = { i, blob, shift };
  }
  if (!best) {
    console.log("no confident mover found");
    return;
  }
  // Render the best pair's residual (boosted) + the cur frame with a marker.
  const prevF = files[best.i - 1];
  const curF = files[best.i];
  const prev = await luma(join(dir, prevF));
  const cur = await luma(join(dir, curF));
  const resid = compensatedResidual(prev, cur, W, H, best.shift);
  // Boost residual for visibility.
  const boosted = Buffer.alloc(W * H);
  for (let i = 0; i < W * H; i++) boosted[i] = Math.min(255, resid[i] * 4);
  await sharp(boosted, { raw: { width: W, height: H, channels: 1 } }).png().toFile("/tmp/residual.png");
  // Mark detection on the cur frame.
  const bx = Math.round(best.blob!.cx * W);
  const by = Math.round(best.blob!.cy * H);
  const rgb = Buffer.alloc(W * H * 3);
  for (let i = 0; i < W * H; i++) { rgb[i * 3] = cur[i]; rgb[i * 3 + 1] = cur[i]; rgb[i * 3 + 2] = cur[i]; }
  for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) {
    if (Math.abs(dx) !== 10 && Math.abs(dy) !== 10) continue;
    const x = bx + dx, y = by + dy;
    if (x >= 0 && x < W && y >= 0 && y < H) { const j = (y * W + x) * 3; rgb[j] = 0; rgb[j + 1] = 255; rgb[j + 2] = 0; }
  }
  await sharp(rgb, { raw: { width: W, height: H, channels: 3 } }).png().toFile("/tmp/detected.png");
  console.log(`\nBEST: ${curF}  shift(${best.shift.dx},${best.shift.dy})  PLANE @ (${bx},${by}) ${best.blob!.peakSigma.toFixed(0)}σ`);
  console.log("wrote /tmp/residual.png and /tmp/detected.png");
}

void main();
