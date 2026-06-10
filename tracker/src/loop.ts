// The control loop: ties the upstream aircraft feed, the pointing pipeline,
// the calibration session, and the camera driver together, and assembles the
// TrackerState the debug UI renders.
//
// Cadence: aircraft fixes arrive ~1 Hz; the loop ticks at predict.commandHz
// (default 15 Hz). On each fresh fix the alpha-beta setpoint trackers are fed
// an observation; every tick they glide toward it, rate-limited, and the
// (deadbanded) result goes to the camera as an absolute move.

import {
  AxisTracker,
  azElFromSite,
  hfovFromZoomUnits,
  mountFromWorld,
  norm180,
  worldFromMount,
  zoomUnitsFromHfov,
  type Aircraft,
  type AzEl,
  type CalibrationRef,
  type CameraPose,
  type Config,
  type PanTilt,
  type TargetMode,
  type TrackerConfig,
  type TrackerMode,
  type TrackerState,
} from "@shared/index.js";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { CameraDriver } from "./camera/driver.js";
import { AutoCalibrator } from "./calibration/auto.js";
import { CalibrationSession } from "./calibration/session.js";
import { planPass, zenithHold, ZENITH_MIN_HFOV } from "./pointing/planner.js";
import { predictAim, TrackHistory, type Prediction } from "./pointing/predict.js";
import { selectTarget, type CurrentTarget } from "./pointing/target.js";
import { chooseZoom } from "./pointing/zoom.js";
import { decodeLuma, VISION_W, VISION_H, type Detection } from "./vision/detect.js";
import {
  estimateShift,
  compensatedResidual,
  findMovingBlobs,
  maskBelowHorizon,
} from "./vision/motion.js";
import { TrackTable, type WorldObs } from "./vision/tracks.js";
import type { Recorder } from "./record.js";
import type { Upstream } from "./upstream.js";
import type { VideoStream } from "./video/stream.js";

const TRACKER_DIR = dirname(fileURLToPath(import.meta.url));
const AUTOCAL_FILE = resolve(TRACKER_DIR, "../data/autocal.json");

export class ControlLoop {
  private mode: TrackerMode = "auto";
  private manualHex: string | null = null;
  private current: CurrentTarget | null = null;
  private history = new TrackHistory();
  private azTracker = new AxisTracker(true);
  private elTracker = new AxisTracker(false);
  private lastObservedTs = 0;
  private lastCommanded: CameraPose | null = null;
  /** Last acquisition move sent (dedupe; null while in velocity pursuit). */
  private lastAbsolute: CameraPose | null = null;
  /** A velocity drive is active — must be zeroed if the target vanishes. */
  private pursuing = false;
  /** Idle ready-position state. */
  private lastTargetAt = 0;
  private atHome = false;
  /** Last carrot re-issue time. */
  private lastCarrotAt = 0;
  /** Last commanded velocity rates (direction-flip hysteresis + accel cap). */
  private lastPanRateCmd = 0;
  private lastTiltRateCmd = 0;
  /** Low-passed pose error for the velocity P term (pose-snap spike damping). */
  private errPanLP = 0;
  private errTiltLP = 0;
  /** Learned rate deficit (integral term) — the dither's accel transients
   *  make the true average rate run under the commanded one, and P alone
   *  holds a steady error to compensate (= visible trailing). */
  private panRateI = 0;
  private tiltRateI = 0;

  // --- vision (Phase B) ---
  private visionTimer: ReturnType<typeof setInterval> | null = null;
  private visionBusy = false;
  private lastDetection:
    | (Detection & { t: number; frameT: number; offAzDeg: number; offElDeg: number })
    | null = null;
  /** Estimated ADS-B-vs-vision aim bias (blob − prediction), world deg. */
  private corrAz = 0;
  private corrEl = 0;
  /** What's actually applied to the aim — slews toward corrAz/corrEl at a
   *  bounded rate so corrections GLIDE in (per-detection steps read as jank). */
  private corrAzApp = 0;
  private corrElApp = 0;
  /** World-frame candidate tracks: the plane is whichever one MOVES like
   *  the ADS-B prediction, not whichever blob shone brightest this frame. */
  private tracks = new TrackTable();
  /** Continuous auto-calibration from locked passes. */
  private autoCal = new AutoCalibrator(AUTOCAL_FILE);
  private lastAutoCalAt = 0;
  /** Previous vision frame (luma) + its pose, for motion-compensated diffing. */
  private prevLuma: Uint8Array | null = null;
  private prevAim: { azDeg: number; elDeg: number; hfov: number } | null = null;
  /** Since when the detection has been continuously near center (0 = not). */
  private visionGoodSince = 0;
  /** Zoom ladder rung (0 = full wide); climbs as vision lock sustains. */
  private ladderIdx = 0;
  private lastLadderAt = 0;
  /** Focus-on-plane state: zoom units at the last one-push AF, and when.
   *  lastFocusZoom: >0 = focused at that zoom; 0 = unfocused/reset; -1 = a sweep
   *  failed and we fell back to the infinity stop (re-arm on the next clean frame). */
  private lastFocusZoom = 0;
  private lastFocusAt = 0;
  /** When the last one-push AF was fired (0 = none pending a result check). */
  private focusCheckAt = 0;
  /**
   * Recent aim/prediction history so vision can reference the pose AT FRAME
   * TIME — frames lag ~0.7 s, and measuring a blob in an old frame against
   * the CURRENT pose creates a motion ghost that rails the correction.
   */
  private aimHistory: {
    t: number;
    aimAz: number;
    aimEl: number;
    predAz: number;
    predEl: number;
    /** The plane's TRUE direction at entry time (no aim lead) — the
     *  calibration truth. */
    truthAz: number;
    truthEl: number;
    /** Setpoint angular rates (for the tracker's velocity matching). */
    rateAzDps: number;
    rateElDps: number;
    /** Mechanical pose (auto-calibration needs raw pan/tilt). */
    panDeg: number;
    tiltDeg: number;
    hfov: number;
  }[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastState: TrackerState | null = null;

  readonly calibration = new CalibrationSession();

  constructor(
    private upstream: Upstream,
    private driver: () => CameraDriver,
    private recorder: Recorder,
    private video: VideoStream,
    private swapDriver: (kind: "sim" | "visca") => void,
    private mseStatus: () => { running: boolean; gen: number } = () => ({
      running: false,
      gen: 0,
    }),
    private videoRecStatus: () => {
      recording: boolean;
      file?: string;
      startedAt?: number;
    } = () => ({ recording: false }),
  ) {}

  // --- lifecycle ---

  start(): void {
    if (this.timer) return;
    this.rescheduleTick();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    if (this.visionTimer) clearInterval(this.visionTimer);
    this.timer = null;
    this.visionTimer = null;
  }

  private rescheduleTick(): void {
    if (this.timer) clearInterval(this.timer);
    const hz = this.cfg().predict.commandHz || 15;
    this.timer = setInterval(() => this.tick(), 1000 / hz);
    if (this.visionTimer) clearInterval(this.visionTimer);
    const ms = this.cfg().vision.intervalMs || 250;
    this.visionTimer = setInterval(() => void this.visionTick(), ms);
  }

  private cfg(): TrackerConfig {
    return this.upstream.getConfig().tracker;
  }

  // --- events from upstream / UI ---

  onSnapshot(now: number, aircraft: Aircraft[]): void {
    for (const ac of aircraft) this.history.observe(ac, now);
    this.history.prune(now);
    this.recorder.write("snapshot", { aircraft });
  }

  onConfig(config: Config): void {
    this.rescheduleTick();
    const t = config.tracker;
    const d = this.driver();
    if (d.kind !== t.driver) this.swapDriver(t.driver);
    // Re-evaluate the idle park: a changed home az/el should move the camera
    // now, not after the next pass.
    this.atHome = false;
  }

  setMode(mode: TrackerMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.manualHex = null;
    this.resetSetpoint();
    this.driver().stopMotion();
    this.recorder.write("mode", { mode });
  }

  getMode(): TrackerMode {
    return this.mode;
  }

  setTargetMode(mode: TargetMode): void {
    this.upstream.patchConfig({ tracker: { targetMode: mode } } as Partial<Config>);
    this.current = null;
    this.resetSetpoint();
  }

  manualTarget(hex: string | null): void {
    this.manualHex = hex;
    this.current = hex ? { hex, sinceMs: Date.now() } : null;
    this.resetSetpoint();
    this.recorder.write("manualTarget", { hex });
  }

  jog(pan: number, tilt: number, zoom: number): void {
    if (this.mode === "auto") this.setMode("manual");
    this.driver().jog(pan, tilt, zoom);
    this.recorder.write("jog", { pan, tilt, zoom });
  }

  stopJog(): void {
    this.driver().stopMotion();
  }

  /** Point at a world direction through the mount model (calibration verify). */
  gotoAzEl(azDeg: number, elDeg: number): void {
    if (this.mode === "auto") this.setMode("manual");
    const cfg = this.cfg();
    const pt = mountFromWorld(azDeg, elDeg, cfg.mount);
    const zoom = this.driver().getPose()?.zoomUnits ?? cfg.units.zoomWideUnits;
    const pose = { ...pt, zoomUnits: zoom };
    this.lastCommanded = pose;
    this.driver().gotoAbsolute(pose);
    this.recorder.write("gotoAzEl", { azDeg, elDeg, pose });
  }

  /** Raw mechanical move (units-per-degree measurement). */
  gotoPanTilt(panDeg: number, tiltDeg: number, zoomUnits?: number): void {
    if (this.mode === "auto") this.setMode("manual");
    const pose = {
      panDeg,
      tiltDeg,
      zoomUnits: zoomUnits ?? this.driver().getPose()?.zoomUnits ?? 0,
    };
    this.lastCommanded = pose;
    this.driver().gotoAbsolute(pose);
    this.recorder.write("gotoPanTilt", { pose });
  }

  calibCapture(ref: CalibrationRef): void {
    const pose = this.driver().getPose();
    if (!pose) return;
    const cfg = this.cfg();
    const cap = this.calibration.capture(
      ref,
      pose,
      cfg.site,
      (hex) => this.upstream.find(hex),
      Date.now(),
    );
    this.recorder.write("calibCapture", { ref, pose, cap });
  }

  calibSolve(solveGains: boolean, solveLevel: boolean): void {
    const model = this.calibration.solve(this.cfg().mount, solveGains, solveLevel);
    this.recorder.write("calibSolve", { model });
  }

  calibApply(): void {
    const model = this.calibration.takeSolved();
    if (!model) return;
    this.upstream.patchConfig({ tracker: { mount: model } } as Partial<Config>);
    this.recorder.write("calibApply", { model });
  }

  // --- the tick ---

  private resetSetpoint(): void {
    this.azTracker.reset();
    this.elTracker.reset();
    this.lastObservedTs = 0;
    // Vision state is target-specific — a stale lock or correction from the
    // previous plane must not smear onto the new one.
    this.lastDetection = null;
    this.tracks.reset();
    this.prevLuma = null;
    this.prevAim = null;
    this.corrAz = 0;
    this.corrEl = 0;
    this.corrAzApp = 0;
    this.corrElApp = 0;
    this.visionGoodSince = 0;
    this.ladderIdx = 0;
    this.lastLadderAt = 0;
    this.lastFocusZoom = 0;
    this.lastFocusAt = 0;
    this.focusCheckAt = 0;
    this.panRateI = 0;
    this.tiltRateI = 0;
    this.lastPanRateCmd = 0;
    this.lastTiltRateCmd = 0;
    this.errPanLP = 0;
    this.errTiltLP = 0;
  }

  private tick(): void {
    const now = Date.now();
    const cfg = this.cfg();
    const d = this.driver();
    const pose = d.getPose();
    // Dead-reckoned current pose: position replies stall around drives, so
    // the raw pose can lag by seconds mid-pursuit — closing the loop on it
    // injects phase lag that reads as trailing/hunting. All error terms and
    // slew estimates use this; the raw pose is kept for state/recording.
    const poseEst = d.getPoseEstimate() ?? pose;

    let prediction: Prediction | null = null;
    let targetAc: Aircraft | undefined;
    let commandedPanTilt: PanTilt | null = null;
    let hfovDeg: number | null = null;
    let angular: number | null = null;

    // Target selection runs in auto mode, or pinned in manual-target mode.
    const tracking =
      this.mode === "auto" || (this.mode === "manual" && this.manualHex != null);

    const selection = selectTarget(
      this.upstream.getAircraft(),
      cfg.site,
      now,
      this.current,
      cfg.targetMode,
      cfg.target,
    );

    if (tracking) {
      const hex = this.manualHex ?? selection.hex;
      if (hex !== this.current?.hex) {
        this.current = hex ? { hex, sinceMs: now } : null;
        this.resetSetpoint();
        this.recorder.write("target", { hex });
      }
      targetAc = hex ? this.upstream.find(hex) : undefined;
      // A manually-pinned target may be outside the candidate filter — allow it.
      if (this.manualHex && !targetAc) this.current = null;
    }

    if (tracking && targetAc) {
      prediction = predictAim(
        targetAc,
        cfg.site,
        now,
        this.history,
        cfg.predict,
        cfg.mount,
        cfg.limits,
        poseEst ? { panDeg: poseEst.panDeg, tiltDeg: poseEst.tiltDeg } : null,
      );
    }

    if (prediction && targetAc) {
      // Trajectory plan: detects near-zenith passes where chasing azimuth
      // would exceed the pan rate — pre-rotates to the exit side instead.
      const plan = planPass(targetAc, cfg.site, now);
      const hold = zenithHold(plan, prediction.azEl.elDeg);

      let az: number;
      let el: number;
      if (hold) {
        // Park on the exit azimuth, tilt just under vertical, fly-through.
        az = hold.azDeg;
        el = hold.elDeg;
        this.azTracker.reset(az);
        this.elTracker.reset(el);
        this.lastObservedTs = 0; // re-seed the filters on regime exit
        // The crossing reverses the needed pan rate — an inbound-learned
        // rate deficit is wrong-signed outbound.
        this.panRateI = 0;
        this.tiltRateI = 0;
      } else {
        // Feed the alpha-beta trackers on each fresh fix.
        const fixTs = targetAc.ts ?? now;
        if (fixTs !== this.lastObservedTs) {
          const dtFix = this.lastObservedTs ? (fixTs - this.lastObservedTs) / 1000 : 1;
          this.azTracker.observe(prediction.azEl.azDeg, dtFix, prediction.azRateDps);
          this.elTracker.observe(prediction.azEl.elDeg, dtFix, prediction.elRateDps);
          this.lastObservedTs = fixTs;
        }
        const dt = 1 / (cfg.predict.commandHz || 15);
        az = this.azTracker.propagate(dt, cfg.limits.panSpeedMaxDps);
        el = this.elTracker.propagate(dt, cfg.limits.tiltSpeedMaxDps);
        // Vision steering: the bias estimator's output, applied through a
        // rate limiter — the correction GLIDES onto the aim instead of
        // stepping per detection (4 Hz steps read as jank at tele).
        if (cfg.vision.applyCorrection) {
          const step = (cfg.vision.correctionSlewDps || 1.2) * dt;
          this.corrAzApp += clamp(norm180(this.corrAz - this.corrAzApp), -step, step);
          this.corrElApp += clamp(this.corrEl - this.corrElApp, -step, step);
          az += this.corrAzApp;
          el = Math.min(90, Math.max(0, el + this.corrElApp));
        }
      }

      const pt = mountFromWorld(az, el, cfg.mount);
      // Target's angular rate across the sky (for the zoom-out floor).
      const rateDps = Math.hypot(
        this.azTracker.rate * Math.cos((el * Math.PI) / 180),
        this.elTracker.rate,
      );
      // How far the camera is trailing its command right now, deg.
      const lagDeg = poseEst
        ? Math.hypot(
            norm180(poseEst.panDeg - pt.panDeg) * Math.cos((el * Math.PI) / 180),
            poseEst.tiltDeg - pt.tiltDeg,
          )
        : 0;
      // Vision is closing the loop when it has held the plane centered for a
      // while — then the pointing uncertainty is the tiny detector residual,
      // which lets the zoom tighten past the open-loop ~5× floor.
      const visionLocked =
        cfg.vision.applyCorrection &&
        this.visionGoodSince > 0 &&
        now - this.visionGoodSince > 1500;
      const zoom = chooseZoom(
        targetAc, prediction.azEl, cfg, rateDps, lagDeg,
        visionLocked ? cfg.zoom.lockedSigmaDeg : undefined,
      );
      if (hold && zoom.hfovDeg < ZENITH_MIN_HFOV) {
        // Wide through the crossing so the fly-through stays framed.
        zoom.hfovDeg = ZENITH_MIN_HFOV;
        zoom.zoomUnits = zoomUnitsFromHfov(ZENITH_MIN_HFOV, cfg.zoom.fovLut);
      }
      if (cfg.vision.lockWide) {
        // Vision-gated zoom LADDER: start fully wide; each time the detector
        // holds the plane near center for a while, step one rung tighter
        // (toward the ADS-B-chosen framing). Step back out when the lock
        // degrades; snap to wide when it's lost. Stepping (instead of the
        // old wide<->tight jump) keeps the plane in frame through every
        // transition — a rung is only granted after the PREVIOUS rung held.
        const wideHfov = hfovFromZoomUnits(cfg.units.zoomWideUnits, cfg.zoom.fovLut);
        const RUNG = 0.62; // hfov ratio per rung
        const detFresh =
          this.lastDetection && now - this.lastDetection.t < 1200;
        const locked = visionLocked;
        if (!detFresh && this.lastDetection == null) {
          this.ladderIdx = 0; // lost entirely -> reacquire wide
        } else if (locked && detFresh && now - this.lastLadderAt > 1500) {
          this.ladderIdx++;
          this.lastLadderAt = now;
          this.visionGoodSince = 0; // earn the next rung from scratch
        } else if (
          this.ladderIdx > 0 &&
          this.visionGoodSince === 0 &&
          now - this.lastLadderAt > 2000
        ) {
          this.ladderIdx--; // lock degrading -> widen one rung
          this.lastLadderAt = now;
        }
        const ladderHfov = wideHfov * Math.pow(RUNG, this.ladderIdx);
        if (ladderHfov > zoom.hfovDeg) {
          zoom.hfovDeg = ladderHfov;
          zoom.zoomUnits = zoomUnitsFromHfov(ladderHfov, cfg.zoom.fovLut);
        } else {
          // Ladder has reached the ADS-B framing — clamp the rung so a later
          // demotion starts from here, not from rungs banked beyond it.
          this.ladderIdx = Math.max(
            0,
            Math.round(Math.log(zoom.hfovDeg / wideHfov) / Math.log(RUNG)),
          );
        }
      }
      hfovDeg = zoom.hfovDeg;
      angular = zoom.angularSizeDeg;
      commandedPanTilt = pt;

      // Focus strategy:
      //  - Zoomed in AND vision-locked on the plane: one-push autofocus ON
      //    THE PLANE. The lens is not parfocal, so the fixed infinity far-stop
      //    softens at high zoom; focusing on the actual subject is the only way
      //    to stay sharp toward 20×. BUT fire the sweep only once the framing
      //    has SETTLED (the zoom ladder isn't still stepping) and the plane is
      //    sitting on the centre AF zone with the camera keeping up — a sweep
      //    fired mid-ladder or with the plane off-centre makes contrast-AF lock
      //    the blank sky / a passing cloud and rack the lens (the "can't find
      //    focus at high zoom" hunt). After a sweep, if the plane is no longer
      //    being detected the AF defocused it, so fall back to the infinity
      //    far-stop: stable and only mildly soft, never hunting. Re-arm on the
      //    next clean, centred frame.
      //  - Otherwise: infinity far-stop when meaningfully zoomed (a long lens
      //    on open sky makes autofocus hunt), autofocus at wide (hyperfocal
      //    depth covers everything and the plane is too small to lock).
      const zoomedIn = zoom.hfovDeg < 25;
      const fdet = this.lastDetection;
      const fFresh = fdet != null && now - fdet.t < 1000;
      const fCentered =
        fFresh && Math.abs(fdet.cx - 0.5) < 0.18 && Math.abs(fdet.cy - 0.5) < 0.18;
      const fZoomSettled = now - this.lastLadderAt > 1200; // ladder not stepping
      const fSteady = lagDeg < Math.max(0.4 * hfovDeg, 0.5); // camera keeping up
      if (zoomedIn && visionLocked && cfg.vision.autofocusOnZoom) {
        const zoomChanged = Math.abs(zoom.zoomUnits - this.lastFocusZoom) > 120;
        const wantFocus = this.lastFocusZoom <= 0 || zoomChanged;
        if (
          wantFocus && fCentered && fSteady && fZoomSettled &&
          now - this.lastFocusAt > 2000
        ) {
          d.onePushAutofocus();
          this.lastFocusZoom = zoom.zoomUnits;
          this.lastFocusAt = now;
          this.focusCheckAt = now;
        } else if (this.focusCheckAt > 0 && now - this.focusCheckAt > 2500) {
          // The sweep has had time to settle; if the plane dropped out of
          // detection the AF lost it — park at infinity and re-arm.
          if (!fFresh) {
            d.setFocusInfinity(true);
            this.lastFocusZoom = -1;
          }
          this.focusCheckAt = 0;
        }
      } else {
        d.setFocusInfinity(zoomedIn);
        this.lastFocusZoom = 0; // re-arm one-push AF for the next lock
        this.focusCheckAt = 0;
      }

      const cmd: CameraPose = { ...pt, zoomUnits: zoom.zoomUnits };
      this.lastCommanded = cmd;

      // Two regimes:
      //  - ACQUIRE: far off target -> one absolute move at full speed.
      //  - PURSUE: near target -> continuous velocity drive (feedforward
      //    setpoint rate + P-correction). Streamed absolute moves each run
      //    at max speed and hard-stop, which is what made motion choppy.
      const errPan = poseEst ? norm180(pt.panDeg - poseEst.panDeg) : Infinity;
      const errTilt = poseEst ? pt.tiltDeg - poseEst.tiltDeg : Infinity;
      const farOff = Math.max(Math.abs(errPan), Math.abs(errTilt)) > 10;
      // Shortest-path velocity drive can pin against the ±175° stop when the
      // goal lies across the pan dead-zone — those slews (and the zenith
      // park) genuinely need an absolute move, which the camera routes
      // within its encoder range. The exact test: if pose+err does NOT land
      // on the goal, the shortest path wraps through the zone (observed:
      // pose −175°, goal −140° "via" −214° → pinned 35° behind for a full
      // minute). Everything else, including catch-up after falling behind,
      // goes through the velocity drive: its P-term saturates at the top
      // speed step with NO accel-ramp restarts (streamed absolutes re-ramp
      // on every re-issue and crawl — observed ballooning 9°→105° of error
      // right at a pass peak).
      const pathBlocked =
        poseEst != null && Math.abs(poseEst.panDeg + errPan - pt.panDeg) > 1;
      const wrapAround = Math.abs(errPan) > 90;
      const useAbsolute =
        !pose ||
        hold ||
        pathBlocked ||
        (farOff && (wrapAround || cfg.predict.pursuit !== "velocity"));

      if (useAbsolute) {
        if (
          !this.lastAbsolute ||
          Math.abs(norm180(cmd.panDeg - this.lastAbsolute.panDeg)) > 1 ||
          Math.abs(cmd.tiltDeg - this.lastAbsolute.tiltDeg) > 1
        ) {
          this.lastAbsolute = cmd;
          d.gotoAbsolute(cmd);
          this.recorder.write("command", { cmd, predicted: prediction.azEl, mode: "acquire" });
        }
      } else if (cfg.predict.pursuit === "velocity") {
        this.lastAbsolute = null;
        // Setpoint rates live in world az/el; mount rates differ by the gains.
        // PI on the pose error: P pulls toward the setpoint, I learns the
        // systematic rate deficit (dither accel transients) so the camera
        // CENTERS the plane instead of trailing it by a held error.
        // Gentle P: the dead-reckoned pose is noisy (inquiry stalls snap it,
        // observed errPan rms ~8° while vision held the plane centered), so a
        // strong P just injects that measurement noise as rate jumps. Lean on
        // the (smooth) feedforward rate + vision for centering; P only trims.
        const KP = 0.85; // 1/s
        const KI = 0.6; // 1/s² — converges in ~2-3 s, anti-windup below
        const dtTick = 1 / (cfg.predict.commandHz || 15);
        // Low-pass the pose error before the P term: the dead-reckoned pose
        // SNAPS when an inquiry reply lands after a stall, and raw P turns
        // each snap into a velocity step (wobble). The filter rides through
        // the snap; real tracking error changes slowly enough to pass.
        const eS = clamp(cfg.predict.errSmoothing ?? 0, 0, 0.95);
        this.errPanLP += (1 - eS) * (errPan - this.errPanLP);
        this.errTiltLP += (1 - eS) * (errTilt - this.errTiltLP);
        // Integrate the (raw) error so steady bias still converges out.
        this.panRateI = clamp(this.panRateI + KI * errPan * dtTick, -6, 6);
        this.tiltRateI = clamp(this.tiltRateI + KI * errTilt * dtTick, -6, 6);
        let panRate =
          this.azTracker.rate / cfg.mount.panGain + KP * this.errPanLP + this.panRateI;
        let tiltRate =
          this.elTracker.rate / cfg.mount.tiltGain + KP * this.errTiltLP + this.tiltRateI;
        // Deadband so the camera rests when locked on (frame-relative: what
        // counts as "centered" is a fraction of the field of view, not a
        // fixed angle). The driver dithers against stop below the table
        // floor, so slow rates ARE commandable — only near-still targets
        // should rest.
        const dead = Math.max(0.15, (hfovDeg ?? 56) * 0.015);
        if (Math.abs(errPan) < dead && Math.abs(panRate) < 0.4) panRate = 0;
        if (Math.abs(errTilt) < dead && Math.abs(tiltRate) < 0.4) tiltRate = 0;
        // Direction-flip hysteresis: micro sign-flips (vision-correction
        // jitter at tele) command alternating bursts of the camera's MINIMUM
        // step — visible left-right rocking. Rest instead of reversing until
        // the rate is clearly real.
        if (
          this.lastPanRateCmd !== 0 &&
          Math.sign(panRate) === -Math.sign(this.lastPanRateCmd) &&
          Math.abs(panRate) < 0.8
        ) {
          panRate = 0;
        }
        if (
          this.lastTiltRateCmd !== 0 &&
          Math.sign(tiltRate) === -Math.sign(this.lastTiltRateCmd) &&
          Math.abs(tiltRate) < 0.8
        ) {
          tiltRate = 0;
        }
        // Anti-windup: while resting (deadband/hysteresis), bleed the
        // integrator instead of accumulating against a motionless camera.
        if (panRate === 0) this.panRateI *= 0.9;
        if (tiltRate === 0) this.tiltRateI *= 0.9;
        // Jerk limit: cap how fast the COMMANDED velocity may change, turning
        // any residual step into a brief smooth ramp ("one smooth movement").
        // Resting (rate 0 from deadband) is exempt so it can still stop
        // promptly when truly centered.
        const accel = cfg.predict.maxAccelDps2 ?? 0;
        if (accel > 0) {
          const maxStep = accel * dtTick;
          if (panRate !== 0) {
            panRate = this.lastPanRateCmd +
              clamp(panRate - this.lastPanRateCmd, -maxStep, maxStep);
          }
          if (tiltRate !== 0) {
            tiltRate = this.lastTiltRateCmd +
              clamp(tiltRate - this.lastTiltRateCmd, -maxStep, maxStep);
          }
        }
        this.lastPanRateCmd = panRate;
        this.lastTiltRateCmd = tiltRate;
        d.trackRate(panRate, tiltRate);
        this.pursuing = panRate !== 0 || tiltRate !== 0;
        d.setZoom(zoom.zoomUnits);
        this.recorder.write("command", {
          panRate, tiltRate, errPan, errTilt, zoom: zoom.zoomUnits, mode: "pursue",
        });
      } else if (poseEst && now - this.lastCarrotAt >= cfg.predict.carrotMs) {
        // CARROT pursuit: command a goal ~horizon seconds ahead along the
        // smoothed track, at a speed matched to arrive exactly then — and
        // re-issue before arrival, so the camera glides without ever
        // decelerating or hunting between coarse speed steps.
        this.lastCarrotAt = now;
        this.lastAbsolute = null;
        this.pursuing = false;
        const h = cfg.predict.carrotHorizonSec;
        const azAhead = az + this.azTracker.rate * h;
        const elAhead = Math.min(90, Math.max(0, el + this.elTracker.rate * h));
        const ptAhead = mountFromWorld(azAhead, elAhead, cfg.mount);
        const panDps = Math.abs(norm180(ptAhead.panDeg - poseEst.panDeg)) / h;
        const tiltDps = Math.abs(ptAhead.tiltDeg - poseEst.tiltDeg) / h;
        d.gotoAbsolute(
          { ...ptAhead, zoomUnits: zoom.zoomUnits },
          { panDps, tiltDps },
        );
        this.recorder.write("command", {
          ptAhead, panDps, tiltDps, errPan, errTilt, mode: "carrot",
        });
      }
    }

    // Target vanished while a velocity drive was active -> stop the motors,
    // or the camera glides into its mechanical limit on the last command.
    if (!prediction && this.pursuing) {
      this.driver().trackRate(0, 0);
      this.pursuing = false;
    }

    // No target for a while in auto mode -> park at the ready position
    // (default: 15° tilt along the bearing toward SFO, full wide), so the
    // next departure climbs straight into frame.
    if (prediction) {
      this.lastTargetAt = now;
      this.atHome = false;
    } else if (
      this.mode === "auto" &&
      cfg.home.enabled &&
      !this.atHome &&
      now - this.lastTargetAt > cfg.home.afterSec * 1000
    ) {
      const azDeg =
        cfg.home.mode === "sfo"
          ? azElFromSite(cfg.site, { ...SFO_ARP, altM: cfg.site.altM }).azDeg
          : cfg.home.azDeg;
      const pt = mountFromWorld(azDeg, cfg.home.elDeg, cfg.mount);
      this.atHome = true;
      this.lastAbsolute = null;
      this.driver().gotoAbsolute({ ...pt, zoomUnits: cfg.units.zoomWideUnits });
      this.driver().setFocusInfinity(false); // wide + idle -> autofocus
      this.recorder.write("home", { azDeg, elDeg: cfg.home.elDeg });
    }

    // Aim/prediction history for frame-lag-compensated vision. Entries are
    // timestamped `now`, so the dead-reckoned pose is the right aim here.
    if (poseEst && prediction) {
      const aim = worldFromMount(poseEst, cfg.mount);
      this.aimHistory.push({
        t: now,
        aimAz: aim.azDeg,
        aimEl: aim.elDeg,
        predAz: prediction.azEl.azDeg,
        predEl: prediction.azEl.elDeg,
        truthAz: prediction.nowAzEl.azDeg,
        truthEl: prediction.nowAzEl.elDeg,
        rateAzDps: this.azTracker.rate,
        rateElDps: this.elTracker.rate,
        panDeg: poseEst.panDeg,
        tiltDeg: poseEst.tiltDeg,
        hfov: hfovFromZoomUnits(poseEst.zoomUnits, cfg.zoom.fovLut),
      });
      if (this.aimHistory.length > 150) this.aimHistory.shift();
    } else if (!prediction) {
      this.aimHistory.length = 0;
    }

    // Between passes: refit the mount from the pass's calibration samples.
    // trySolve only returns a model that clearly beats the current one; the
    // bias estimator's state is folded into the model and reset.
    if (
      !prediction &&
      cfg.vision.autoCalibrate &&
      this.autoCal.hasNewData &&
      now - this.lastAutoCalAt > 30_000
    ) {
      this.lastAutoCalAt = now;
      const out = this.autoCal.trySolve(cfg.mount, now);
      if (out) {
        this.upstream.patchConfig({ tracker: { mount: out.model } } as Partial<Config>);
        this.corrAz = 0;
        this.corrEl = 0;
        this.corrAzApp = 0;
        this.corrElApp = 0;
        this.recorder.write("autoCal", { ...out });
        console.log(
          `[tracker] auto-calibration applied: rms ${out.rmsBeforeDeg.toFixed(2)}° -> ` +
            `${out.rmsAfterDeg.toFixed(2)}° over ${out.n} samples` +
            (out.solvedGains ? " (gains)" : "") + (out.solvedLevel ? " (level)" : ""),
        );
      }
    }

    this.lastState = this.assembleState(
      now, cfg, pose, selection.candidates, targetAc, prediction,
      commandedPanTilt, hfovDeg, angular,
    );
    this.recorder.write("pose", { pose });
  }

  private assembleState(
    now: number,
    cfg: TrackerConfig,
    pose: CameraPose | null,
    candidates: TrackerState["candidates"],
    targetAc: Aircraft | undefined,
    prediction: Prediction | null,
    commandedPanTilt: PanTilt | null,
    hfovDeg: number | null,
    angularSizeDeg: number | null,
  ): TrackerState {
    return {
      now,
      mode: this.mode,
      targetMode: cfg.targetMode,
      driver: this.driver().diagnostics(),
      pose,
      commanded: this.lastCommanded,
      target: {
        hex: this.current?.hex ?? null,
        flight: targetAc?.flight,
        predicted: prediction?.azEl ?? null,
        commandedPanTilt,
        leadSec: prediction?.leadSec ?? 0,
        hfovDeg,
        angularSizeDeg,
      },
      candidates,
      calibration: this.calibration.state(cfg.site, now),
      recording: this.recorder.recording,
      videoRec: this.videoRecStatus(),
      video: {
        ...this.video.status(),
        mseRunning: this.mseStatus().running,
        mseGen: this.mseStatus().gen,
      },
      vision: {
        enabled: cfg.vision.enabled,
        applying: cfg.vision.applyCorrection,
        lockWide: cfg.vision.lockWide,
        detection: this.lastDetection
          ? {
              cx: this.lastDetection.cx,
              cy: this.lastDetection.cy,
              boxX: this.lastDetection.box.x,
              boxY: this.lastDetection.box.y,
              boxW: this.lastDetection.box.w,
              boxH: this.lastDetection.box.h,
              contrastSigma: this.lastDetection.contrastSigma,
              offAzDeg: this.lastDetection.offAzDeg,
              offElDeg: this.lastDetection.offElDeg,
              ageMs: now - this.lastDetection.t,
            }
          : null,
        correctionAzDeg: this.corrAzApp,
        correctionElDeg: this.corrElApp,
        tracks: this.tracks.all().map((t) => {
          const v = t.velocity();
          const p = t.latest();
          return {
            azDeg: p.azDeg,
            elDeg: p.elDeg,
            hits: t.hits,
            ageMs: now - t.createdAt,
            azRateDps: v.azRateDps,
            elRateDps: v.elRateDps,
            locked: this.tracks.lockedTrack()?.id === t.id,
          };
        }),
        autoCal: cfg.vision.autoCalibrate ? this.autoCal.status(cfg.mount) : undefined,
        net: undefined, // retired: motion-compensated detection replaced it
      },
      site: cfg.site,
      upstream: {
        connected: this.upstream.isConnected(),
        aircraftCount: this.upstream.getAircraft().length,
      },
    };
  }

  /** Aim/prediction history entry nearest a timestamp. */
  private aimAt(t: number): (typeof this.aimHistory)[number] | undefined {
    let best: (typeof this.aimHistory)[number] | undefined;
    for (const h of this.aimHistory) {
      if (!best || Math.abs(h.t - t) < Math.abs(best.t - t)) best = h;
    }
    return best;
  }

  /**
   * Vision pass (Phase C): track-before-detect. Every candidate blob is
   * converted to WORLD az/el at frame time and fed to a small track table;
   * the plane is whichever track MOVES like the ADS-B prediction says the
   * plane moves (clouds are world-static, noise is incoherent). The locked
   * track drives the bias estimator, the zoom gate, and auto-calibration.
   * Everything is referenced to FRAME TIME via per-frame arrival timestamps.
   */
  private async visionTick(): Promise<void> {
    const cfg = this.cfg();
    const tracking =
      (this.mode === "auto" || this.mode === "manual") && this.current !== null;
    if (!cfg.vision.enabled || !tracking) {
      this.lastDetection = null;
      this.tracks.reset();
      this.prevLuma = null;
      this.prevAim = null;
      this.corrAz = 0;
      this.corrEl = 0;
      return;
    }
    if (this.visionBusy) return;
    const frame = this.video.latestFrame();
    // Captured WITH the frame — by the time the detector finishes, a newer
    // frame may have arrived and overwritten the stream's timestamp.
    const frameArrivedAt = this.video.latestFrameAt();
    const pose = this.driver().getPose();
    if (!frame || !pose || !this.video.status().running) return;

    const hfov = hfovFromZoomUnits(pose.zoomUnits, cfg.zoom.fovLut);
    const vfov = hfov * (9 / 16);
    const aim = worldFromMount(pose, cfg.mount);
    const predicted = this.lastState?.target.predicted ?? null;

    // Frame-time reference: the camera and the plane both moved while this
    // frame crossed the RTSP/MJPEG pipeline. Arrival is timestamped exactly;
    // encodeLagMs covers the residual (exposure -> encode -> RTSP -> decode).
    const preT = Date.now();
    const frameT = (frameArrivedAt > 0 ? frameArrivedAt : preT) - cfg.vision.encodeLagMs;
    const sign = Math.sign(cfg.mount.panGain) || 1;
    const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

    // Where should the plane be in THIS frame? The locked track's WORLD
    // position advanced to frame time (world-frame stickiness — it moves
    // with both the camera and the plane by construction), else ADS-B.
    let exX = 0.5;
    let exY = 0.5;
    const lockedBefore = this.tracks.lockedTrack();
    const atPre = this.aimAt(frameT);
    if (lockedBefore && atPre) {
      const p = lockedBefore.positionAt(frameT);
      const cosE = Math.max(0.2, Math.cos((atPre.aimEl * Math.PI) / 180));
      exX = clamp01(0.5 + (norm180(p.azDeg - atPre.aimAz) * cosE * sign) / atPre.hfov);
      exY = clamp01(0.5 - (p.elDeg - atPre.aimEl) / (atPre.hfov * (9 / 16)));
    } else if (predicted) {
      const dAz =
        norm180(predicted.azDeg - aim.azDeg) *
        Math.cos((predicted.elDeg * Math.PI) / 180);
      exX = clamp01(0.5 + dAz / hfov);
      exY = clamp01(0.5 - (predicted.elDeg - aim.elDeg) / vfov);
    }

    this.visionBusy = true;
    try {
      // --- camera-motion-compensated detection ---
      // Decode this frame and diff it against the previous one AFTER cancelling
      // the camera's KNOWN motion: the world-static sky / cloud / haze / ground
      // clutter subtracts to noise, and the plane is the residual that moved
      // against it. No contrast machinery, no neural net — just a seeded shift
      // search, a subtract, a horizon mask, and a blob find. The world-velocity
      // tracker (below) still arbitrates which residual is the plane.
      const cur = await decodeLuma(frame);
      let cands: Detection[] = [];
      const prev = this.prevLuma;
      const prevAim = this.prevAim;
      if (prev && prevAim && prev.length === cur.length) {
        // Seed the background-shift search with the known aim delta between the
        // two frames (camera moves -> world shifts the opposite way in frame);
        // a tiny ±6 px refine absorbs pose lag / lens distortion.
        const cosE0 = Math.max(0.2, Math.cos((aim.elDeg * Math.PI) / 180));
        const seedDx = Math.round(
          ((-norm180(aim.azDeg - prevAim.azDeg) * cosE0 * sign) / Math.max(1, hfov)) * VISION_W,
        );
        const seedDy = Math.round(((aim.elDeg - prevAim.elDeg) / Math.max(1, vfov)) * VISION_H);
        const shift = estimateShift(prev, cur, VISION_W, VISION_H, seedDx, seedDy, 6);
        const resid = compensatedResidual(prev, cur, VISION_W, VISION_H, shift);
        // The plane is always above the horizon; drop the ground (known from
        // tilt) so rooftops/poles/trees never become false movers.
        maskBelowHorizon(resid, VISION_W, VISION_H, aim.elDeg, vfov, 2);
        const blobs = findMovingBlobs(resid, VISION_W, VISION_H, 5, {
          expectedX: exX,
          expectedY: exY,
          maxDistFrac: lockedBefore ? 0.22 : 0.45,
          minPeakSigma: 5,
        });
        cands = blobs.map((b) => {
          const bw = Math.max(0.012, Math.sqrt(Math.max(1, b.areaPx)) / VISION_W);
          return {
            cx: b.cx,
            cy: b.cy,
            areaPx: b.areaPx,
            contrastSigma: b.peakSigma,
            box: { x: b.cx - bw / 2, y: b.cy - bw / 2, w: bw, h: bw },
            score: b.peakSigma,
          };
        });
      }
      // Roll the frame buffer forward for the next tick's diff.
      this.prevLuma = cur;
      this.prevAim = { azDeg: aim.azDeg, elDeg: aim.elDeg, hfov };

      const now = Date.now();
      const atFrame = this.aimAt(frameT);
      const latest = this.aimHistory[this.aimHistory.length - 1];
      if (!atFrame || !latest) return; // need history to place obs in the world

      // Aim slew rate around frame time: fast sweeps can't be lag-compensated
      // precisely (±150 ms of frame-time error at 60°/s = ±9° of phantom
      // offset), so association gates widen and integration pauses.
      const nb = this.aimHistory.filter((h) => Math.abs(h.t - frameT) < 250);
      let aimRateDps = Infinity;
      if (nb.length >= 2) {
        const a0 = nb[0];
        const a1 = nb[nb.length - 1];
        const dt = Math.max(0.05, (a1.t - a0.t) / 1000);
        aimRateDps = Math.hypot(
          norm180(a1.aimAz - a0.aimAz) * Math.cos((a1.aimEl * Math.PI) / 180),
          a1.aimEl - a0.aimEl,
        ) / dt;
      }

      // ALL candidates become world observations at frame time.
      const hfovF = atFrame.hfov;
      const elF = atFrame.aimEl;
      const cosE = Math.max(0.2, Math.cos((elF * Math.PI) / 180));
      const toWorld = (cx: number, cy: number) => ({
        azDeg: atFrame.aimAz + ((cx - 0.5) * hfovF * sign) / cosE,
        elDeg: atFrame.aimEl + (0.5 - cy) * hfovF * (9 / 16),
      });
      const obs: WorldObs[] = cands.map((c) => {
        const w = toWorld(c.cx, c.cy);
        return {
          t: frameT, azDeg: w.azDeg, elDeg: w.elDeg,
          cx: c.cx, cy: c.cy, box: c.box,
          contrastSigma: c.contrastSigma, areaPx: c.areaPx,
        };
      });

      // Associate into tracks; pick the lock by MOTION against the
      // prediction. Clouds lose on velocity even when brighter and closer.
      const gateDeg =
        Math.max(0.4, hfovF * 0.07) +
        (Number.isFinite(aimRateDps) ? 0.18 * Math.min(20, aimRateDps) : 1.5);
      this.tracks.update(obs, frameT, gateDeg);
      const sel = this.tracks.select(
        {
          azDeg: atFrame.predAz,
          elDeg: atFrame.predEl,
          azRateDps: atFrame.rateAzDps,
          elRateDps: atFrame.rateElDps,
        },
        frameT,
        { posScaleDeg: Math.max(1.6, hfovF * 0.05) },
      );
      const lt = sel?.track ?? null;
      const freshLock = lt != null && lt.lastSeen === frameT; // updated THIS frame

      if (lt && freshLock) {
        const o = lt.latest();
        // True-azimuth offset of the blob from the aim (matches the offAzDeg
        // convention everywhere else: an azimuth delta, not an arc length).
        const offAz = norm180(o.azDeg - atFrame.aimAz);
        const offElDeg = o.elDeg - atFrame.aimEl;
        this.lastDetection = {
          cx: o.cx,
          cy: o.cy,
          areaPx: o.areaPx,
          contrastSigma: o.contrastSigma,
          box: o.box,
          score: sel?.score ?? 0,
          t: now,
          frameT,
          offAzDeg: offAz,
          offElDeg,
        };

        const steady = aimRateDps < 8 && elF < 75;
        // The lock IS the temporal confirmation now: a track only becomes
        // the lock after several coherent, motion-consistent frames.
        const confirmed = lt.hits >= 4;

        if (confirmed && cfg.vision.applyCorrection) {
          // Where the blob (the plane, then) is NOW, world frame:
          //   blob@frame + plane's own predicted motion since the frame.
          const blobNowAz =
            atFrame.aimAz + offAz + norm180(latest.predAz - atFrame.predAz);
          const blobNowEl =
            atFrame.aimEl + offElDeg + (latest.predEl - atFrame.predEl);
          // Residual still to close, vs the CURRENT aim (zoom gate below).
          const residAz = norm180(blobNowAz - latest.aimAz);
          const residEl = blobNowEl - latest.aimEl;
          // Correction = the ADS-B bias the detector reveals: where vision
          // says the plane is MINUS where the prediction says it is. Direct
          // estimate of an exogenous offset — no feedback path through the
          // camera's ~1 s delay, so it cannot wind up or oscillate. Only
          // integrates when quasi-static (slew gate); the ZOOM gate keeps
          // updating regardless or the ladder would demote on a perfectly-
          // held lock just because the camera is sweeping.
          if (steady) {
            const biasAz = norm180(blobNowAz - latest.predAz);
            const biasEl = blobNowEl - latest.predEl;
            const G = hfovF < 15 ? 0.15 : 0.35;
            this.corrAz = clamp(this.corrAz + G * norm180(biasAz - this.corrAz), -6, 6);
            this.corrEl = clamp(this.corrEl + G * (biasEl - this.corrEl), -6, 6);
          }
          // Zoom gate: "near center" judged on the lag-compensated residual.
          const residFrac =
            Math.hypot(residAz * cosE, residEl) / (hfovF / 2);
          if (residFrac < 0.35) {
            if (!this.visionGoodSince) this.visionGoodSince = now;
          } else if (residFrac > 0.6) {
            this.visionGoodSince = 0;
          }

          // AUTO-CALIBRATION: a steady, well-established lock pairs the
          // mechanical pose that would CENTER the plane with the plane's
          // true direction (no aim lead) — a free wizard-grade sample.
          // Geometric altitude only; baro fallback is hundreds of feet off.
          if (cfg.vision.autoCalibrate && steady && lt.hits >= 6 && elF >= 8 && elF < 72) {
            const ac = this.current ? this.upstream.find(this.current.hex) : undefined;
            if (ac && ac.altGeom != null && (ac.gs ?? 0) > 80 && (ac.seen ?? 99) < 3) {
              this.autoCal.add({
                panDeg: atFrame.panDeg + offAz / cfg.mount.panGain,
                tiltDeg: atFrame.tiltDeg + offElDeg / cfg.mount.tiltGain,
                azDeg: atFrame.truthAz,
                elDeg: atFrame.truthEl,
                t: now,
              });
            }
          }
          this.recorder.write("vision", {
            det: o, offAzDeg: offAz, offElDeg, residAz, residEl, steady,
            lockScore: sel?.score, lockHits: lt.hits,
            corrAz: this.corrAz, corrEl: this.corrEl,
          });
        } else {
          this.recorder.write("vision", {
            det: o, offAzDeg: offAz, offElDeg, lockScore: sel?.score, lockHits: lt.hits,
          });
        }
      } else {
        // No fresh lock this frame: bleed the correction; the lock itself
        // survives short droughts via the track table's grace window.
        this.corrAz *= 0.85;
        this.corrEl *= 0.85;
        if (!lt) this.visionGoodSince = 0;
        if (this.lastDetection && now - this.lastDetection.t > 3000) {
          this.lastDetection = null;
        }
      }
    } catch (err) {
      this.recorder.write("visionError", { err: String(err) });
    } finally {
      this.visionBusy = false;
    }
  }

  getState(): TrackerState | null {
    return this.lastState;
  }

  /** Where the camera is actually looking in the world (for the UI overlay). */
  cameraWorldAim(): AzEl | null {
    const pose = this.driver().getPose();
    if (!pose) return null;
    const cfg = this.cfg();
    const w = worldFromMount(pose, cfg.mount);
    return { ...w, slantM: 0 };
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** SFO airport reference point (the project's anchor airport). */
const SFO_ARP = { lat: 37.6213, lon: -122.379 };
