import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { runMultiFrameTracker, buildRawToLogicalMap } from "./tracker.js";

/** HTTP API base: empty when page is opened as file:// (use npm start or local TSV). */
function getApiRoot() {
  if (typeof window === "undefined") return "";
  if (window.location.protocol === "file:") return "";
  return window.location.origin;
}

function apiUrl(path) {
  const root = getApiRoot();
  if (!root) return "";
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${root}${p}`;
}

/** Python http.server / static hosts return HTML for unknown paths; POST often returns 501. */
function responseLooksLikeStaticServerHtml(text) {
  const s = String(text).trimStart();
  return s.startsWith("<!DOCTYPE") || s.startsWith("<!doctype") || s.startsWith("<html");
}

const WRONG_SERVER_MSG =
  "Use the Node app: in the project folder run npm start, then open http://127.0.0.1:8765/ — not Python http.server or Live Server.";

const RESTART_HINT =
  "In the project folder run: npm run restart (or npm start). If something else was on this port (e.g. python -m http.server), stop it first. Then hard-refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R.";

/** @typedef {{ x: number, y: number, z: number }} Vec3 */
/** @typedef {{ frame: number, time: number, points: Vec3[] }} FrameRow */

/** Larger spheres (mm-ish scene units) for easier picking */
const SPHERE_RADIUS = 22;
const SPHERE_SEGMENTS = 16;

/** At 1× playback, target ~25 frames per wall-clock second (40 ms/frame). */
const BASE_PLAY_FPS = 25;
const PLAYBACK_TICK_MS = 16;

/** Top row Q–P → groups 1–10; home row A–L → 11–19 */
const LETTER_TO_GROUP = {
  q: 1,
  w: 2,
  e: 3,
  r: 4,
  t: 5,
  y: 6,
  u: 7,
  i: 8,
  o: 9,
  p: 10,
  a: 11,
  s: 12,
  d: 13,
  f: 14,
  g: 15,
  h: 16,
  j: 17,
  k: 18,
  l: 19,
};

/** @type {'label' | 'track' | 'live'} */
let activeTab = "label";

const trackState = {
  /** @type {object | null} */
  graph: null,
  /** @type {Awaited<ReturnType<typeof runMultiFrameTracker>> | null} */
  result: null,
  /** True while async tracker is still filling `result` (Play can show partial data). */
  trackingInProgress: false,
  playing: false,
  /** @type {ReturnType<typeof setInterval> | null} */
  playTimer: null,
  /** Fractional frame accumulator for smooth 1× and fast multipliers up to 500× */
  playAccum: 0,
};

const state = {
  /** @type {string | null} basename in data/, e.g. mar4qualisystrial1.tsv */
  sourceFileName: null,
  /** @type {FrameRow[]} */
  frames: [],
  baselineFrameIndex: 0,
  currentFrameIndex: 0,
  /** baseline point index → rigid group id (1-based) */
  markerGroupByIndex: new Map(),
  /** unique unordered pairs { groupA, groupB } with groupA < groupB */
  /** @type {{ groupA: number; groupB: number }[]} */
  segmentEdges: [],
  /** first rigid group chosen with "1" key; second press completes link */
  linkPendingGroup: null,
  /** @type {string | null} frameIndex:pointIndex */
  hoveredPointKey: null,
  cameraFitted: false,
  /** Raw TSV text when loaded via local file (so window limits can re-parse). */
  lastTsvText: null,
};

let scene, camera, renderer, controls, raycaster, pointer;
/** @type {THREE.Group} */
let pointsRoot;
/** @type {number | null} */
let hoverRaf = null;

function stripTrailingZeros(nums) {
  const out = [...nums];
  while (out.length >= 3 && out[out.length - 3] === 0 && out[out.length - 2] === 0 && out[out.length - 1] === 0) {
    out.length -= 3;
  }
  return out;
}

/**
 * @param {string} text
 * @param {number} maxSeconds
 * @param {number} maxFramesCap
 * @returns {FrameRow[]}
 */
function parseTsv(text, maxSeconds, maxFramesCap) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  /** @type {FrameRow[]} */
  const rows = [];
  for (const line of lines) {
    const parts = line.split("\t").map((p) => p.trim());
    if (parts.length < 3) continue;
    const frame = Number(parts[0]);
    const time = Number(parts[1]);
    if (!Number.isFinite(frame) || !Number.isFinite(time)) continue;
    if (Number.isFinite(maxSeconds) && time > maxSeconds) break;
    const rest = parts.slice(2).map(Number);
    const trimmed = stripTrailingZeros(rest);
    if (trimmed.length % 3 !== 0) continue;
    /** @type {Vec3[]} */
    const points = [];
    for (let i = 0; i < trimmed.length; i += 3) {
      points.push({ x: trimmed[i], y: trimmed[i + 1], z: trimmed[i + 2] });
    }
    rows.push({ frame, time, points });
    if (maxFramesCap > 0 && rows.length >= maxFramesCap) break;
  }
  return rows;
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function disposeNamedGroup(name) {
  const g = scene?.getObjectByName(name);
  if (!g) return;
  g.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
      else o.material.dispose();
    }
  });
  scene.remove(g);
}

/** @param {number} groupId */
function hexColorForRigidGroup(groupId) {
  if (groupId == null || groupId <= 0) return 0x455a64;
  const hue = ((groupId * 0.17) % 1 + 1) % 1;
  return new THREE.Color().setHSL(hue, 0.72, 0.52).getHex();
}

/**
 * @param {object} graph
 * @param {number} groupId
 * @returns {number[]}
 */
function baselineIndicesForRigidGroup(graph, groupId) {
  const mg = graph.markerRigidGroupByBaselineIndex;
  if (!mg) return [];
  const out = [];
  for (const [k, v] of Object.entries(mg)) {
    if (Number(v) === groupId) out.push(Number(k));
  }
  return out.sort((a, b) => a - b);
}

function stopTrackPlayback() {
  if (trackState.playTimer != null) {
    clearInterval(trackState.playTimer);
    trackState.playTimer = null;
  }
  trackState.playing = false;
  trackState.playAccum = 0;
  for (const id of ["btn-play-track", "btn-play-live"]) {
    const btn = document.getElementById(id);
    if (btn) btn.textContent = "Play";
  }
}

/** Start playback without requiring a finished tracker result (used with incremental tracking). */
function startPlayDuringTrackerRun() {
  if (state.frames.length < 2) return;
  if (trackState.playTimer != null) {
    clearInterval(trackState.playTimer);
    trackState.playTimer = null;
  }
  trackState.playing = true;
  trackState.playAccum = 0;
  for (const id of ["btn-play-track", "btn-play-live"]) {
    const btn = document.getElementById(id);
    if (btn) btn.textContent = "Pause";
  }
  trackState.playTimer = setInterval(playbackTick, PLAYBACK_TICK_MS);
}

function readReappearMm() {
  const order =
    activeTab === "live"
      ? ["track-reappear-live", "track-reappear"]
      : ["track-reappear", "track-reappear-live"];
  for (const id of order) {
    const v = Number(document.getElementById(id)?.value);
    if (Number.isFinite(v) && v > 0) return Math.min(500, Math.max(0.5, v));
  }
  return 50;
}

function readPlaybackSpeed() {
  const order =
    activeTab === "live"
      ? ["track-playback-speed-live", "track-playback-speed"]
      : ["track-playback-speed", "track-playback-speed-live"];
  for (const id of order) {
    const v = Number(document.getElementById(id)?.value);
    if (Number.isFinite(v) && v > 0) return Math.min(500, Math.max(0.25, v));
  }
  return 1;
}

/** @returns {number} */
function readToleranceMm() {
  const order =
    activeTab === "live"
      ? ["track-tolerance-live", "track-tolerance", "tolerance-mm"]
      : activeTab === "track"
        ? ["track-tolerance", "track-tolerance-live", "tolerance-mm"]
        : ["tolerance-mm", "track-tolerance", "track-tolerance-live"];
  for (const id of order) {
    const v = Number(document.getElementById(id)?.value);
    if (Number.isFinite(v) && v > 0) return Math.min(500, Math.max(0.1, v));
  }
  return 100;
}

/** @returns {number} */
function readFollowLookbackFrames() {
  const order =
    activeTab === "live"
      ? ["track-lookback-live", "track-lookback"]
      : activeTab === "track"
        ? ["track-lookback", "track-lookback-live"]
        : ["track-lookback", "track-lookback-live"];
  for (const id of order) {
    const v = Number(document.getElementById(id)?.value);
    if (Number.isFinite(v) && v >= 1) return Math.min(30, Math.max(1, Math.floor(v)));
  }
  return 10;
}

/** @returns {number | null} Empty field = no edge-error highlighting. */
function readEdgeWarningThresholdMm() {
  const order =
    activeTab === "live"
      ? ["track-edge-warn-mm-live", "track-edge-warn-mm"]
      : activeTab === "track"
        ? ["track-edge-warn-mm", "track-edge-warn-mm-live"]
        : ["track-edge-warn-mm", "track-edge-warn-mm-live"];
  for (const id of order) {
    const raw = document.getElementById(id)?.value?.trim();
    if (raw === "" || raw == null) return null;
    const v = Number(raw);
    if (Number.isFinite(v) && v > 0) return Math.min(500, Math.max(0.1, v));
  }
  return 150;
}

/**
 * Keep tolerance inputs aligned when a graph is loaded from disk.
 * @param {object} graph
 */
function applyToleranceInputsFromGraph(graph) {
  const t = graph?.toleranceMm;
  if (!Number.isFinite(Number(t)) || Number(t) <= 0) return;
  const s = String(Math.min(500, Math.max(0.1, Number(t))));
  for (const id of ["tolerance-mm", "track-tolerance", "track-tolerance-live"]) {
    const el = document.getElementById(id);
    if (el) el.value = s;
  }
}

function playbackTick() {
  if (!trackState.playing) return;
  const speed = readPlaybackSpeed();
  trackState.playAccum += (BASE_PLAY_FPS * speed * PLAYBACK_TICK_MS) / 1000;
  let guard = 0;
  while (trackState.playAccum >= 1 && guard++ < 15000) {
    trackState.playAccum -= 1;
    if (state.currentFrameIndex >= state.frames.length - 1) {
      state.currentFrameIndex = 0;
    } else {
      state.currentFrameIndex += 1;
    }
  }
  syncFrameSliders();
  syncUi();
}

function restartPlaybackInterval() {
  if (!trackState.playing) return;
  if (trackState.playTimer != null) {
    clearInterval(trackState.playTimer);
    trackState.playTimer = null;
  }
  trackState.playAccum = 0;
  trackState.playTimer = setInterval(playbackTick, PLAYBACK_TICK_MS);
}

/**
 * @param {HTMLCanvasElement | null} canvas
 * @param {number[]} values
 * @param {{ ymin: number; ymax: number; stroke: string; baselineIndex: number | null; flags?: boolean[] }} opts
 */
function paintTimelineChart(canvas, values, opts) {
  if (!canvas || values.length < 2) return;
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(320, Math.floor(rect.width) || 560);
  const cssH = Number(canvas.getAttribute("height")) || 120;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW;
  const H = cssH;
  ctx.fillStyle = "#12151a";
  ctx.fillRect(0, 0, W, H);
  const m = { l: 40, r: 8, t: 8, b: 20 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const { ymin, ymax, stroke, baselineIndex, flags } = opts;
  const span = Math.max(1e-6, ymax - ymin);
  const n = values.length;

  ctx.strokeStyle = "#2a2f38";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = m.t + ph * (1 - g / 4);
    ctx.beginPath();
    ctx.moveTo(m.l, y);
    ctx.lineTo(m.l + pw, y);
    ctx.stroke();
  }

  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = m.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const v = Math.min(ymax, Math.max(ymin, values[i]));
    const y = m.t + ph * (1 - (v - ymin) / span);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (flags && flags.length === n) {
    ctx.fillStyle = "rgba(255, 90, 90, 0.95)";
    for (let i = 0; i < n; i++) {
      if (!flags[i]) continue;
      const x = m.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
      const v = Math.min(ymax, Math.max(ymin, values[i]));
      const y = m.t + ph * (1 - (v - ymin) / span);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (baselineIndex != null && baselineIndex >= 0 && baselineIndex < n) {
    const x = m.l + (n === 1 ? pw / 2 : (baselineIndex / (n - 1)) * pw);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, m.t);
    ctx.lineTo(x, m.t + ph);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = "#9aa0a6";
  ctx.font = "10px system-ui,sans-serif";
  ctx.fillText(String(ymax.toFixed(ymax >= 100 ? 0 : 1)), 4, m.t + 10);
  ctx.fillText(String(ymin.toFixed(ymin === 0 ? 0 : 1)), 4, m.t + ph);
}

function clearTrackingReport() {
  const rep = document.getElementById("track-report");
  if (rep) {
    rep.classList.add("hidden");
    rep.hidden = true;
  }
  document.querySelector(".charts-rail-inner")?.classList.add("charts-rail-inner--empty");
  document.getElementById("track-per-frame-tbody")?.replaceChildren();
  const note = document.getElementById("track-report-table-note");
  if (note) note.textContent = "";
}

/** @param {{ frameStats: object[]; summary: object }} analytics */
function renderTrackingReport(analytics) {
  if (!analytics) {
    clearTrackingReport();
    return;
  }
  const rep = document.getElementById("track-report");
  if (rep) {
    rep.classList.remove("hidden");
    rep.hidden = false;
  }
  document.querySelector(".charts-rail-inner")?.classList.remove("charts-rail-inner--empty");

  const s = analytics.summary;
  const sm = document.getElementById("track-report-summary");
  if (sm) {
    const worstRow = s.worstFrameIndex != null ? s.worstFrameIndex + 1 : "—";
    const warn =
      s.edgeWarningThresholdMm != null
        ? `${s.edgeWarningThresholdMm} mm (orange or blue mean edge error)`
        : "off (set Phase 3 threshold)";
    sm.innerHTML = `<dl>
      <dt>Frames (total / tracked excl. baseline)</dt><dd>${s.framesTotal} / ${s.framesTracked}</dd>
      <dt>Logical markers</dt><dd>${s.logicalMarkers}</dd>
      <dt>Mean assignment rate</dt><dd>${s.meanAssignRatePct.toFixed(2)}%</dd>
      <dt>Min assignment rate</dt><dd>${s.minAssignRatePct.toFixed(2)}% (row ${worstRow}, file #${s.worstFileFrame ?? "—"})</dd>
      <dt>Frames with all logical assigned</dt><dd>${s.framesWithAllLogicalAssigned} / ${s.framesTracked}</dd>
      <dt>Σ logical misses</dt><dd>${s.totalLogicalMisses}</dd>
      <dt>Σ unassigned raw</dt><dd>${s.totalUnassignedRawDetections}</dd>
      <dt>Edge warn threshold</dt><dd>${warn}</dd>
      <dt>Frames flagged (edge quality)</dt><dd>${s.framesEdgeFlagged ?? 0} / ${s.framesTracked}</dd>
    </dl>`;
  }

  const fs = analytics.frameStats;
  const rates = fs.map((x) => x.rate);
  const raws = fs.map((x) => x.unassignedRaw);
  const misses = fs.map((x) => x.missing);
  const orangeErr = fs.map((x) => (x.orangeLenErr != null ? x.orangeLenErr : 0));
  const blueErr = fs.map((x) => (x.blueLenErr != null ? x.blueLenErr : 0));
  const edgeFlags = fs.map((x) => Boolean(x.edgeFlagged));
  const b = s.baselineFrameIndex;

  paintTimelineChart(document.getElementById("track-chart-rate"), rates, {
    ymin: 0,
    ymax: 100,
    stroke: "#4c8bf5",
    baselineIndex: b,
  });
  const maxRaw = Math.max(1, ...raws, 0);
  paintTimelineChart(document.getElementById("track-chart-raw"), raws, {
    ymin: 0,
    ymax: maxRaw * 1.08,
    stroke: "#f5a623",
    baselineIndex: b,
  });
  const maxMiss = Math.max(1, ...misses, 0);
  paintTimelineChart(document.getElementById("track-chart-miss"), misses, {
    ymin: 0,
    ymax: maxMiss * 1.08,
    stroke: "#e57373",
    baselineIndex: b,
  });
  const maxO = Math.max(0.1, ...orangeErr, 0);
  paintTimelineChart(document.getElementById("track-chart-orange-err"), orangeErr, {
    ymin: 0,
    ymax: maxO * 1.1,
    stroke: "#ff9800",
    baselineIndex: b,
    flags: edgeFlags,
  });
  const maxB = Math.max(0.1, ...blueErr, 0);
  paintTimelineChart(document.getElementById("track-chart-blue-err"), blueErr, {
    ymin: 0,
    ymax: maxB * 1.1,
    stroke: "#42a5f5",
    baselineIndex: b,
    flags: edgeFlags,
  });

  const tbody = document.getElementById("track-per-frame-tbody");
  const note = document.getElementById("track-report-table-note");
  const MAX_ROWS = 2500;
  const n = fs.length;
  const step = n <= MAX_ROWS ? 1 : Math.ceil(n / MAX_ROWS);
  if (tbody) {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < n; i += step) {
      const row = fs[i];
      const tr = document.createElement("tr");
      const base = row.isBaseline ? "Yes" : "";
      tr.innerHTML = `<td>${row.frameIndex + 1}</td><td>${row.fileFrame}</td><td>${row.timeSec.toFixed(3)}</td><td>${base}</td><td>${row.matched}</td><td>${row.missing}</td><td>${row.unassignedRaw}</td><td>${row.rate.toFixed(1)}</td>`;
      frag.appendChild(tr);
    }
    tbody.replaceChildren(frag);
  }
  if (note) {
    note.textContent =
      step > 1
        ? `Showing every ${step} frames (${Math.ceil(n / step)} rows) to keep the table responsive. Full data drives the charts above.`
        : `All ${n} frames listed.`;
  }
}

function syncFrameSliders() {
  const max = Math.max(0, state.frames.length - 1);
  const v = String(Math.min(state.currentFrameIndex, max));
  for (const id of ["frame-slider", "frame-slider-track", "frame-slider-live"]) {
    const s = document.getElementById(id);
    if (!s) continue;
    s.disabled = state.frames.length === 0;
    s.max = String(max);
    s.value = v;
  }
  const line = state.frames.length
    ? (() => {
        const r = state.frames[state.currentFrameIndex];
        return `Frame ${state.currentFrameIndex + 1}/${state.frames.length} (file #${r.frame}, t=${r.time.toFixed(3)}s)`;
      })()
    : "";
  const readoutTrack = document.getElementById("frame-readout-track");
  if (readoutTrack) readoutTrack.textContent = line || "No file loaded";
  const readoutLive = document.getElementById("frame-readout-live");
  if (readoutLive) readoutLive.textContent = line || "No file loaded";
}

function setActiveTab(tab) {
  activeTab = tab;
  const labelPanel = document.getElementById("panel-label");
  const trackPanel = document.getElementById("panel-track");
  const livePanel = document.getElementById("panel-live");
  const liveDock = document.getElementById("live-dock");
  document.querySelectorAll(".tab").forEach((b) => {
    const on = b.dataset.tab === tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  if (labelPanel) {
    labelPanel.classList.toggle("hidden", tab !== "label");
    labelPanel.hidden = tab !== "label";
  }
  if (trackPanel) {
    trackPanel.classList.toggle("hidden", tab !== "track");
    trackPanel.hidden = tab !== "track";
  }
  if (livePanel) {
    livePanel.classList.toggle("hidden", tab !== "live");
    livePanel.hidden = tab !== "live";
  }
  if (liveDock) {
    liveDock.classList.toggle("hidden", tab !== "live");
    liveDock.hidden = tab !== "live";
  }
  if (tab !== "track" && tab !== "live") stopTrackPlayback();

  if (tab === "live" && trackState.graph && !trackState.result && (state.sourceFileName || state.lastTsvText)) {
    void runTrackerAndVisualize();
    return;
  }
  syncUi();
}

async function fetchSkeletonForDataset() {
  trackState.graph = null;
  trackState.result = null;
  clearTrackingReport();
  const el = document.getElementById("skeleton-status");
  if (!state.sourceFileName) {
    if (el) el.textContent = "No dataset selected.";
    updateTrackButtons();
    return;
  }
  if (el) el.textContent = "Loading…";
  const base = apiUrl("/api/skeleton");
  if (!base) {
    if (el) el.textContent = "Skeleton API needs http:// (npm start). Load graph manually after labelling on server.";
    updateTrackButtons();
    return;
  }
  try {
    const res = await fetch(`${base}?name=${encodeURIComponent(state.sourceFileName)}`, { cache: "no-store" });
    const raw = await res.text();
    if (!res.ok || responseLooksLikeStaticServerHtml(raw)) {
      trackState.graph = null;
      if (el) el.textContent = WRONG_SERVER_MSG;
      updateTrackButtons();
      return;
    }
    let data = {};
    try {
      data = JSON.parse(raw);
    } catch {
      if (el) el.textContent = WRONG_SERVER_MSG;
      updateTrackButtons();
      return;
    }
    if (res.ok && data.schema) {
      trackState.graph = data;
      applyToleranceInputsFromGraph(data);
      if (el) el.textContent = `Loaded ${state.sourceFileName.replace(/\.[^.]+$/, "")}_labelled_skeleton/mocap-graph.json`;
    } else {
      trackState.graph = null;
      if (el)
        el.textContent =
          data.message || "No skeleton file yet — save from the Label tab, or check the folder name.";
    }
  } catch {
    trackState.graph = null;
    if (el) el.textContent = "Could not load skeleton (server running?)";
  }
  updateTrackButtons();
  if (activeTab === "live" && trackState.graph && !trackState.result && (state.sourceFileName || state.lastTsvText)) {
    await runTrackerAndVisualize();
  }
}

/**
 * Replace `state.frames` with the full TSV (no time or row cap) for tracking.
 * Uses cached local file text or re-fetches from `/api/dataset`. On fetch failure, keeps existing frames.
 * @returns {Promise<boolean>} true if there is at least one frame to track
 */
async function reloadFullTsvForTracking() {
  const noTimeCap = Number.POSITIVE_INFINITY;
  const noFrameCap = 0;
  if (state.lastTsvText) {
    state.frames = parseTsv(state.lastTsvText, noTimeCap, noFrameCap);
    return state.frames.length > 0;
  }
  const name = state.sourceFileName;
  const base = apiUrl("/api/dataset");
  if (!name || !base) return state.frames.length > 0;
  try {
    const res = await fetch(`${base}?name=${encodeURIComponent(name)}`, { cache: "no-store" });
    const text = await res.text();
    if (!res.ok || responseLooksLikeStaticServerHtml(text)) {
      setStatus("Could not reload full file for tracking — using frames already in memory.");
      return state.frames.length > 0;
    }
    state.frames = parseTsv(text, noTimeCap, noFrameCap);
    return state.frames.length > 0;
  } catch {
    setStatus("Network error loading full file for tracking — using frames already in memory.");
    return state.frames.length > 0;
  }
}

function updateTrackButtons() {
  const hasFrames = state.frames.length > 0;
  const hasGraph = trackState.graph != null;
  const hasDataset = Boolean(state.sourceFileName || state.lastTsvText);
  const canRunTracker = hasGraph && hasDataset;
  const bRun = document.getElementById("btn-run-tracker");
  const bPlay = document.getElementById("btn-play-track");
  const bRunLive = document.getElementById("btn-run-live");
  const bPlayLive = document.getElementById("btn-play-live");
  const bSk = document.getElementById("btn-reload-skeleton");
  if (bSk) bSk.disabled = !hasDataset || !state.sourceFileName;
  if (bRun) bRun.disabled = !canRunTracker;
  if (bPlay) bPlay.disabled = !hasFrames || (trackState.result == null && !trackState.trackingInProgress);
  if (bRunLive) bRunLive.disabled = !canRunTracker;
  if (bPlayLive) bPlayLive.disabled = !hasFrames || (trackState.result == null && !trackState.trackingInProgress);
}

async function runTrackerAndVisualize() {
  if (!trackState.graph) return;
  stopTrackPlayback();
  trackState.trackingInProgress = true;
  trackState.result = null;
  clearTrackingReport();
  updateTrackButtons();
  syncUi();

  setStatus("Loading full clip for tracker…");
  const ok = await reloadFullTsvForTracking();
  if (!ok || !state.frames.length) {
    trackState.trackingInProgress = false;
    setStatus("No frames to track. Load a dataset first.");
    updateTrackButtons();
    syncUi();
    return;
  }
  state.currentFrameIndex = Math.min(Math.max(0, state.currentFrameIndex), state.frames.length - 1);
  syncFrameSliders();
  const reappear = readReappearMm();
  const toleranceMm = readToleranceMm();
  const followLookbackFrames = readFollowLookbackFrames();
  const edgeWarningThresholdMm = readEdgeWarningThresholdMm();
  const b = Math.min(Math.max(0, trackState.graph.baselineFrameIndex ?? 0), state.frames.length - 1);
  if (state.frames[b].points.length === 0) {
    trackState.trackingInProgress = false;
    setStatus("Baseline frame has no points.");
    updateTrackButtons();
    syncUi();
    return;
  }

  state.currentFrameIndex = b;
  syncFrameSliders();
  setStatus("Tracking… playback is live; solver running in the background.");
  startPlayDuringTrackerRun();
  updateTrackButtons();

  try {
    trackState.result = await runMultiFrameTracker(state.frames, trackState.graph, {
      reappearMm: reappear,
      toleranceMm,
      followLookbackFrames,
      edgeWarningThresholdMm,
      yieldEvery: 2,
      onYield: async (r) => {
        trackState.result = r;
        await new Promise((res) => setTimeout(res, 0));
        syncUi();
      },
    });
  } catch (e) {
    console.error(e);
    setStatus("Tracker failed — see console.");
  } finally {
    trackState.trackingInProgress = false;
  }

  state.currentFrameIndex = b;
  syncFrameSliders();
  const st = document.getElementById("track-stats");
  if (st && trackState.result) {
    const s = trackState.result.stats;
    st.innerHTML = `Frames: ${s.frames} · Logical: ${s.logicalMarkers} · Last frame: matched ${s.lastFrameMatched}, missing ${s.lastFrameMissing} · <strong>Full run</strong>: all frames except baseline row ${b + 1} (forward + backward)`;
  }
  if (trackState.result?.analytics) {
    renderTrackingReport(trackState.result.analytics);
  }
  if (trackState.result) {
    setStatus(
      `Tracker finished on ${trackState.result.stats.frames} frames (full file; baseline row ${b + 1}). Playback still running — press Pause to stop.`
    );
  }
  updateTrackButtons();
  syncUi();
}

function rebuildTrackPoints(frameIndex) {
  disposeNamedGroup("rigidBodyEdges");
  disposeNamedGroup("segmentEdges");
  clearPoints();
  disposeNamedGroup("trackRigidEdges");
  disposeNamedGroup("trackSegmentEdges");

  const row = state.frames[frameIndex];
  if (!row || !trackState.result) return;

  const res = trackState.result;
  const rawToLogical = buildRawToLogicalMap(state.frames, res, frameIndex);

  row.points.forEach((p, j) => {
    const logical = rawToLogical.get(j);
    const g = logical != null ? res.groupByLogical.get(logical) : null;
    const col =
      logical != null ? hexColorForRigidGroup(g != null && g > 0 ? g : ((logical % 19) + 1)) : 0x263238;
    const mat = new THREE.MeshStandardMaterial({
      color: col,
      metalness: 0.12,
      roughness: 0.5,
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_SEGMENTS), mat);
    mesh.position.set(p.x, p.y, p.z);
    mesh.userData.pointKey = pointKey(frameIndex, j);
    mesh.userData.trackLogical = logical ?? null;
    pointsRoot.add(mesh);
  });

  if (!state.cameraFitted && row.points.length) {
    const c = centerOfPoints(row.points);
    camera.position.set(c.x + 900, c.y + 700, c.z + 900);
    controls.target.copy(c);
    controls.update();
    state.cameraFitted = true;
  } else if (state.cameraFitted && readCameraFollowEnabled() && row.points.length) {
    applyCentroidCameraFollow(row.points);
  }

  const graph = trackState.graph;
  if (!graph) return;

  const rigidGrp = new THREE.Group();
  rigidGrp.name = "trackRigidEdges";
  const segGrp = new THREE.Group();
  segGrp.name = "trackSegmentEdges";

  const pf = res.perFrame[frameIndex];
  const orange = [];
  const groupIds = new Set();
  const mg = graph.markerRigidGroupByBaselineIndex;
  if (mg) for (const v of Object.values(mg)) groupIds.add(Number(v));
  for (const gid of groupIds) {
    const idx = baselineIndicesForRigidGroup(graph, gid);
    for (let a = 0; a < idx.length; a++) {
      for (let b = a + 1; b < idx.length; b++) {
        const ia = idx[a];
        const ib = idx[b];
        const pa = pf.logicalPos[ia];
        const pb = pf.logicalPos[ib];
        if (!pa || !pb) continue;
        orange.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
      }
    }
  }
  flushLineBuffer(orange, 0xff9800, 0.82, rigidGrp);

  const segs = graph.segmentEdgesBetweenRigidBodies || [];
  const blue = [];
  for (const e of segs) {
    const ia = baselineIndicesForRigidGroup(graph, e.groupA);
    const ib = baselineIndicesForRigidGroup(graph, e.groupB);
    for (const i of ia) {
      for (const j of ib) {
        const pa = pf.logicalPos[i];
        const pb = pf.logicalPos[j];
        if (!pa || !pb) continue;
        blue.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
      }
    }
  }
  flushLineBuffer(blue, 0x42a5f5, 0.72, segGrp);

  scene.add(rigidGrp);
  scene.add(segGrp);
}

/**
 * @param {number} frameIndex
 */
function updateLiveAssignmentTable(frameIndex) {
  const tbody = document.getElementById("live-assignment-tbody");
  const unEl = document.getElementById("live-unassigned-block");
  const titleEl = document.getElementById("live-dock-title");
  const metaEl = document.getElementById("live-dock-meta");
  if (!tbody || !unEl) return;

  if (!trackState.result || !state.frames[frameIndex]) {
    tbody.innerHTML = "";
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "hint";
    td.style.padding = "12px 0";
    td.textContent = "Load a dataset and skeleton, then run the tracker (or open Live motion to auto-run).";
    tr.appendChild(td);
    tbody.appendChild(tr);
    unEl.textContent = "";
    if (titleEl) titleEl.textContent = "Assignments";
    if (metaEl) metaEl.textContent = "";
    return;
  }

  const res = trackState.result;
  const pf = res.perFrame[frameIndex];
  const row = state.frames[frameIndex];
  const hingeSet = new Set(res.hingeLogical ?? []);

  let assignedN = 0;
  let missingN = 0;
  const frag = document.createDocumentFragment();

  for (let i = 0; i < res.numLogical; i++) {
    const rb = res.groupByLogical.get(i);
    const raw = pf.rawForLogical[i];
    const ok = raw != null;
    if (ok) assignedN++;
    else missingN++;

    const tr = document.createElement("tr");
    const tdL = document.createElement("td");
    tdL.textContent = String(i);
    const tdR = document.createElement("td");
    tdR.textContent = rb != null ? String(rb) : "—";
    const tdH = document.createElement("td");
    tdH.textContent = hingeSet.has(i) ? "Yes" : "";
    const tdRaw = document.createElement("td");
    tdRaw.textContent = raw != null ? String(raw) : "—";
    const tdSt = document.createElement("td");
    tdSt.textContent = ok ? "Assigned" : "Missing";
    tdSt.className = ok ? "status-ok" : "status-miss";
    tr.append(tdL, tdR, tdH, tdRaw, tdSt);
    frag.appendChild(tr);
  }

  tbody.replaceChildren(frag);

  const usedRaw = new Set();
  for (let i = 0; i < res.numLogical; i++) {
    const r = pf.rawForLogical[i];
    if (r != null) usedRaw.add(r);
  }
  /** @type {number[]} */
  const unassigned = [];
  for (let j = 0; j < row.points.length; j++) {
    if (!usedRaw.has(j)) unassigned.push(j);
  }

  if (unassigned.length) {
    unEl.innerHTML = `<strong>Unassigned raw detections (${unassigned.length}):</strong> ${unassigned.join(", ")}`;
  } else {
    unEl.innerHTML = "<strong>Unassigned raw detections:</strong> none (every detection maps to a logical marker).";
  }

  if (titleEl) titleEl.textContent = "Assignments";
  if (metaEl) {
    const r = row;
    metaEl.textContent = `Frame ${frameIndex + 1}/${state.frames.length} · file #${r.frame} · logical assigned ${assignedN}/${res.numLogical} · unassigned raw ${unassigned.length}`;
  }
}

function baselineRow() {
  return state.frames[state.baselineFrameIndex] ?? state.frames[0];
}

function pointKey(frameIndex, pointIndex) {
  return `${frameIndex}:${pointIndex}`;
}

function centerOfPoints(points) {
  if (!points.length) return new THREE.Vector3();
  const s = new THREE.Vector3();
  for (const p of points) s.add(new THREE.Vector3(p.x, p.y, p.z));
  s.multiplyScalar(1 / points.length);
  return s;
}

/** Orbit target + camera move together so rotation/zoom are preserved while the cloud stays centered. */
function applyCentroidCameraFollow(points) {
  if (!camera || !controls) return;
  if (!points.length) return;
  const c = centerOfPoints(points);
  const delta = c.clone().sub(controls.target);
  if (delta.lengthSq() < 1e-12) return;
  camera.position.add(delta);
  controls.target.copy(c);
  controls.update();
}

function readCameraFollowEnabled() {
  const id = activeTab === "live" ? "camera-follow-live" : "camera-follow-track";
  const el = document.getElementById(id);
  if (el) return Boolean(el.checked);
  return false;
}

function initThree(container) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1114);

  camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 1e6);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enableKeys = false;

  scene.add(new THREE.AmbientLight(0xffffff, 0.88));
  const dir = new THREE.DirectionalLight(0xffffff, 0.45);
  dir.position.set(1, 2, 3);
  scene.add(dir);

  pointsRoot = new THREE.Group();
  scene.add(pointsRoot);

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2();

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  container.addEventListener("pointermove", onPointerMove);
  container.addEventListener("pointerdown", () => container.focus());

  function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();
}

function updatePointerFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onPointerMove(event) {
  if (!state.frames.length || activeTab !== "label") return;
  updatePointerFromEvent(event);
  if (hoverRaf != null) return;
  hoverRaf = requestAnimationFrame(() => {
    hoverRaf = null;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(pointsRoot.children, false);
    const key = hits.length && hits[0].object.userData.pointKey ? hits[0].object.userData.pointKey : null;
    if (key !== state.hoveredPointKey) {
      state.hoveredPointKey = key;
      rebuildPoints(state.currentFrameIndex);
      rebuildEdgeLines();
    }
  });
}

function clearPoints() {
  while (pointsRoot.children.length) {
    const o = pointsRoot.children[0];
    pointsRoot.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
    else if (o.material) o.material.dispose();
  }
}

function makeMarkerMaterial(isLabelled, isHover) {
  const color = isLabelled ? 0xf5f5f5 : 0xd32f2f;
  const mat = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.15,
    roughness: 0.55,
    emissive: isHover ? 0x2244aa : 0x000000,
    emissiveIntensity: isHover ? 0.35 : 0,
  });
  return mat;
}

/**
 * @param {number} frameIndex
 */
function rebuildPoints(frameIndex) {
  disposeNamedGroup("trackRigidEdges");
  disposeNamedGroup("trackSegmentEdges");
  clearPoints();
  const row = state.frames[frameIndex];
  if (!row) return;

  row.points.forEach((p, i) => {
    const key = pointKey(frameIndex, i);
    const group = state.markerGroupByIndex.get(i);
    const labelled = group != null && group > 0;
    const isHover = state.hoveredPointKey === key;
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_SEGMENTS),
      makeMarkerMaterial(labelled, isHover)
    );
    mesh.position.set(p.x, p.y, p.z);
    mesh.userData.pointKey = key;
    if (isHover) mesh.scale.setScalar(1.12);
    pointsRoot.add(mesh);
  });

  if (!state.cameraFitted && row.points.length) {
    const c = centerOfPoints(row.points);
    camera.position.set(c.x + 900, c.y + 700, c.z + 900);
    controls.target.copy(c);
    controls.update();
    state.cameraFitted = true;
  }
}

/**
 * @param {number} groupId
 * @returns {number[]}
 */
function baselineIndicesInGroup(groupId) {
  const base = baselineRow();
  if (!base) return [];
  const out = [];
  for (let i = 0; i < base.points.length; i++) {
    if (state.markerGroupByIndex.get(i) === groupId) out.push(i);
  }
  return out;
}

/**
 * @param {(a: Vec3, b: Vec3) => void} emit
 */
function forEachPairInGroup(groupId, emit) {
  const base = baselineRow();
  if (!base) return;
  const idx = baselineIndicesInGroup(groupId);
  for (let a = 0; a < idx.length; a++) {
    for (let b = a + 1; b < idx.length; b++) {
      emit(base.points[idx[a]], base.points[idx[b]]);
    }
  }
}

/**
 * @param {(a: Vec3, b: Vec3) => void} emit
 */
function forEachCrossGroupPair(groupA, groupB, emit) {
  const base = baselineRow();
  if (!base) return;
  const ia = baselineIndicesInGroup(groupA);
  const ib = baselineIndicesInGroup(groupB);
  for (const i of ia) {
    for (const j of ib) {
      emit(base.points[i], base.points[j]);
    }
  }
}

function flushLineBuffer(positions, color, opacity, parent) {
  if (positions.length < 6) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  parent.add(new THREE.LineSegments(geo, mat));
}

function rebuildEdgeLines() {
  if (activeTab !== "label") return;
  const oldRigid = scene.getObjectByName("rigidBodyEdges");
  const oldSeg = scene.getObjectByName("segmentEdges");
  for (const grp of [oldRigid, oldSeg]) {
    if (!grp) continue;
    grp.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
        else o.material.dispose();
      }
    });
    scene.remove(grp);
  }

  const base = baselineRow();
  if (!base) return;

  const rigidGroup = new THREE.Group();
  rigidGroup.name = "rigidBodyEdges";
  const segGroup = new THREE.Group();
  segGroup.name = "segmentEdges";

  const orangeBuf = [];
  const groupIds = new Set(state.markerGroupByIndex.values());
  for (const g of groupIds) {
    forEachPairInGroup(g, (pa, pb) => {
      orangeBuf.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
    });
  }
  flushLineBuffer(orangeBuf, 0xff9800, 0.85, rigidGroup);

  const blueBuf = [];
  for (const e of state.segmentEdges) {
    forEachCrossGroupPair(e.groupA, e.groupB, (pa, pb) => {
      blueBuf.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
    });
  }
  flushLineBuffer(blueBuf, 0x42a5f5, 0.75, segGroup);

  scene.add(rigidGroup);
  scene.add(segGroup);
}

function setStatus(msg) {
  const el = document.getElementById("status-bar");
  if (el) el.textContent = msg;
}

function normalizePair(ga, gb) {
  return ga < gb ? { groupA: ga, groupB: gb } : { groupA: gb, groupB: ga };
}

function segmentExists(ga, gb) {
  const n = normalizePair(ga, gb);
  return state.segmentEdges.some((e) => e.groupA === n.groupA && e.groupB === n.groupB);
}

function serializeProgressText() {
  if (!state.frames.length) return "";
  const r = state.frames[state.baselineFrameIndex];
  const lines = [`Frame ${r.frame}`];
  const ids = [...new Set(state.markerGroupByIndex.values())].sort((a, b) => a - b);
  for (const g of ids) {
    const idx = baselineIndicesInGroup(g);
    lines.push(`RB ${g}: ${idx.length} marker(s) — [${idx.join(", ")}]`);
  }
  lines.push("");
  const segs = [...state.segmentEdges].sort((a, b) => a.groupA - b.groupA || a.groupB - b.groupB);
  for (const e of segs) {
    lines.push(`RB ${e.groupA} ↔ RB ${e.groupB}`);
  }
  return lines.join("\n");
}

/**
 * @returns {{ ok: true, warnings?: string[] } | { ok: false, error: string }}
 */
function parseAndApplyProgressText(text) {
  /** @type {string[]} */
  const warnings = [];
  if (!state.frames.length) return { ok: false, error: "Load a dataset first." };
  const trimmed = text.replace(/^\uFEFF/, "").trim();
  if (!trimmed) return { ok: false, error: "No text to parse." };

  let fileFrame = /** @type {number | null} */ (null);
  /** @type {Map<number, number[]>} */
  const rbMap = new Map();
  /** @type {{ groupA: number; groupB: number }[]} */
  const segments = [];

  const lines = trimmed.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;

    const fm = line.match(/^Frame\s+(\d+)\s*$/i);
    if (fm) {
      fileFrame = Number(fm[1]);
      continue;
    }

    const rbHead = line.match(/^\s*RB\s+(\d+)\s*:\s*(.*)$/i);
    if (rbHead) {
      const g = Number(rbHead[1]);
      const rest = rbHead[2].trim();
      const rbBody = rest.match(/^(?:(\d+)\s+marker\(s\)\s*)?[—\u2013\u2014\-]\s*\[([\d,\s]*)\]\s*$/i);
      if (!rbBody) {
        return { ok: false, error: `Bad RB line (expected "RB n: k marker(s) — […]"): "${line.slice(0, 70)}…"` };
      }
      const declared = rbBody[1] != null ? Number(rbBody[1]) : null;
      const parts = rbBody[2]
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const indices = parts.map((p) => Number(p)).filter((n) => Number.isFinite(n));
      if (declared != null && declared !== indices.length) {
        warnings.push(`RB ${g}: header says ${declared} marker(s) but list has ${indices.length} (using list).`);
      }
      rbMap.set(g, indices);
      continue;
    }

    const segm = line.match(/^\s*RB\s+(\d+)\s*(?:\u2194|↔|<->)\s*RB\s+(\d+)\s*$/i);
    if (segm) {
      segments.push(normalizePair(Number(segm[1]), Number(segm[2])));
      continue;
    }

    const segm2 = line.match(/^\s*RB\s+(\d+)\s+-\s+RB\s+(\d+)\s*$/i);
    if (segm2) {
      segments.push(normalizePair(Number(segm2[1]), Number(segm2[2])));
      continue;
    }

    return {
      ok: false,
      error: `Unrecognized line: "${line.length > 70 ? `${line.slice(0, 70)}…` : line}"`,
    };
  }

  if (fileFrame == null) return { ok: false, error: 'Add a first line like: Frame 568' };

  const frameIdx = state.frames.findIndex((fr) => fr.frame === fileFrame);
  if (frameIdx < 0) {
    return {
      ok: false,
      error: `No loaded row with file frame ${fileFrame}. Increase "Load first seconds" / frames, then try again.`,
    };
  }

  /** @type {Map<number, number>} */
  const markerGroup = new Map();
  for (const [g, indices] of rbMap) {
    for (const idx of indices) {
      const prev = markerGroup.get(idx);
      if (prev != null && prev !== g) {
        return { ok: false, error: `Marker index ${idx} is listed under RB ${prev} and RB ${g}.` };
      }
      markerGroup.set(idx, g);
    }
  }

  const row = state.frames[frameIdx];
  for (const idx of markerGroup.keys()) {
    if (idx < 0 || idx >= row.points.length) {
      warnings.push(
        `Index ${idx} is outside this frame's point count (${row.points.length}); check baseline / dataset.`
      );
    }
  }

  state.baselineFrameIndex = frameIdx;
  state.currentFrameIndex = frameIdx;
  state.markerGroupByIndex = markerGroup;
  state.segmentEdges = [];
  const seenSeg = new Set();
  for (const e of segments) {
    const key = `${e.groupA},${e.groupB}`;
    if (seenSeg.has(key)) continue;
    seenSeg.add(key);
    state.segmentEdges.push(e);
  }
  state.linkPendingGroup = null;
  state.hoveredPointKey = null;
  state.cameraFitted = false;

  return warnings.length ? { ok: true, warnings } : { ok: true };
}

function syncSummary() {
  const gs = document.getElementById("group-summary");
  if (gs) {
    const ids = [...new Set(state.markerGroupByIndex.values())].sort((a, b) => a - b);
    if (!ids.length) gs.textContent = "No labels yet.";
    else {
      gs.innerHTML = ids
        .map((g) => {
          const idx = baselineIndicesInGroup(g);
          return `<strong>RB ${g}</strong>: ${idx.length} marker(s) — [${idx.join(", ")}]`;
        })
        .join("<br/>");
    }
  }

  const sl = document.getElementById("segment-list");
  if (sl) {
    sl.innerHTML = state.segmentEdges.length
      ? state.segmentEdges.map((e) => `RB ${e.groupA} ↔ RB ${e.groupB}`).join("<br/>")
      : "None.";
  }
}

function syncUi() {
  const has = state.frames.length > 0;
  const rf = document.getElementById("btn-refresh-datasets");
  if (rf) rf.disabled = !getApiRoot();
  syncFrameSliders();

  const readout = document.getElementById("frame-readout");
  if (readout && has) {
    const r = state.frames[state.currentFrameIndex];
    const pend =
      state.linkPendingGroup != null ? ` · pending link from RB ${state.linkPendingGroup}` : "";
    readout.textContent = `Frame ${state.currentFrameIndex + 1}/${state.frames.length} (file #${r.frame}, t=${r.time.toFixed(3)}s) · baseline ${state.baselineFrameIndex}${pend}`;
  }

  for (const id of [
    "btn-baseline",
    "btn-export",
    "btn-clear-pending",
    "btn-clear-labels",
    "btn-copy-progress",
    "btn-apply-progress",
  ]) {
    const b = document.getElementById(id);
    if (b) b.disabled = !has;
  }

  if (activeTab === "label") {
    syncSummary();
    rebuildPoints(state.currentFrameIndex);
    rebuildEdgeLines();
  } else if (activeTab === "track" || activeTab === "live") {
    if (trackState.result) {
      rebuildTrackPoints(state.currentFrameIndex);
      if (activeTab === "live") updateLiveAssignmentTable(state.currentFrameIndex);
    } else {
      disposeNamedGroup("rigidBodyEdges");
      disposeNamedGroup("segmentEdges");
      disposeNamedGroup("trackRigidEdges");
      disposeNamedGroup("trackSegmentEdges");
      clearPoints();
      const row = state.frames[state.currentFrameIndex];
      if (row?.points.length && !state.cameraFitted) {
        const c = centerOfPoints(row.points);
        camera.position.set(c.x + 900, c.y + 700, c.z + 900);
        controls.target.copy(c);
        controls.update();
        state.cameraFitted = true;
      } else if (row?.points.length && state.cameraFitted && readCameraFollowEnabled()) {
        applyCentroidCameraFollow(row.points);
      }
      if (row?.points.length) {
        row.points.forEach((p) => {
          const mat = new THREE.MeshStandardMaterial({ color: 0x37474f, metalness: 0.1, roughness: 0.55 });
          const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_SEGMENTS),
            mat
          );
          mesh.position.set(p.x, p.y, p.z);
          pointsRoot.add(mesh);
        });
      }
      if (activeTab === "live") updateLiveAssignmentTable(state.currentFrameIndex);
    }
  }
}

function isTypingInField(target) {
  const t = target;
  if (!t || !t.tagName) return false;
  const tag = t.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable;
}

function onKeyDown(e) {
  if (isTypingInField(e.target)) return;
  if ((activeTab === "track" || activeTab === "live") && e.code === "Escape" && trackState.playing) {
    stopTrackPlayback();
    updateTrackButtons();
    e.preventDefault();
    return;
  }
  if (!state.frames.length) return;
  if (activeTab !== "label") return;

  if (e.code === "Escape") {
    state.linkPendingGroup = null;
    setStatus("Cleared pending segment link.");
    syncUi();
    e.preventDefault();
    return;
  }

  const key = e.key.toLowerCase();
  if (LETTER_TO_GROUP[key] != null) {
    e.preventDefault();
    if (state.currentFrameIndex !== state.baselineFrameIndex) {
      setStatus("Scrub to the baseline frame to assign rigid groups.");
      return;
    }
    if (!state.hoveredPointKey) {
      setStatus("Hover a marker first.");
      return;
    }
    const pi = Number(state.hoveredPointKey.split(":")[1]);
    const g = LETTER_TO_GROUP[key];
    state.markerGroupByIndex.set(pi, g);
    setStatus(`Marker ${pi} → rigid body ${g}`);
    syncUi();
    return;
  }

  if (e.code === "Digit1" || e.code === "Numpad1") {
    e.preventDefault();
    if (state.currentFrameIndex !== state.baselineFrameIndex) {
      setStatus("Scrub to the baseline frame to define segment edges.");
      return;
    }
    if (!state.hoveredPointKey) {
      setStatus("Hover a labelled marker (RB) to use segment link.");
      return;
    }
    const pi = Number(state.hoveredPointKey.split(":")[1]);
    const g = state.markerGroupByIndex.get(pi);
    if (g == null) {
      setStatus("That marker has no rigid group. Use Q–P or A–L first.");
      return;
    }
    if (state.linkPendingGroup == null) {
      state.linkPendingGroup = g;
      setStatus(`Pending: segment from RB ${g}. Hover another RB, press 1 again.`);
      syncUi();
      return;
    }
    const a = state.linkPendingGroup;
    const b = g;
    state.linkPendingGroup = null;
    if (a === b) {
      setStatus("Same rigid body — no segment added. Pick another RB.");
      syncUi();
      return;
    }
    if (segmentExists(a, b)) {
      setStatus(`Segment RB ${Math.min(a, b)} ↔ RB ${Math.max(a, b)} already exists.`);
      syncUi();
      return;
    }
    const n = normalizePair(a, b);
    state.segmentEdges.push(n);
    setStatus(`Added segment edge RB ${n.groupA} ↔ RB ${n.groupB} (blue).`);
    syncUi();
  }
}

function buildGraphPayload() {
  const base = baselineRow();
  const tol = readToleranceMm();

  const rigidBodies = [...new Set(state.markerGroupByIndex.values())]
    .sort((a, b) => a - b)
    .map((id) => ({
      id,
      baselinePointIndices: baselineIndicesInGroup(id),
    }));

  const internalEdges = [];
  for (const rb of rigidBodies) {
    const pts = base.points;
    const idx = rb.baselinePointIndices;
    for (let i = 0; i < idx.length; i++) {
      for (let j = i + 1; j < idx.length; j++) {
        internalEdges.push({
          rigidBodyId: rb.id,
          fromIndex: idx[i],
          toIndex: idx[j],
          baselineLengthMm: dist(pts[idx[i]], pts[idx[j]]),
        });
      }
    }
  }

  const segmentEdgesDetailed = state.segmentEdges.map((e) => {
    const pairs = [];
    forEachCrossGroupPair(e.groupA, e.groupB, (pa, pb) => {
      pairs.push({ baselineLengthMm: dist(pa, pb) });
    });
    return {
      groupA: e.groupA,
      groupB: e.groupB,
      crossMarkerPairCount: pairs.length,
      baselineCrossDistancesMm: pairs.map((p) => p.baselineLengthMm),
    };
  });

  const markerToGroup = {};
  for (const [idx, g] of state.markerGroupByIndex) {
    markerToGroup[String(idx)] = g;
  }

  return {
    schema: "mocap-rigid-graph-v2",
    units: "mm",
    toleranceMm: tol,
    sourceDataset: state.sourceFileName,
    baselineFrameIndex: state.baselineFrameIndex,
    baselineFileFrame: base?.frame ?? null,
    baselineTimeSec: base?.time ?? null,
    markerRigidGroupByBaselineIndex: markerToGroup,
    rigidBodies,
    rigidInternalEdges: internalEdges,
    segmentEdgesBetweenRigidBodies: state.segmentEdges,
    segmentEdgesDetailed,
    notes: {
      colours: "UI: unlabelled red, labelled white; orange = within-RB; blue = segment between RBs.",
      hotkeys:
        "Q–P → groups 1–10; A–L → 11–19. Press 1 twice on two RBs to add a blue segment (all cross pairs).",
    },
  };
}

async function saveGraphToServer() {
  if (!state.sourceFileName) {
    setStatus("Choose a dataset first.");
    return;
  }
  const graph = buildGraphPayload();
  const hint = document.getElementById("save-path-hint");
  try {
    const url = apiUrl("/api/save-graph");
    if (!url) {
      if (hint) hint.textContent = "";
      setStatus("Save needs the dev server (npm start), not file://");
      return;
    }
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sourceFileName: state.sourceFileName, graph }),
    });
    const text = await res.text();
    if (!res.ok || responseLooksLikeStaticServerHtml(text)) {
      if (hint) hint.textContent = "";
      setStatus(responseLooksLikeStaticServerHtml(text) ? WRONG_SERVER_MSG : `Save failed: ${res.status} ${text.slice(0, 120)}`);
      return;
    }
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
    if (!data) {
      if (hint) hint.textContent = "";
      setStatus(WRONG_SERVER_MSG);
      return;
    }
    if (hint && data?.path) {
      hint.textContent = `Saved: ${data.path}`;
    }
    setStatus(`Saved graph → ${data?.path ?? "OK"}`);
  } catch (err) {
    if (hint) hint.textContent = "";
    setStatus(`Save failed (is the server running? npm start): ${err.message}`);
  }
}

function addDisabledDatasetOption(sel, label, titleText) {
  const o = document.createElement("option");
  o.value = "";
  o.textContent = label;
  o.disabled = true;
  if (titleText) o.title = titleText;
  sel.appendChild(o);
}

async function refreshDatasetDropdown() {
  const sel = document.getElementById("dataset-select");
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = '<option value="">— choose —</option>';
  const url = apiUrl("/api/datasets");
  if (!url) {
    addDisabledDatasetOption(sel, "file:// — use Load local TSV or npm start", WRONG_SERVER_MSG);
    setStatus("Opened as file:// — use npm start, or load a TSV with “Load local TSV”.");
    return;
  }
  try {
    const res = await fetch(`${url}${url.includes("?") ? "&" : "?"}_=${Date.now()}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    const body = await res.text();
    const clean = body.replace(/^\uFEFF/, "").trimStart();

    if (responseLooksLikeStaticServerHtml(body)) {
      const port = window.location.port || "(default)";
      addDisabledDatasetOption(
        sel,
        "HTML response — wrong process on this port",
        `${RESTART_HINT} Port: ${port}. /api/datasets must return JSON from node server.mjs.`
      );
      setStatus(
        `Got a web page instead of JSON from ${url}. Another program may be using port ${port} (common: python -m http.server). ${RESTART_HINT}`
      );
      return;
    }

    if (!res.ok) {
      addDisabledDatasetOption(sel, `HTTP ${res.status} — ${clean.slice(0, 35)}…`, clean.slice(0, 300));
      setStatus(`/api/datasets failed (${res.status}). ${RESTART_HINT}`);
      return;
    }

    let names;
    try {
      names = JSON.parse(clean);
    } catch {
      addDisabledDatasetOption(sel, "Invalid JSON from /api/datasets", clean.slice(0, 200));
      setStatus(`Dataset list was not valid JSON. ${RESTART_HINT}`);
      return;
    }

    if (!Array.isArray(names)) {
      addDisabledDatasetOption(sel, "Server JSON was not an array", String(typeof names));
      setStatus("MoCap server should return a JSON array of filenames. " + RESTART_HINT);
      return;
    }

    if (!names.length) {
      addDisabledDatasetOption(sel, "No .tsv in server data/ folder", "Add .tsv files next to server.mjs under data/");
      setStatus("Server returned no datasets — add .tsv files under project data/ and click Refresh dataset list.");
      return;
    }
    for (const n of names) {
      const o = document.createElement("option");
      o.value = n;
      o.textContent = n;
      sel.appendChild(o);
    }
    if (current && names.includes(current)) sel.value = current;
    setStatus(`Loaded dataset list (${names.length} file(s)).`);
  } catch (e) {
    addDisabledDatasetOption(sel, `Network: ${String(e.message).slice(0, 45)}…`, String(e.message));
    setStatus(`Could not reach ${url}. ${RESTART_HINT} — ${e.message}`);
  }
}

function resetLabellingState() {
  state.markerGroupByIndex = new Map();
  state.segmentEdges = [];
  state.linkPendingGroup = null;
  state.hoveredPointKey = null;
  state.cameraFitted = false;
  const hint = document.getElementById("save-path-hint");
  if (hint) hint.textContent = "";
}

/**
 * @param {string} fileName
 * @param {string} text
 */
function applyLoadedTsvText(fileName, text) {
  const maxSec = Number(document.getElementById("max-seconds")?.value) || 10;
  const maxFrames = Number(document.getElementById("max-frames")?.value) || 0;
  state.sourceFileName = fileName;
  state.frames = parseTsv(text, maxSec, maxFrames);
  state.baselineFrameIndex = 0;
  state.currentFrameIndex = 0;
  resetLabellingState();
  trackState.graph = null;
  trackState.result = null;
  clearTrackingReport();
  stopTrackPlayback();
  setStatus(`Loaded ${fileName}: ${state.frames.length} frames. Hover + Q–P / A–L to label.`);
}

/**
 * @param {string} fileName
 */
async function loadDatasetFile(fileName) {
  if (!fileName) {
    state.frames = [];
    state.sourceFileName = null;
    state.lastTsvText = null;
    trackState.graph = null;
    trackState.result = null;
    clearTrackingReport();
    stopTrackPlayback();
    const sk = document.getElementById("skeleton-status");
    if (sk) sk.textContent = "No dataset selected.";
    updateTrackButtons();
    syncUi();
    return;
  }
  const url = apiUrl("/api/dataset");
  if (!url) {
    setStatus("Use “Load local TSV” below, or open the app at http://127.0.0.1:8765/ (npm start).");
    syncUi();
    return;
  }
  const res = await fetch(`${url}?name=${encodeURIComponent(fileName)}`, { cache: "no-store" });
  const text = await res.text();
  if (!res.ok || responseLooksLikeStaticServerHtml(text)) {
    setStatus(responseLooksLikeStaticServerHtml(text) ? WRONG_SERVER_MSG : `Could not load ${fileName}: ${res.status}`);
    state.frames = [];
    state.sourceFileName = null;
    state.lastTsvText = null;
    trackState.graph = null;
    trackState.result = null;
    clearTrackingReport();
    stopTrackPlayback();
    syncUi();
    return;
  }
  state.lastTsvText = null;
  applyLoadedTsvText(fileName, text);
  syncUi();
  await fetchSkeletonForDataset();
  syncUi();
}

function wireUi() {
  const viewport = document.getElementById("viewport");

  document.getElementById("dataset-select")?.addEventListener("change", async (e) => {
    const sel = e.target;
    const name = sel && "value" in sel ? sel.value : "";
    await loadDatasetFile(name);
  });

  document.getElementById("btn-refresh-datasets")?.addEventListener("click", () => {
    refreshDatasetDropdown();
  });

  document.getElementById("local-tsv")?.addEventListener("change", async (e) => {
    const input = e.target;
    const f = input && "files" in input ? input.files?.[0] : null;
    if (!f) return;
    const text = await f.text();
    state.lastTsvText = text;
    applyLoadedTsvText(f.name, text);
    syncUi();
    await fetchSkeletonForDataset();
    syncUi();
    input.value = "";
  });

  function onFrameInput(e) {
    state.currentFrameIndex = Number(e.target.value);
    syncUi();
  }
  document.getElementById("frame-slider")?.addEventListener("input", onFrameInput);
  document.getElementById("frame-slider-track")?.addEventListener("input", onFrameInput);
  document.getElementById("frame-slider-live")?.addEventListener("input", onFrameInput);

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = btn.dataset.tab;
      if (t === "label" || t === "track" || t === "live") setActiveTab(t);
    });
  });

  document.getElementById("btn-reload-skeleton")?.addEventListener("click", () => {
    fetchSkeletonForDataset();
  });

  document.getElementById("btn-run-tracker")?.addEventListener("click", () => {
    void runTrackerAndVisualize();
  });

  document.getElementById("btn-run-live")?.addEventListener("click", () => {
    void runTrackerAndVisualize();
  });

  function toggleTrackPlayback() {
    if (state.frames.length < 2) return;
    if (!trackState.result && !trackState.trackingInProgress) return;
    if (trackState.playing) {
      stopTrackPlayback();
      updateTrackButtons();
      return;
    }
    trackState.playing = true;
    trackState.playAccum = 0;
    for (const id of ["btn-play-track", "btn-play-live"]) {
      const btn = document.getElementById(id);
      if (btn) btn.textContent = "Pause";
    }
    trackState.playTimer = setInterval(playbackTick, PLAYBACK_TICK_MS);
  }

  document.getElementById("btn-play-track")?.addEventListener("click", toggleTrackPlayback);
  document.getElementById("btn-play-live")?.addEventListener("click", toggleTrackPlayback);

  for (const id of ["track-playback-speed", "track-playback-speed-live"]) {
    document.getElementById(id)?.addEventListener("change", () => {
      restartPlaybackInterval();
    });
  }

  for (const id of ["max-seconds", "max-frames"]) {
    document.getElementById(id)?.addEventListener("change", async () => {
      if (!state.sourceFileName) return;
      if (state.lastTsvText) {
        applyLoadedTsvText(state.sourceFileName, state.lastTsvText);
        syncUi();
        await fetchSkeletonForDataset();
        syncUi();
      } else if (getApiRoot()) {
        await loadDatasetFile(state.sourceFileName);
      }
    });
  }

  document.getElementById("btn-baseline")?.addEventListener("click", () => {
    state.baselineFrameIndex = state.currentFrameIndex;
    state.cameraFitted = false;
    setStatus(`Baseline = frame index ${state.baselineFrameIndex}. Re-label if point count/order changed.`);
    syncUi();
  });

  document.getElementById("btn-clear-pending")?.addEventListener("click", () => {
    state.linkPendingGroup = null;
    setStatus("Cleared pending segment link.");
    syncUi();
  });

  document.getElementById("btn-clear-labels")?.addEventListener("click", () => {
    state.markerGroupByIndex.clear();
    state.segmentEdges = [];
    state.linkPendingGroup = null;
    setStatus("Cleared all rigid groups and segment edges.");
    syncUi();
  });

  document.getElementById("btn-export")?.addEventListener("click", () => {
    saveGraphToServer();
  });

  document.getElementById("btn-copy-progress")?.addEventListener("click", async () => {
    const s = serializeProgressText();
    const ta = document.getElementById("progress-text");
    if (ta) ta.value = s;
    try {
      await navigator.clipboard.writeText(s);
      setStatus("Progress text copied to clipboard and filled in the box.");
    } catch {
      setStatus("Progress text is in the box; browser blocked clipboard — copy manually.");
    }
  });

  document.getElementById("btn-apply-progress")?.addEventListener("click", () => {
    const ta = document.getElementById("progress-text");
    const text = ta?.value ?? "";
    const result = parseAndApplyProgressText(text);
    if (!result.ok) {
      setStatus(`Apply failed: ${result.error}`);
      return;
    }
    if (result.warnings?.length) {
      setStatus(`Applied. ${result.warnings.join(" ")}`);
    } else {
      setStatus("Applied pasted labels; baseline set to matching file frame.");
    }
    syncUi();
  });

  window.addEventListener("keydown", onKeyDown);
}

async function main() {
  const viewport = document.getElementById("viewport");
  if (!viewport) return;
  initThree(viewport);
  wireUi();
  await refreshDatasetDropdown();
  updateTrackButtons();
  syncUi();
}

main();
