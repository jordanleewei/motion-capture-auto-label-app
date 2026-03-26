/**
 * Node-only: run the same tracker as the browser on data/*.tsv + mocap-graph.json.
 * CLI: node track-benchmark.mjs [dataset.tsv]
 * API: GET /api/track-benchmark?name=dataset.tsv (see server.mjs)
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { runMultiFrameTracker } from "./web/tracker.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, "data");

function stripTrailingZeros(nums) {
  const out = [...nums];
  while (out.length >= 3 && out[out.length - 3] === 0 && out[out.length - 2] === 0 && out[out.length - 1] === 0) {
    out.length -= 3;
  }
  return out;
}

function parseTsv(text, maxSeconds, maxFramesCap) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
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
    const points = [];
    for (let i = 0; i < trimmed.length; i += 3) {
      points.push({ x: trimmed[i], y: trimmed[i + 1], z: trimmed[i + 2] });
    }
    rows.push({ frame, time, points });
    if (maxFramesCap > 0 && rows.length >= maxFramesCap) break;
  }
  return rows;
}

/** @param {string} name basename only, e.g. mar4qualisystrial1.tsv */
function safeDataFilename(name) {
  if (!name || typeof name !== "string") return null;
  const base = path.basename(name);
  if (base !== name) return null;
  if (base.includes("..") || base.includes("/") || base.includes("\\")) return null;
  if (!/\.(tsv|txt)$/i.test(base)) return null;
  return base;
}

/** @param {object} summary analytics.summary from tracker */
function buildAssessment(summary, totalMs, frameCount, logicalN) {
  const mean = summary.meanAssignRatePct;
  const min = summary.minAssignRatePct;
  const tracked = summary.framesTracked || 1;
  const fullPct = (100 * summary.framesWithAllLogicalAssigned) / tracked;
  const msPerFrame = frameCount > 0 ? totalMs / frameCount : 0;
  const parts = [];

  if (msPerFrame < 2) parts.push(`Throughput is very fast (~${msPerFrame.toFixed(2)} ms/frame wall time on this machine).`);
  else if (msPerFrame < 8) parts.push(`Throughput is good (~${msPerFrame.toFixed(2)} ms/frame).`);
  else if (msPerFrame < 25) parts.push(`Throughput is moderate (~${msPerFrame.toFixed(2)} ms/frame).`);
  else parts.push(`Throughput is heavy (~${msPerFrame.toFixed(2)} ms/frame); long clips will take noticeable time.`);

  if (mean >= 99) parts.push("Label assignment quality is excellent on average across non-baseline frames.");
  else if (mean >= 95) parts.push("Assignment quality is strong on average.");
  else if (mean >= 85) parts.push("Assignment quality is acceptable on average but not tight.");
  else parts.push("Average assignment rate is low — check tolerance, occlusions, or graph vs data.");

  if (min < 70) parts.push(`Worst-frame assignment drops to ${min.toFixed(1)}% — expect visible tracking stress there.`);
  else if (min < 90) parts.push(`Some frames dip to ${min.toFixed(1)}% assignment; review worst frame in the report.`);

  if (fullPct >= 98) parts.push("Nearly all frames have every logical marker assigned.");
  else if (fullPct >= 90) parts.push("Most frames are fully assigned; a minority have missing logicals.");
  else parts.push("Many frames have missing logical assignments — motion or marker count may exceed solver assumptions.");

  if (summary.meanOrangeLengthErrMm != null) {
    const orangeDeg = summary.meanOrangeAngleErrRad * 180 / Math.PI;
    parts.push(`Orange edges (intra-rigid) avg deviation: ${summary.meanOrangeLengthErrMm.toFixed(2)}mm, ${orangeDeg.toFixed(2)}°.`);
    if (summary.meanBlueLengthErrMm != null) {
      parts.push(`Blue edges (hinges/inter-rigid) avg deviation: ${summary.meanBlueLengthErrMm.toFixed(2)}mm.`);
    }
  }

  parts.push(`Clip: ${frameCount} loaded rows, ${logicalN} logical markers (baseline graph).`);
  return parts.join(" ");
}

/**
 * @param {string} datasetFileName e.g. mar4qualisystrial1.tsv
 * @param {{ reappearMm?: number; toleranceMm?: number; edgeAuditMm?: number; edgeAuditEveryFrames?: number; yieldEvery?: number; logProgress?: boolean }} opts
 */
export async function runTrackBenchmark(datasetFileName, opts = {}) {
  const base = safeDataFilename(datasetFileName);
  if (!base) {
    return { ok: false, error: "Invalid dataset name" };
  }

  const tsvPath = path.join(DATA_DIR, base);
  const stem = path.parse(base).name;
  const graphPath = path.join(DATA_DIR, `${stem}_labelled_skeleton`, "mocap-graph.json");

  let text;
  let graphRaw;
  try {
    text = await fs.readFile(tsvPath, "utf8");
  } catch {
    return { ok: false, error: `Missing TSV: ${base}` };
  }
  try {
    graphRaw = await fs.readFile(graphPath, "utf8");
  } catch {
    return { ok: false, error: `Missing graph: data/${stem}_labelled_skeleton/mocap-graph.json` };
  }

  let graph;
  try {
    graph = JSON.parse(graphRaw);
  } catch {
    return { ok: false, error: "Invalid mocap-graph.json" };
  }
  if (!graph.schema) {
    return { ok: false, error: "mocap-graph.json has no schema" };
  }

  const frames = parseTsv(text, Number.POSITIVE_INFINITY, 0);
  if (!frames.length) {
    return { ok: false, error: "No frames parsed from TSV" };
  }

  const b = Math.min(Math.max(0, graph.baselineFrameIndex ?? 0), frames.length - 1);
  if (!frames[b]?.points?.length) {
    return { ok: false, error: "Baseline row has no points" };
  }

  const reappearMm = opts.reappearMm != null ? Number(opts.reappearMm) : 50;
  const toleranceMm = opts.toleranceMm != null ? Number(opts.toleranceMm) : 15;
  const edgeAuditMm = opts.edgeAuditMm != null ? Number(opts.edgeAuditMm) : 100;
  const edgeAuditEveryFrames = opts.edgeAuditEveryFrames != null ? Math.max(1, Math.floor(opts.edgeAuditEveryFrames)) : 1000;
  const n = frames.length;
  const yieldEvery =
    opts.yieldEvery != null && opts.yieldEvery > 0
      ? Math.floor(opts.yieldEvery)
      : Math.max(1, Math.floor(n / 40));

  /** @type {object[]} */
  const live = [];
  const t0 = performance.now();
  let prevT = t0;
  let prevFrameIdx = b;

  const result = await runMultiFrameTracker(frames, graph, {
    reappearMm,
    toleranceMm,
    edgeAuditMm,
    edgeAuditEveryFrames,
    yieldEvery,
    onYield: async (_r, meta) => {
      const t = performance.now();
      const elapsedMs = Math.round((t - t0) * 100) / 100;
      let segmentFrames = null;
      let instantFps = null;
      const dtSec = (t - prevT) / 1000;
      if (meta?.phase === "forward" && meta.frameIndex != null && dtSec > 0) {
        segmentFrames = meta.frameIndex - prevFrameIdx;
        if (segmentFrames > 0) instantFps = Math.round((segmentFrames / dtSec) * 10) / 10;
      }
      if (meta?.phase === "backward" && meta.frameIndex != null && dtSec > 0) {
        segmentFrames = prevFrameIdx - meta.frameIndex;
        if (segmentFrames > 0) instantFps = Math.round((segmentFrames / dtSec) * 10) / 10;
      }
      const row = {
        elapsedMs,
        phase: meta?.phase ?? "unknown",
        frameIndex: meta?.frameIndex ?? null,
        framesThisSegment: segmentFrames,
        instantFps,
      };
      live.push(row);
      if (opts.logProgress) {
        const fps = instantFps != null ? ` ~${instantFps} frames/s` : "";
        console.error(
          `[track-bench] ${elapsedMs.toFixed(0)} ms  ${meta?.phase}${meta?.frameIndex != null ? ` @${meta.frameIndex}` : ""}${fps}`
        );
      }
      prevT = t;
      if (meta?.phase === "forward" || meta?.phase === "forward_done") {
        prevFrameIdx = meta.frameIndex != null ? meta.frameIndex : prevFrameIdx;
      }
      if (meta?.phase === "backward" || meta?.phase === "backward_done") {
        prevFrameIdx = meta.frameIndex != null ? meta.frameIndex : prevFrameIdx;
      }
      if (meta?.phase === "init") prevFrameIdx = b;
    },
  });

  const totalMs = Math.round((performance.now() - t0) * 100) / 100;
  const summary = result.analytics?.summary;
  if (!summary) {
    return { ok: false, error: "Tracker returned no analytics" };
  }

  const overall = {
    wallTimeMs: totalMs,
    loadedRows: frames.length,
    baselineIndex: b,
    logicalMarkers: result.numLogical,
    framesPerSecondWall: Math.round((frames.length / (totalMs / 1000)) * 100) / 100,
    yieldIntervalFrames: yieldEvery,
    reappearMm,
    toleranceMm,
    edgeAuditMm,
    edgeAuditEveryFrames,
    lastFrameMatched: result.stats.lastFrameMatched,
    lastFrameMissing: result.stats.lastFrameMissing,
    trackingQuality: {
      meanAssignRatePct: Math.round(summary.meanAssignRatePct * 100) / 100,
      minAssignRatePct: Math.round(summary.minAssignRatePct * 100) / 100,
      framesWithAllLogicalAssigned: summary.framesWithAllLogicalAssigned,
      framesTrackedExcludingBaseline: summary.framesTracked,
      totalLogicalMissesSummed: summary.totalLogicalMisses,
      totalUnassignedRawSummed: summary.totalUnassignedRawDetections,
      worstFrameIndex: summary.worstFrameIndex,
      worstFileFrame: summary.worstFileFrame,
      meanOrangeLengthErrMm: summary.meanOrangeLengthErrMm != null ? Math.round(summary.meanOrangeLengthErrMm * 100) / 100 : null,
      meanOrangeAngleErrDeg: summary.meanOrangeAngleErrRad != null ? Math.round((summary.meanOrangeAngleErrRad * 180 / Math.PI) * 100) / 100 : null,
      meanBlueLengthErrMm: summary.meanBlueLengthErrMm != null ? Math.round(summary.meanBlueLengthErrMm * 100) / 100 : null,
    },
  };

  const assessment = buildAssessment(summary, totalMs, frames.length, result.numLogical);

  return {
    ok: true,
    dataset: base,
    graphPath: `data/${stem}_labelled_skeleton/mocap-graph.json`,
    liveProgress: live,
    overall,
    diagnostics: result.diagnostics ?? null,
    assessment,
  };
}

const __self = fileURLToPath(import.meta.url);
const ranAsCli = process.argv[1] && path.resolve(process.argv[1]) === __self;
if (ranAsCli) {
  const name = process.argv[2] || "mar4qualisystrial1.tsv";
  
  let toleranceArg = undefined;
  if (process.argv[3] && process.argv[3].startsWith("--tol=")) {
    toleranceArg = Number(process.argv[3].split("=")[1]);
  }

  const out = await runTrackBenchmark(name, { 
    logProgress: true,
    toleranceMm: toleranceArg
  });
  if (!out.ok) {
    console.error(out.error || "Failed");
    process.exit(1);
  }
  console.log("\n=== Backend tracker — live checkpoints (stderr above) ===\n");
  console.log(JSON.stringify({ liveProgress: out.liveProgress, overall: out.overall, diagnostics: out.diagnostics }, null, 2));
  console.log("\n=== Overall performance ===\n");
  console.log(JSON.stringify(out.overall, null, 2));
  console.log("\n=== How it performs ===\n");
  console.log(out.assessment);
  if (out.diagnostics?.narrative?.length) {
    console.log("\n=== Why assignment quality may be low (diagnostics) ===\n");
    console.log(out.diagnostics.narrative.join("\n"));
  }
}
