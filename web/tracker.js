/** @typedef {{ x: number, y: number, z: number }} Vec3 */
/** @typedef {{ frame: number, time: number, points: Vec3[] }} FrameRow */

/**
 * @param {Vec3} a
 * @param {Vec3} b
 */
export function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function buildBaselineIndicesByGroup(graph) {
  const mg = graph.markerRigidGroupByBaselineIndex;
  const map = new Map();
  if (!mg) return map;
  for (const [k, v] of Object.entries(mg)) {
    const groupId = Number(v);
    const mIdx = Number(k);
    let arr = map.get(groupId);
    if (!arr) {
      arr = [];
      map.set(groupId, arr);
    }
    arr.push(mIdx);
  }
  for (const arr of map.values()) {
    arr.sort((a, b) => a - b);
  }
  return map;
}

function buildNeighborGroupsFor(graph) {
  const map = new Map();
  for (const e of graph.segmentEdgesBetweenRigidBodies || []) {
    let arrA = map.get(e.groupA);
    if (!arrA) { arrA = new Set(); map.set(e.groupA, arrA); }
    arrA.add(e.groupB);

    let arrB = map.get(e.groupB);
    if (!arrB) { arrB = new Set(); map.set(e.groupB, arrB); }
    arrB.add(e.groupA);
  }
  const out = new Map();
  for (const [k, v] of map.entries()) {
    out.set(k, [...v]);
  }
  return out;
}

function buildHingeLogicalSet(graph, baselinePts, baselineIndicesByGroup) {
  const hinge = new Set();
  const segs = graph.segmentEdgesBetweenRigidBodies || [];
  for (const e of segs) {
    const ia = baselineIndicesByGroup.get(e.groupA) || [];
    const ib = baselineIndicesByGroup.get(e.groupB) || [];
    let best = Infinity;
    let pi = -1;
    let pj = -1;
    for (const i of ia) {
      for (const j of ib) {
        const d = dist(baselinePts[i], baselinePts[j]);
        if (d < best) {
          best = d;
          pi = i;
          pj = j;
        }
      }
    }
    if (pi >= 0) {
      hinge.add(pi);
      hinge.add(pj);
    }
  }
  return hinge;
}

/**
 * @param {Vec3} p
 */
function copyVec(p) {
  return { x: p.x, y: p.y, z: p.z };
}

function createDiagnosticsPass() {
  return {
    frames: 0,
    bootstrapAssignmentsTotal: 0,
    finalAssignmentsTotal: 0,
    assignmentDeltaVsBootstrapTotal: 0,
    missingLogicalSlotObservations: 0,
    missingWithZeroAssignedNeighbors: 0,
    missingWithAtLeastOneAssignedNeighbor: 0,
    framesWithPointCloudLtLogical: 0,
    minRawPointCount: Number.POSITIVE_INFINITY,
    maxRawPointCount: 0,
    sumRawPointCount: 0,
  };
}

/**
 * @param {ReturnType<typeof createDiagnosticsPass>} forward
 * @param {ReturnType<typeof createDiagnosticsPass>} backward
 * @param {number} numLogical
 * @param {number} framesTracked
 */
function buildDiagnosticsSummary(forward, backward, numLogical, framesTracked) {
  const fFrames = forward.frames || 1;
  const bFrames = backward.frames || 1;
  const fmt = (pass, label) => {
    const avgBoot = (pass.bootstrapAssignmentsTotal / pass.frames).toFixed(2);
    const avgFinal = (pass.finalAssignmentsTotal / pass.frames).toFixed(2);
    const avgDelta = (pass.assignmentDeltaVsBootstrapTotal / pass.frames).toFixed(2);
    const miss = pass.missingLogicalSlotObservations;
    const miss0 = pass.missingWithZeroAssignedNeighbors;
    const missN = pass.missingWithAtLeastOneAssignedNeighbor;
    const pct0 = miss > 0 ? ((100 * miss0) / miss).toFixed(1) : "0";
    const pctN = miss > 0 ? ((100 * missN) / miss).toFixed(1) : "0";
    return [
      `${label}: avg carry-forward (same raw index) ≈ ${avgBoot}; avg final assigned ≈ ${avgFinal}; avg (final − carry-forward) ≈ ${avgDelta} (fingerprint / rescue fill; can be negative).`,
      `${label}: missing logical slots (sum over frames) ${miss}; of those, ${miss0} (${pct0}%) had zero assigned graph neighbors (fingerprint cannot constrain); ${missN} (${pctN}%) had ≥1 neighbor but still no valid raw point within tolerance.`,
      `${label}: frames with raw point count < numLogical: ${pass.framesWithPointCloudLtLogical}; raw count min ${!Number.isFinite(pass.minRawPointCount) ? "n/a" : pass.minRawPointCount}, max ${pass.maxRawPointCount}, mean ${pass.frames > 0 ? (pass.sumRawPointCount / pass.frames).toFixed(1) : "n/a"}.`,
    ];
  };
  const narrative = [
    "Why quality may be low (heuristic interpretation):",
    "• Forward vs backward: forward (frames after baseline) is where identity is propagated through time; backward (frames before baseline) walks from the seeded baseline backward, so carry-forward often stays complete for f<b and backward metrics can look much healthier than forward — prioritize forward-pass numbers for clips mostly after baseline.",
    "• Carry-forward ties identity to the same raw index as the previous frame; if markers reorder in the cloud or indices shift, wrong points are carried and edge errors explode.",
    "• Fingerprint fill only assigns when ≥1 neighbor is already placed and some raw point matches all constraints within tolerance; large empty regions of missing neighbors block recovery.",
    "• Orange angles add arc-length error terms; strict tolerance rejects many candidates even when lengths look fine.",
    "• Edge-length quality is reported in the run report (plots); it does not remove assignments in the tracker.",
    ...fmt(forward, "Forward pass"),
    ...fmt(backward, "Backward pass"),
  ];
  return {
    forward,
    backward,
    framesTrackedExcludingBaseline: framesTracked,
    logicalMarkers: numLogical,
    narrative,
  };
}

/**
 * @param {FrameRow[]} frames
 * @param {object} graph
 * @param {{
 *   reappearMm?: number;
 *   toleranceMm?: number;
 *   followLookbackFrames?: number;
 *   edgeWarningThresholdMm?: number | null;
 *   yieldEvery?: number;
 *   onYield?: (r: object, meta?: { phase: string; frameIndex?: number }) => void | Promise<void>;
 * }} opts
 */
export async function runMultiFrameTracker(frames, graph, opts) {
  const reappear = opts.reappearMm != null ? Number(opts.reappearMm) : 50;
  const tol = opts.toleranceMm != null ? Number(opts.toleranceMm) : 100;
  const lookback = Math.max(1, Math.min(30, Math.floor(opts.followLookbackFrames != null ? Number(opts.followLookbackFrames) : 10)));
  const yieldEvery = opts.yieldEvery != null && opts.yieldEvery > 0 ? Math.floor(opts.yieldEvery) : 2;
  const onYield = opts.onYield;

  const b = Math.min(Math.max(0, graph.baselineFrameIndex ?? 0), frames.length - 1);
  const baselinePts = frames[b].points;
  const numLogical = baselinePts.length;

  /** @type {Map<number, number>} */
  const groupByLogical = new Map();
  const mg = graph.markerRigidGroupByBaselineIndex;
  if (mg && typeof mg === "object") {
    for (const [k, v] of Object.entries(mg)) {
      groupByLogical.set(Number(k), Number(v));
    }
  }

  const baselineIndicesByGroup = buildBaselineIndicesByGroup(graph);
  
  // Precompute Geometric Fingerprints from baseline
  const orangeNeighbors = Array.from({length: numLogical}, () => []);
  const blueNeighbors = Array.from({length: numLogical}, () => []);
  const orangeAngles = Array.from({length: numLogical}, () => []);

  for (const idx of baselineIndicesByGroup.values()) {
    for (let i = 0; i < idx.length; i++) {
      for (let j = 0; j < idx.length; j++) {
        if (i === j) continue;
        const u = idx[i], v = idx[j];
        orangeNeighbors[u].push({ j: v, d0: dist(baselinePts[u], baselinePts[v]) });
      }
    }
    for (let i = 0; i < idx.length; i++) {
      for (let j = 0; j < idx.length; j++) {
        for (let k = j + 1; k < idx.length; k++) {
          if (i === j || i === k) continue;
          const u = idx[i], v1 = idx[j], v2 = idx[k];
          const d0p = dist(baselinePts[u], baselinePts[v1]);
          const d0q = dist(baselinePts[u], baselinePts[v2]);
          if (d0p > 0 && d0q > 0) {
            const px = baselinePts[v1].x - baselinePts[u].x;
            const py = baselinePts[v1].y - baselinePts[u].y;
            const pz = baselinePts[v1].z - baselinePts[u].z;
            const qx = baselinePts[v2].x - baselinePts[u].x;
            const qy = baselinePts[v2].y - baselinePts[u].y;
            const qz = baselinePts[v2].z - baselinePts[u].z;
            let cos0 = (px * qx + py * qy + pz * qz) / (d0p * d0q);
            if (cos0 > 1) cos0 = 1; else if (cos0 < -1) cos0 = -1;
            orangeAngles[u].push({ j: v1, k: v2, angle0: Math.acos(cos0), d0p, d0q });
          }
        }
      }
    }
  }

  const segs = graph.segmentEdgesBetweenRigidBodies || [];
  for (const e of segs) {
    const ia = baselineIndicesByGroup.get(e.groupA) || [];
    const ib = baselineIndicesByGroup.get(e.groupB) || [];
    for (const u of ia) {
      for (const v of ib) {
        const d0 = dist(baselinePts[u], baselinePts[v]);
        blueNeighbors[u].push({ j: v, d0 });
        blueNeighbors[v].push({ j: u, d0 });
      }
    }
  }

  const hingeLogical = buildHingeLogicalSet(graph, baselinePts, baselineIndicesByGroup);
  const hingeLogicalArr = [...hingeLogical];

  const perFrame = frames.map(() => ({
    logicalPos: /** @type {(Vec3 | null)[]} */ new Array(numLogical).fill(null),
    rawForLogical: /** @type {(number | null)[]} */ new Array(numLogical).fill(null),
  }));

  for (let i = 0; i < numLogical; i++) {
    const p = baselinePts[i];
    perFrame[b].logicalPos[i] = { x: p.x, y: p.y, z: p.z };
    perFrame[b].rawForLogical[i] = i;
  }

  const diagnostics = {
    forward: createDiagnosticsPass(),
    backward: createDiagnosticsPass(),
  };

  const result = {
    baselineFrameIndex: b,
    numLogical,
    groupByLogical,
    hingeLogical: hingeLogicalArr,
    perFrame,
    stats: { frames: frames.length, logicalMarkers: numLogical, lastFrameMatched: 0, lastFrameMissing: numLogical },
    analytics: null,
    diagnostics: null,
  };

  async function yieldNow(meta) {
    await onYield?.(result, meta);
  }

  await yieldNow({ phase: "init", frameIndex: b });

  /**
   * Nearest-in-time reference position for logical marker `i` from up to `lookback` already-solved frames.
   * Forward: f-1, f-2, … backward in time. Backward pass: f+1, f+2, … forward in time.
   */
  function refLogicalPosFromHistory(f, i, pass) {
    for (let k = 1; k <= lookback; k++) {
      const idx = pass === "forward" ? f - k : f + k;
      if (idx < 0 || idx >= frames.length) break;
      const lp = perFrame[idx].logicalPos[i];
      if (lp != null) return lp;
    }
    return null;
  }

  /**
   * @param {number} f
   * @param {'forward' | 'backward'} pass
   */
  function processFrame(f, pass) {
    const diag = diagnostics[pass];
    const curPts = frames[f].points;
    const n = curPts.length;
    const usedRaw = new Set();
    const rawForLogical = new Array(numLogical).fill(null);
    const logicalPos = new Array(numLogical).fill(null);

    diag.frames++;
    diag.minRawPointCount = Math.min(diag.minRawPointCount, n);
    diag.maxRawPointCount = Math.max(diag.maxRawPointCount, n);
    diag.sumRawPointCount += n;
    if (n < numLogical) diag.framesWithPointCloudLtLogical++;

    // 1. Follow recent frames: nearest raw point to the most recent known logical position (within lookback), within reappear distance.
    let bootstrapCount = 0;
    for (let i = 0; i < numLogical; i++) {
      const prevPos = refLogicalPosFromHistory(f, i, pass);
      if (!prevPos) continue;

      let bestR = -1;
      let bestD = Infinity;
      for (let r = 0; r < n; r++) {
        if (usedRaw.has(r)) continue;
        const d = dist(curPts[r], prevPos);
        if (d < bestD) {
          bestD = d;
          bestR = r;
        }
      }

      if (bestR !== -1 && bestD <= reappear) {
        bootstrapCount++;
        rawForLogical[i] = bestR;
        logicalPos[i] = copyVec(curPts[bestR]);
        usedRaw.add(bestR);
      }
    }
    diag.bootstrapAssignmentsTotal += bootstrapCount;

    // 2. Fingerprint fill (unassigned logicals)
    function runFingerprintFill() {
      let progress = true;
      while (progress) {
        progress = false;
        const candidates = [];

        for (let i = 0; i < numLogical; i++) {
          if (rawForLogical[i] != null) continue;

          for (let r = 0; r < n; r++) {
            if (usedRaw.has(r)) continue;

            let maxErr = 0;
            let count = 0;

            for (const nb of orangeNeighbors[i]) {
              if (rawForLogical[nb.j] == null) continue;
              const refPos = logicalPos[nb.j];
              const d = dist(curPts[r], refPos);
              maxErr = Math.max(maxErr, Math.abs(d - nb.d0));
              count++;
            }

            for (const nb of blueNeighbors[i]) {
              if (rawForLogical[nb.j] == null) continue;
              const refPos = logicalPos[nb.j];
              const d = dist(curPts[r], refPos);
              maxErr = Math.max(maxErr, Math.abs(d - nb.d0));
              count++;
            }

            for (const ang of orangeAngles[i]) {
              if (rawForLogical[ang.j] == null || rawForLogical[ang.k] == null) continue;
              const refJ = logicalPos[ang.j];
              const refK = logicalPos[ang.k];
              const dp = dist(curPts[r], refJ);
              const dq = dist(curPts[r], refK);
              if (dp > 0 && dq > 0) {
                const vpx = refJ.x - curPts[r].x;
                const vpy = refJ.y - curPts[r].y;
                const vpz = refJ.z - curPts[r].z;
                const vqx = refK.x - curPts[r].x;
                const vqy = refK.y - curPts[r].y;
                const vqz = refK.z - curPts[r].z;
                let cos = (vpx * vqx + vpy * vqy + vpz * vqz) / (dp * dq);
                if (cos > 1) cos = 1;
                else if (cos < -1) cos = -1;
                const angle = Math.acos(cos);
                const da = Math.abs(angle - ang.angle0);
                const ea = da * ((ang.d0p + ang.d0q) / 2);
                maxErr = Math.max(maxErr, ea);
                count++;
              }
            }

            if (count > 0 && maxErr <= tol) {
              candidates.push({ i, r, err: maxErr, constraints: count });
            }
          }
        }

        if (candidates.length > 0) {
          candidates.sort((a, b) => a.err - b.err || b.constraints - a.constraints);
          for (const cand of candidates) {
            if (rawForLogical[cand.i] == null && !usedRaw.has(cand.r)) {
              rawForLogical[cand.i] = cand.r;
              logicalPos[cand.i] = copyVec(curPts[cand.r]);
              usedRaw.add(cand.r);
              progress = true;
              break;
            }
          }
        }
      }
    }

    runFingerprintFill();

    // 3. Rigid-body rescue (unassigned bodies)
    // For completely missing rigid bodies, search the unassigned raw points for a subset 
    // that exactly matches the baseline shape (all orange edges).
    for (const [groupId, indices] of baselineIndicesByGroup.entries()) {
      // Check if this rigid body is completely lost (no assigned markers)
      let anyAssigned = false;
      for (const i of indices) {
        if (rawForLogical[i] != null) {
          anyAssigned = true;
          break;
        }
      }
      
      if (!anyAssigned && indices.length >= 3) {
        // Collect unassigned raw points
        const unassignedRaw = [];
        for (let r = 0; r < n; r++) {
          if (!usedRaw.has(r)) unassignedRaw.push(r);
        }
        
        // Very basic brute-force combinatorial search for a matching shape.
        // We look for a subset of raw points that match ALL orange distances of this rigid body.
        // (This can be computationally expensive if there are many unassigned points, 
        // but is critical to break the deadlock).
        let foundMatch = null;
        
        // To keep it simple and fast enough, we just try to find a mapping for the first 3 indices of the body.
        // If 3 points match their 3 distances, the rigid body is recovered enough for Fingerprint Fill to get the rest next frame.
        const i0 = indices[0];
        const i1 = indices[1];
        const i2 = indices[2];
        const d01 = dist(baselinePts[i0], baselinePts[i1]);
        const d02 = dist(baselinePts[i0], baselinePts[i2]);
        const d12 = dist(baselinePts[i1], baselinePts[i2]);
        
        for (let a = 0; a < unassignedRaw.length; a++) {
          const ra = unassignedRaw[a];
          for (let b = 0; b < unassignedRaw.length; b++) {
            if (a === b) continue;
            const rb = unassignedRaw[b];
            if (Math.abs(dist(curPts[ra], curPts[rb]) - d01) > tol) continue;
            
            for (let c = 0; c < unassignedRaw.length; c++) {
              if (c === a || c === b) continue;
              const rc = unassignedRaw[c];
              if (Math.abs(dist(curPts[ra], curPts[rc]) - d02) > tol) continue;
              if (Math.abs(dist(curPts[rb], curPts[rc]) - d12) > tol) continue;
              
              foundMatch = [ra, rb, rc];
              break;
            }
            if (foundMatch) break;
          }
          if (foundMatch) break;
        }
        
        if (foundMatch) {
          rawForLogical[i0] = foundMatch[0]; logicalPos[i0] = copyVec(curPts[foundMatch[0]]); usedRaw.add(foundMatch[0]);
          rawForLogical[i1] = foundMatch[1]; logicalPos[i1] = copyVec(curPts[foundMatch[1]]); usedRaw.add(foundMatch[1]);
          rawForLogical[i2] = foundMatch[2]; logicalPos[i2] = copyVec(curPts[foundMatch[2]]); usedRaw.add(foundMatch[2]);
          // Run fingerprint again immediately to fill any remaining points on this body now that it has anchors
          runFingerprintFill();
        }
      }
    }

    let assignedFinal = 0;
    for (let i = 0; i < numLogical; i++) {
      if (rawForLogical[i] != null) assignedFinal++;
    }
    diag.finalAssignmentsTotal += assignedFinal;
    diag.assignmentDeltaVsBootstrapTotal += assignedFinal - bootstrapCount;

    for (let i = 0; i < numLogical; i++) {
      if (rawForLogical[i] != null) continue;
      diag.missingLogicalSlotObservations++;
      const seenNbr = new Set();
      for (const nb of orangeNeighbors[i]) {
        if (rawForLogical[nb.j] != null) seenNbr.add(nb.j);
      }
      for (const nb of blueNeighbors[i]) {
        if (rawForLogical[nb.j] != null) seenNbr.add(nb.j);
      }
      if (seenNbr.size === 0) diag.missingWithZeroAssignedNeighbors++;
      else diag.missingWithAtLeastOneAssignedNeighbor++;
    }

    for (let i = 0; i < numLogical; i++) {
      perFrame[f].logicalPos[i] = logicalPos[i];
      perFrame[f].rawForLogical[i] = rawForLogical[i];
    }
  }

  for (let f = b + 1; f < frames.length; f++) {
    processFrame(f, "forward");
    if (yieldEvery > 0 && (f - b) % yieldEvery === 0) await yieldNow({ phase: "forward", frameIndex: f });
  }
  await yieldNow({ phase: "forward_done", frameIndex: frames.length > 0 ? Math.max(b, frames.length - 1) : b });

  let backStep = 0;
  for (let f = b - 1; f >= 0; f--) {
    processFrame(f, "backward");
    backStep++;
    if (yieldEvery > 0 && backStep % yieldEvery === 0) await yieldNow({ phase: "backward", frameIndex: f });
  }
  await yieldNow({ phase: "backward_done", frameIndex: 0 });

  let matchedLast = 0;
  let missingLast = 0;
  if (frames.length) {
    const last = perFrame[frames.length - 1];
    for (let i = 0; i < numLogical; i++) {
      if (last.rawForLogical[i] != null) matchedLast++;
      else missingLast++;
    }
  }

  result.stats.lastFrameMatched = matchedLast;
  result.stats.lastFrameMissing = missingLast;
  result.analytics = computeTrackingAnalytics(frames, { baselineFrameIndex: b, numLogical, perFrame }, graph, {
    edgeWarningThresholdMm: opts.edgeWarningThresholdMm,
  });
  result.diagnostics = buildDiagnosticsSummary(
    diagnostics.forward,
    diagnostics.backward,
    numLogical,
    Math.max(0, frames.length - 1)
  );

  await yieldNow({ phase: "complete" });
  return result;
}

/**
 * Per-frame and summary metrics for the full run (baseline is the seeded frame, excluded from aggregates).
 * @param {FrameRow[]} frames
 * @param {{ baselineFrameIndex: number; numLogical: number; perFrame: { rawForLogical: (number | null)[] }[] }} result
 * @param {object | null} graph
 * @param {{ edgeWarningThresholdMm?: number | null }} [opts]
 */
export function computeTrackingAnalytics(frames, result, graph, opts) {
  const warnTh =
    opts?.edgeWarningThresholdMm != null && Number.isFinite(Number(opts.edgeWarningThresholdMm))
      ? Number(opts.edgeWarningThresholdMm)
      : null;
  const b = result.baselineFrameIndex;
  const n = result.numLogical;
  const baselinePts = frames[b].points;

  const orangeEdges = [];
  const orangeAngles = [];
  const blueEdges = [];

  if (graph) {
    const baselineIndicesByGroup = buildBaselineIndicesByGroup(graph);
    for (const idx of baselineIndicesByGroup.values()) {
      for (let i = 0; i < idx.length; i++) {
        for (let j = i + 1; j < idx.length; j++) {
          const u = idx[i], v = idx[j];
          orangeEdges.push({ u, v, d0: dist(baselinePts[u], baselinePts[v]) });
        }
      }
      for (let i = 0; i < idx.length; i++) {
        for (let j = 0; j < idx.length; j++) {
          for (let k = j + 1; k < idx.length; k++) {
            if (i === j || i === k) continue;
            const u = idx[i], v1 = idx[j], v2 = idx[k];
            const d0p = dist(baselinePts[u], baselinePts[v1]);
            const d0q = dist(baselinePts[u], baselinePts[v2]);
            if (d0p > 0 && d0q > 0) {
              const px = baselinePts[v1].x - baselinePts[u].x;
              const py = baselinePts[v1].y - baselinePts[u].y;
              const pz = baselinePts[v1].z - baselinePts[u].z;
              const qx = baselinePts[v2].x - baselinePts[u].x;
              const qy = baselinePts[v2].y - baselinePts[u].y;
              const qz = baselinePts[v2].z - baselinePts[u].z;
              let cos0 = (px * qx + py * qy + pz * qz) / (d0p * d0q);
              if (cos0 > 1) cos0 = 1; else if (cos0 < -1) cos0 = -1;
              orangeAngles.push({ u, v1, v2, angle0: Math.acos(cos0), d0p, d0q });
            }
          }
        }
      }
    }
    const segs = graph.segmentEdgesBetweenRigidBodies || [];
    for (const e of segs) {
      const ia = baselineIndicesByGroup.get(e.groupA) || [];
      const ib = baselineIndicesByGroup.get(e.groupB) || [];
      for (const u of ia) {
        for (const v of ib) {
          blueEdges.push({ u, v, d0: dist(baselinePts[u], baselinePts[v]) });
        }
      }
    }
  }

  const frameStats = frames.map((row, fi) => {
    const pf = result.perFrame[fi];
    let matched = 0;
    let missing = 0;
    for (let i = 0; i < n; i++) {
      if (pf.rawForLogical[i] != null) matched++;
      else missing++;
    }
    const used = new Set();
    for (let i = 0; i < n; i++) {
      const r = pf.rawForLogical[i];
      if (r != null) used.add(r);
    }
    let unassignedRaw = 0;
    for (let j = 0; j < row.points.length; j++) {
      if (!used.has(j)) unassignedRaw++;
    }
    const rate = n > 0 ? (matched / n) * 100 : 0;

    let orangeLenErr = 0; let orangeLenCount = 0;
    let orangeAngErr = 0; let orangeAngCount = 0;
    let blueLenErr = 0; let blueLenCount = 0;

    if (graph) {
      const curPts = row.points;
      for (const e of orangeEdges) {
        const ru = pf.rawForLogical[e.u], rv = pf.rawForLogical[e.v];
        if (ru != null && rv != null) {
          orangeLenErr += Math.abs(dist(curPts[ru], curPts[rv]) - e.d0);
          orangeLenCount++;
        }
      }
      for (const e of orangeAngles) {
        const ru = pf.rawForLogical[e.u], rv1 = pf.rawForLogical[e.v1], rv2 = pf.rawForLogical[e.v2];
        if (ru != null && rv1 != null && rv2 != null) {
          const d1 = dist(curPts[ru], curPts[rv1]);
          const d2 = dist(curPts[ru], curPts[rv2]);
          if (d1 > 0 && d2 > 0) {
            const px = curPts[rv1].x - curPts[ru].x;
            const py = curPts[rv1].y - curPts[ru].y;
            const pz = curPts[rv1].z - curPts[ru].z;
            const qx = curPts[rv2].x - curPts[ru].x;
            const qy = curPts[rv2].y - curPts[ru].y;
            const qz = curPts[rv2].z - curPts[ru].z;
            let cos = (px * qx + py * qy + pz * qz) / (d1 * d2);
            if (cos > 1) cos = 1; else if (cos < -1) cos = -1;
            const angle = Math.acos(cos);
            orangeAngErr += Math.abs(angle - e.angle0);
            orangeAngCount++;
          }
        }
      }
      for (const e of blueEdges) {
        const ru = pf.rawForLogical[e.u], rv = pf.rawForLogical[e.v];
        if (ru != null && rv != null) {
          blueLenErr += Math.abs(dist(curPts[ru], curPts[rv]) - e.d0);
          blueLenCount++;
        }
      }
    }

    const oLen = orangeLenCount > 0 ? orangeLenErr / orangeLenCount : null;
    const bLen = blueLenCount > 0 ? blueLenErr / blueLenCount : null;
    let edgeFlagged = false;
    if (warnTh != null && warnTh > 0 && fi !== b) {
      if (oLen != null && oLen > warnTh) edgeFlagged = true;
      if (bLen != null && bLen > warnTh) edgeFlagged = true;
    }

    return {
      frameIndex: fi,
      fileFrame: row.frame,
      timeSec: row.time,
      rawPointCount: row.points.length,
      matched,
      missing,
      unassignedRaw,
      rate,
      isBaseline: fi === b,
      orangeLenErr: oLen,
      orangeAngErr: orangeAngCount > 0 ? orangeAngErr / orangeAngCount : null,
      blueLenErr: bLen,
      edgeFlagged,
    };
  });

  const nonBaseline = frameStats.filter((s) => !s.isBaseline);
  const meanRate =
    nonBaseline.length > 0 ? nonBaseline.reduce((a, s) => a + s.rate, 0) / nonBaseline.length : 0;

  let sumOrangeLen = 0; let nOrangeLen = 0;
  let sumOrangeAng = 0; let nOrangeAng = 0;
  let sumBlueLen = 0; let nBlueLen = 0;

  let minRate = 100;
  let worstFi = 0;
  for (const s of nonBaseline) {
    if (s.rate < minRate) {
      minRate = s.rate;
      worstFi = s.frameIndex;
    }
    if (s.orangeLenErr != null) { sumOrangeLen += s.orangeLenErr; nOrangeLen++; }
    if (s.orangeAngErr != null) { sumOrangeAng += s.orangeAngErr; nOrangeAng++; }
    if (s.blueLenErr != null) { sumBlueLen += s.blueLenErr; nBlueLen++; }
  }
  const totalAssigned = nonBaseline.reduce((a, s) => a + s.matched, 0);
  const totalMissing = nonBaseline.reduce((a, s) => a + s.missing, 0);
  const totalUnassignedRaw = nonBaseline.reduce((a, s) => a + s.unassignedRaw, 0);
  const framesFullAssign = nonBaseline.filter((s) => s.missing === 0).length;
  const framesEdgeFlagged = nonBaseline.filter((s) => s.edgeFlagged).length;

  return {
    frameStats,
    summary: {
      baselineFrameIndex: b,
      framesTotal: frames.length,
      framesTracked: nonBaseline.length,
      logicalMarkers: n,
      meanAssignRatePct: meanRate,
      minAssignRatePct: nonBaseline.length ? minRate : 0,
      worstFrameIndex: nonBaseline.length ? worstFi : null,
      worstFileFrame: nonBaseline.length ? frameStats[worstFi]?.fileFrame ?? null : null,
      framesWithAllLogicalAssigned: framesFullAssign,
      totalLogicalAssignments: totalAssigned,
      totalLogicalMisses: totalMissing,
      totalUnassignedRawDetections: totalUnassignedRaw,
      meanOrangeLengthErrMm: nOrangeLen > 0 ? sumOrangeLen / nOrangeLen : null,
      meanOrangeAngleErrRad: nOrangeAng > 0 ? sumOrangeAng / nOrangeAng : null,
      meanBlueLengthErrMm: nBlueLen > 0 ? sumBlueLen / nBlueLen : null,
      edgeWarningThresholdMm: warnTh,
      framesEdgeFlagged,
    },
  };
}

/**
 * @param {FrameRow[]} frames
 * @param {Awaited<ReturnType<typeof runMultiFrameTracker>>} result
 * @param {number} f
 */
export function buildRawToLogicalMap(frames, result, f) {
  const row = frames[f];
  if (!row) return new Map();
  const map = new Map();
  const pf = result.perFrame[f];
  if (!pf) return map;
  for (let i = 0; i < result.numLogical; i++) {
    const r = pf.rawForLogical[i];
    if (r != null) map.set(r, i);
  }
  return map;
}
