/**
 * Run the tracker with benchmark defaults (50 / 100 / 10 / 150 mm) and write
 * an HTML file with Chart.js plots: raw vs assigned counts, assignment rate, and edge errors over time.
 *
 * Usage: node track-charts.mjs [dataset.tsv]
 * Output: reports/track-defaults-charts.html
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runTrackBenchmark } from "./track-benchmark.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const OUT_DIR = path.join(ROOT, "reports");
const OUT_HTML = path.join(OUT_DIR, "track-defaults-charts.html");

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildHtml(payload) {
  const { overall, frameStats, dataset, assessment } = payload;
  const paramsLine = `reappear ${overall.reappearMm} mm · tolerance ${overall.toleranceMm} mm · lookback ${overall.followLookbackFrames} frames · edge warn ${overall.edgeWarningThresholdMm ?? "off"} mm`;
  const dataJson = JSON.stringify({ frameStats, overall });

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tracker defaults — accuracy &amp; error over time</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root { font-family: system-ui, sans-serif; color: #1a1a1a; background: #f4f4f5; }
    body { margin: 0 auto; max-width: 1100px; padding: 1.25rem 1rem 3rem; }
    h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 0.35rem; }
    .meta { color: #555; font-size: 0.9rem; margin-bottom: 0.75rem; }
    .params { font-family: ui-monospace, monospace; font-size: 0.82rem; background: #fff; border: 1px solid #ddd; padding: 0.5rem 0.65rem; border-radius: 6px; margin-bottom: 1rem; }
    .assessment { font-size: 0.88rem; line-height: 1.45; color: #333; margin-bottom: 1.25rem; max-width: 85ch; }
    .chart-wrap { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.75rem 0.75rem 0.25rem; margin-bottom: 1.25rem; }
    .chart-wrap h2 { font-size: 0.95rem; font-weight: 600; margin: 0 0 0.5rem; color: #333; }
    canvas { max-height: 360px; }
  </style>
</head>
<body>
  <h1>Tracker test — defaults</h1>
  <div class="meta">Dataset: <strong>${escapeHtml(dataset)}</strong> · ${frameStats.length} frames</div>
  <div class="params">${escapeHtml(paramsLine)}</div>
  <p class="assessment">${escapeHtml(assessment)}</p>
  <div class="chart-wrap">
    <h2>Raw detections vs logical markers assigned (per frame)</h2>
    <canvas id="chart-raw-vs-matched"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Assignment rate (accuracy) over time</h2>
    <canvas id="chart-rate"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Mean edge length error — orange (intra-rigid) &amp; blue (inter-rigid), mm</h2>
    <canvas id="chart-edge"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Mean orange angle error (rad)</h2>
    <canvas id="chart-angle"></canvas>
  </div>
  <script>
  const BUNDLE = ${dataJson};
  const fs = BUNDLE.frameStats;
  const labels = fs.map((r) => r.timeSec);
  const xTitle = "Time (s, from TSV)";

  const rawN = fs.map((r) =>
    r.rawPointCount != null ? r.rawPointCount : r.matched + r.unassignedRaw
  );
  const matchedN = fs.map((r) => r.matched);
  const logicalN = BUNDLE.overall.logicalMarkers;

  const rateData = fs.map((r) => (r.isBaseline ? null : r.rate));
  const orangeMm = fs.map((r) => (r.isBaseline ? null : r.orangeLenErr));
  const blueMm = fs.map((r) => (r.isBaseline ? null : r.blueLenErr));
  const angRad = fs.map((r) => (r.isBaseline ? null : r.orangeAngErr));
  const warnTh = BUNDLE.overall.edgeWarningThresholdMm;

  const commonOpts = {
    responsive: true,
    maintainAspectRatio: true,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { title: { display: true, text: xTitle }, ticks: { maxTicksLimit: 12 } },
    },
    plugins: { legend: { position: "top" } },
  };

  const rawVsDatasets = [
    {
      label: "Raw detections (TSV points)",
      data: rawN,
      borderColor: "rgb(22, 163, 74)",
      backgroundColor: "rgba(22, 163, 74, 0.06)",
      pointRadius: 0,
      tension: 0.1,
    },
    {
      label: "Logical markers assigned",
      data: matchedN,
      borderColor: "rgb(59, 130, 246)",
      backgroundColor: "rgba(59, 130, 246, 0.06)",
      pointRadius: 0,
      tension: 0.1,
    },
    {
      label: "Logical count (graph)",
      data: fs.map(() => logicalN),
      borderColor: "rgb(161, 161, 170)",
      borderDash: [4, 4],
      pointRadius: 0,
      fill: false,
    },
  ];

  new Chart(document.getElementById("chart-raw-vs-matched"), {
    type: "line",
    data: { labels, datasets: rawVsDatasets },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: {
          title: { display: true, text: "Count" },
          beginAtZero: true,
          ticks: { precision: 0 },
        },
      },
    },
  });

  new Chart(document.getElementById("chart-rate"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Logical markers assigned (%)",
          data: rateData,
          borderColor: "rgb(37, 99, 235)",
          backgroundColor: "rgba(37, 99, 235, 0.08)",
          spanGaps: false,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: {
          title: { display: true, text: "% assigned" },
          min: 0,
          max: 100,
        },
      },
    },
  });

  const edgeDatasets = [
    {
      label: "Orange mean length error (mm)",
      data: orangeMm,
      borderColor: "rgb(234, 88, 12)",
      backgroundColor: "rgba(234, 88, 12, 0.06)",
      pointRadius: 0,
      tension: 0.1,
    },
    {
      label: "Blue mean length error (mm)",
      data: blueMm,
      borderColor: "rgb(37, 99, 235)",
      backgroundColor: "rgba(37, 99, 235, 0.06)",
      pointRadius: 0,
      tension: 0.1,
    },
  ];
  if (warnTh != null && Number.isFinite(warnTh)) {
    edgeDatasets.push({
      label: "Edge warning threshold (mm)",
      data: fs.map(() => warnTh),
      borderColor: "rgb(220, 38, 38)",
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
    });
  }

  new Chart(document.getElementById("chart-edge"), {
    type: "line",
    data: { labels, datasets: edgeDatasets },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: { title: { display: true, text: "mm" }, beginAtZero: true },
      },
    },
  });

  new Chart(document.getElementById("chart-angle"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Orange mean angle error (rad)",
          data: angRad,
          borderColor: "rgb(147, 51, 234)",
          backgroundColor: "rgba(147, 51, 234, 0.06)",
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: { title: { display: true, text: "rad" }, beginAtZero: true },
      },
    },
  });
  </script>
</body>
</html>
`;
}

async function main() {
  const name = process.argv[2] || "mar4qualisystrial1.tsv";
  const out = await runTrackBenchmark(name, {
    includeFrameStats: true,
    logProgress: true,
  });

  if (!out.ok || !out.frameStats) {
    console.error(out.error || "Missing frame stats");
    process.exit(1);
  }

  await fs.mkdir(OUT_DIR, { recursive: true });
  const html = buildHtml({
    overall: out.overall,
    frameStats: out.frameStats,
    dataset: out.dataset,
    assessment: out.assessment,
  });
  await fs.writeFile(OUT_HTML, html, "utf8");

  console.log("\n=== Defaults test (50 / 100 / 10 / 150 mm) ===\n");
  console.log(JSON.stringify(out.overall, null, 2));
  console.log("\n=== Chart report written ===\n");
  console.log(OUT_HTML);
  console.log("\nOpen the HTML file in a browser to view accuracy & error charts.\n");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
