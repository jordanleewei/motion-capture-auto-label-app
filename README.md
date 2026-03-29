# Motion Capture Auto Label App

This project auto-assigns *logical* motion-capture marker identities across a motion clip using a skeleton graph (rigid bodies + inter-body “segment” edges). It also visualizes assignments in a browser and can run a backend benchmark.

## Quick start (run the project)

0. Prepare dataset file:
   - Add data into the `data/` folder and rename as `mar4qualisystrial1.tsv` (renamed from Google Drive data).
1. Install dependencies:
   - `npm install`
2. Start the local server:
   - `npm start`
3. Open your browser:
   - `http://localhost:3000`
4. In the app:
   - Load a TSV dataset (from `data/`)
   - Load or create/save a matching `mocap-graph.json`
   - Go to **Track** or **Live** and click **Run tracker**

Optional backend benchmark:
- `npm run track-bench -- mar4qualisystrial1.tsv`

## Repository layout

- `web/app.js`: Browser UI. Loads TSV frames + `mocap-graph.json`, runs the tracker, and renders the assigned logical marker positions and edges.
- `web/tracker.js`: The shared tracking algorithm (`runMultiFrameTracker`) used by both the browser UI and Node benchmark scripts.
- `web/index.html`: UI (Track / Live tabs) and parameter controls.
- `web/styles.css`: Basic styling for the UI.
- `server.mjs`: Small local HTTP server that serves the UI and provides dataset/skeleton/benchmark endpoints.
- `track-benchmark.mjs`: Node script that runs the tracker over a dataset and prints performance + diagnostics.
- `test-api.mjs`: Quick sanity checks for the local API endpoints.
- `data/`: Local input datasets and saved skeleton graphs.

## Key concepts

### Frames / raw detections
Each TSV file is parsed into `frames`, where each frame contains:
- `time`: timestamp (if present in the TSV)
- `points`: an array of raw 3D detections for that frame (`Vec3[]`)

### Skeleton graph (`mocap-graph.json`)
The saved skeleton graph defines:
- Which logical marker indices belong to each rigid body (in baseline order)
- Which inter-rigid marker *pairs* are constrained by blue “segment edges”
- Which intra-rigid marker *pairs* are constrained by orange “rigid edges” (derived from rigid-body membership)

When the tracker assigns logical markers for a frame, the UI draws:
- **Orange lines**: between logical markers inside the same rigid body (derived from rigid membership)
- **Blue lines**: between logical markers across rigid bodies (driven by `segmentEdgesBetweenRigidBodies`)

If assignments swap identity (e.g., elbow gets assigned where hip should be), the UI will correctly draw “blue shoulder–elbow” as shoulder–hip physically, which looks like incorrect blue connections.

### Tracker parameters
The solver consumes these options (units are millimeters unless noted):
- `reappearMm`: how far a raw detection may “reappear” from a recent logical position
- `followLookbackFrames`: how many past (forward) or future (backward) frames to search for the last known position (1–30)
- `toleranceMm`: geometric tolerance for fingerprint fill and rigid-body rescue
- `edgeWarningThresholdMm`: optional; if set, the run report flags frames whose mean orange or blue edge-length error exceeds this (does not change assignments)
- `yieldEvery`: how often the tracker yields progress to keep the UI responsive (browser only)

Current code defaults (when options are omitted):
- `reappearMm`: `50`
- `followLookbackFrames`: `10`
- `toleranceMm`: `100`
- `edgeWarningThresholdMm`: `150` (benchmark); UI may use empty field to disable flagging
- `yieldEvery`: `2`

## How tracking works (high level)

`web/tracker.js` implements `runMultiFrameTracker(frames, graph, opts)`:

1. **Seed baseline**
   - Uses `graph.baselineFrameIndex` to read baseline 3D positions for all logical markers.
2. **Forward pass + backward pass**
   - Runs frame-by-frame assignment from the baseline to the end (`forward`)
   - Runs frame-by-frame assignment from the baseline backward in time (`backward`)
3. **Per-frame assignment**
   - **Follow recent frames**: for each logical marker, find the nearest raw point to the most recent known position within up to `followLookbackFrames` already-solved frames, within `reappearMm`.
   - **Fingerprint fill**: fills remaining logical slots by checking whether candidate raw points satisfy the graph’s constrained edges within `toleranceMm`.
   - **Rigid-body rescue**: when an entire rigid body is missing, tries to recover it by matching a small subset of raw points to the baseline rigid-body shape (then re-runs fingerprint fill).
4. **Edge quality (report only)**
   - After the run, analytics compute mean orange/blue edge-length errors per frame. Optional `edgeWarningThresholdMm` flags suspicious frames in the run-report plots (no label stripping in the tracker).

The tracker returns:
- `result.perFrame[frameIndex]`: for each logical marker, the assigned raw index (`rawForLogical`) and computed logical position (`logicalPos`)
- `result.stats`: counts for matched/missing assignments
- `result.analytics` + `result.diagnostics`: summary error/assignment-rate metrics and narrative failure-mode hints

## Browser usage

1. Start the server:
   - `npm start`
2. Open the UI in your browser.
3. Load a dataset TSV and a saved `mocap-graph.json` skeleton.
4. Adjust parameters on the **Track** (full file) or **Live** tab.
5. Click **Run tracker**.

### Relevant UI controls
Track / Live group inputs into three phases:
1. **Follow recent frames**: `reappearMm`, `followLookbackFrames`
2. **Fill unassigned**: `toleranceMm`
3. **Edge quality (plots only)**: optional warn threshold for flagging frames in the run report

## Benchmark / CLI usage

Run a backend benchmark (Node):

```bash
npm run track-bench -- mar4qualisystrial1.tsv
```

This loads:
- `data/<stem>.tsv`
- `data/<stem>_labelled_skeleton/mocap-graph.json`

It prints summary performance and diagnostics.

To override tolerance from the CLI:

```bash
node track-benchmark.mjs mar4qualisystrial1.tsv --tol=15
```

## Common failure mode: “blue edges connecting wrong body parts”

Blue edges are *not* decided by motion semantics in the UI. They are drawn based on:
- which rigid groups a logical marker index belongs to (via the saved graph), and
- which logical markers were assigned to which raw detections for the frame.

So if the solver ever assigns the elbow logical marker to a hip raw point, the UI will still draw the blue “elbow-to-shoulder constraint” line, but physically it will appear as shoulder-to-hip.

Typical causes are:
- a too-large `toleranceMm` (loose matching allows identity swaps),
- too-small `reappearMm` or too-small lookback (dropping markers too aggressively triggers rescue/incorrect recovery),
- heavy occlusion / missing raw detections,
- fast motion where a single-frame follow distance is not enough (increase lookback / reappearance).

## Notes for colleagues

If you want to reproduce debugging behavior and visuals, share:
- `web/tracker.js` (algorithm),
- `web/app.js` (UI integration + rendering rules),
- `web/index.html` (parameter controls/defaults),
- the specific `data/<stem>_labelled_skeleton/mocap-graph.json` that defines the blue-edge constraints.

## Testing steps

1. Confirm the local server is running:
   - `npm start`
   - Open `http://127.0.0.1:8765/`
2. Validate API endpoints quickly:
   - `npm run test-api`
3. Run backend benchmark on the main dataset:
   - `npm run track-bench -- mar4qualisystrial1.tsv`
4. In the browser app:
   - Load dataset `mar4qualisystrial1.tsv`
   - Load skeleton from `mar4qualisystrial1_labelled_skeleton/mocap-graph.json`
   - Go to **Track** and set (defaults are fine):
     - Phase 1: `Reappearance radius (mm)`: `50`, `Look back up to N frames`: `10`
     - Phase 2: `Tolerance (mm)`: `100`
     - Phase 3: `Warn if mean edge error exceeds (mm)`: `150` (or clear the field to disable red-dot flags)
   - Click **Run tracker**
5. Verify behavior:
   - Blue edges only appear for intended logical connections from the saved graph
   - Run report shows orange/blue edge error charts; red dots mark frames flagged by the phase-3 threshold

