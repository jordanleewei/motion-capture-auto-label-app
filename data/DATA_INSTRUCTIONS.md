# Data folder — what goes here

UPLOAD THE TSV FILE HERE AND NAME IT mar4qualisystrial1.tsv

This directory holds **motion-capture frame files** (TSV) and **skeleton graphs** (`mocap-graph.json`) used by the web app, Node benchmarks, and the Python notebook.

All **spatial units are millimetres (mm)** unless your pipeline states otherwise.

---

## TSV clip (per-frame 3D points)

Each nonempty line is **one frame**, **tab-separated**:

1. **Column 0:** frame index (integer; stored as parsed number).
2. **Column 1:** time in **seconds** (float).
3. **Remaining columns:** triples `x y z` for each raw 3D detection in **that** frame, in file order.

The parsers **strip trailing `(0, 0, 0)` triples** from the end of each row (phantom / padding markers). If the remaining coordinate count is not divisible by three, the row is skipped.

**Example (conceptual):**

```text
1	0.0	100.0	200.0	300.0	110.0	205.0	298.0	…
```

There is **no** header row. Use UTF-8 text and tab delimiters.

---

## Skeleton graph (`mocap-graph.json`)

The tracker needs a JSON graph that defines rigid bodies, baseline marker layout, segment edges between bodies, and which frame in the TSV is the **baseline** (`baselineFrameIndex`). The exact schema matches what the **Label** tab saves and what `web/tracker.js` consumes.

### Folder naming (Python notebook and this repo’s layout)

The Python notebook (`python/mocap_tracker.py`) loads a clip and then looks for the graph at:

```text
data/<tsv_basename>_labelled_skeleton/mocap-graph.json
```

So for **`mar4qualisystrial1.tsv`**, the graph path is:

```text
data/mar4qualisystrial1_labelled_skeleton/mocap-graph.json
```

If you add **`myclip.tsv`**, either create `data/myclip_labelled_skeleton/mocap-graph.json` or change the notebook’s `DATASET_NAME` / path logic to match your folder structure.

The **browser app** can load datasets and graphs through the local server; filenames do not have to follow the `_labelled_skeleton` pattern if you pick files manually in the UI.

---

## Quick checklist

| Item | Notes |
|------|--------|
| TSV | Tabs; column 0 = frame, 1 = time (s), rest = `x y z` … |
| Baseline | Baseline frame index in `mocap-graph.json` must exist in the TSV. |
| Counts | Tracker expects enough raw points to assign **graph markers**; missing detections limit assignment rate. |
| Pairing | Keep the graph that was authored for **that** clip and baseline. |

---

## See also

- Project **`README.md`** — app flow, tracker parameters, concepts.
- **`python/README.md`** — running the Jupyter / `py:percent` pipeline and regenerating `mocap_tracker.ipynb`.
