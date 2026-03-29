# Motion Capture Auto-Label — Python version

A pure-Python reimplementation of the motion-capture marker auto-labelling tracker, designed for use in Jupyter notebooks.

**Terminology**

| Term | Meaning |
|------|--------|
| **Graph marker** | A fixed identity in the skeleton (`mocap-graph.json`); one index per marker in the model. |
| **Raw** | A 3D detection in the current TSV row (indices `0…n−1` are not stable identities across frames). |
| **`raw_index_by_graph_marker[i]`** | For graph marker `i`, which raw row index it uses this frame, or `None`. |
| **`reference_position_for_follow`** | Last known 3D position of marker `i` from recent solved frames — anchor for the **follow** step. |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook mocap_tracker.ipynb
```

## Files

| File | Description |
|------|-------------|
| `mocap_tracker.py` | Source notebook in `py:percent` format — editable as a plain `.py` file |
| `mocap_tracker.ipynb` | Jupyter notebook (generated from the `.py` via jupytext) |
| `requirements.txt` | Python dependencies |

## Regenerating the notebook

If you edit `mocap_tracker.py`, regenerate the `.ipynb` while **keeping existing notebook outputs**
(where possible):

```bash
cd python
jupytext --to ipynb mocap_tracker.py -o mocap_tracker.ipynb --update
```

Without `--update`, the `.ipynb` is fully replaced and cell outputs are cleared.

Or sync both ways:

```bash
jupytext --set-formats py:percent,ipynb mocap_tracker.py
jupytext --sync mocap_tracker.py
```

## Dataset

The notebook expects the TSV dataset and skeleton graph in `../data/`:

```
../data/mar4qualisystrial1.tsv
../data/mar4qualisystrial1_labelled_skeleton/mocap-graph.json
```

## Parameters

Edit the constants near the top of the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REAPPEAR_MM` | 50 | Max distance (mm) to follow a marker from its last known position |
| `TOLERANCE_MM` | 100 | Geometric constraint tolerance for fingerprint fill |
| `FOLLOW_LOOKBACK_FRAMES` | 10 | How many solved frames to search for last known position |
| `EDGE_WARNING_THRESHOLD_MM` | 150 | Flag frames with mean edge error above this (mm) |
| `MAX_FRAMES` | 0 | Limit frames for quick testing (0 = all) |

## Performance note

Pure Python runs at ~1,000 frames/s vs ~40,000 frames/s in the JS version (V8 JIT).
For the full 106k-frame dataset, expect ~2 minutes. Set `MAX_FRAMES = 5000` for quick
iteration.
