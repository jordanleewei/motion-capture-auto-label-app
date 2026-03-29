# Motion Capture Auto-Label — Python version

A pure-Python reimplementation of the motion-capture marker auto-labelling tracker, designed for use in Jupyter notebooks.

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

If you edit `mocap_tracker.py`, regenerate the `.ipynb` with:

```bash
jupytext --to ipynb mocap_tracker.py -o mocap_tracker.ipynb
```

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
