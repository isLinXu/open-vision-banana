# Demo

The interactive showcase for Open-Vision-Banana, plus the full results page.

- `index.html` — showcase: a scene relit in the browser from model-painted normals, a photo/output wipe for
  normals, depth and segmentation, a point cloud lifted from predicted depth, the two colour encodings, and
  the headline numbers.
- `results.html` — every run and table.
- `data/results.json`, `data/gallery.json` — the numbers and the sample list both pages render from.
- `assets/samples/` — real model outputs (SANA-1600M adapter, held-out images).

The sample inputs are held-out rows from the public datasets used for training and evaluation
(see `docs/DATASETS.md` for the source list and their licenses). They are bundled for
demonstration only; please follow each dataset's license terms if you reuse them.

The pages load `data/*.json` with `fetch()`, so open them through a local server rather than from disk:

```bash
cd demo && python3 -m http.server 8766   # then open http://127.0.0.1:8766/
```

Everything is static: no build step and no dependencies beyond two web fonts.
