# Paper Reproduction Quick Start

This track is intentionally separate from the optimized multitask experiments.
It records the data contract and decoder protocol before a model is loaded, so
missing or incorrectly laid-out benchmark data fails early and produces an
auditable JSON report.

## 1. Install the data and evaluation extras

```bash
pip install -e ".[data,evaluation,inference]"
```

`h5py` is required when using the official Hypersim
`*.depth_meters.hdf5` files.

## 2. Prepare metric Hypersim depth

The converter keeps both the RGB training target and the original metric
depth map. Invalid metric pixels are stored as `NaN` and are excluded from
standard depth metrics.

```bash
python scripts/prepare_hypersim.py \
  --source /data/hypersim \
  --output data/hypersim_depth \
  --resolution 512 \
  --validate
```

The output contains `images/`, `targets/`, `metric_depth/`, and a
`manifest.json` with mapper parameters, source formats, sample fingerprint,
depth statistics, and RGB round-trip error.

## 3. Preflight benchmark roots

Preflight does not allocate model memory and does not require a model name.
The root may be a single dataset directory or a collection containing named
children such as `nyu/`, `diode/`, and `eth3d/`.

```bash
python scripts/evaluate_paper_benchmarks.py \
  --benchmark depth \
  --depth-root /data/benchmarks \
  --preflight-only \
  --output-json output/eval/preflight.json
```

Continue only when `preflight.ok` is `true`. A non-zero exit code means at
least one adapter found no usable pairs or the root is missing.

## 4. Run decoded benchmark metrics

```bash
python scripts/evaluate_paper_benchmarks.py \
  --benchmark all \
  --depth-root /data/benchmarks \
  --normal-root /data/benchmarks \
  --cityscapes-root /data/cityscapes \
  --model-name sana-600m-512px \
  --lora-path output/checkpoints/sana-depth \
  --num-inference-steps 20 \
  --resolution 512 \
  --output-json output/eval/paper_metrics.json \
  --output-csv output/eval/paper_metrics.csv
```

The report labels the evaluation as decoded task metrics, records the selected
dataset fingerprint, and aggregates depth/normal/semantic values over pixels
rather than averaging per-batch metrics. It must not be described as a paper
benchmark result until the real benchmark roots and matching checkpoints are
present.

## 5. Use the paper instance-decoding protocol

General deployments retain the historical adjacent grouping and adaptive
tolerance defaults. For paper-aligned instance decoding, make the protocol
explicit:

```python
from open_vision_banana.core.rgb_mapper import RGBMapper

mapper = RGBMapper.paper_reproduction()
instances = mapper.decode(rgb_prediction, task="instance")
```

The preset fixes `tau=14`, seed grouping, and disables adaptive tolerance.
The same values are recorded in both paper YAML configs.
