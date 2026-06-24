# Depth Mapping Algorithm Benchmark Results

> Experiment date: 2026-06-19

## 1. Roundtrip Accuracy

| Strategy | Overall Rel↓ | Near (0-5m)↓ | Far (20-100m)↓ | Max Abs↓ |
|----------|--------------|--------------|----------------|----------|
| Paper (Fixed) | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| Adaptive | 0.0045 | 0.0073 | 0.0000 | 0.0913 |
| Linear (Baseline) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 2. Runtime Speed (512×512, 50 runs avg)

| Strategy | Encode ms | Decode ms | Total ms |
|----------|-----------|-----------|----------|
| Paper (Fixed) | 18.14 | 185.36 | 203.50 |
| Adaptive | 17.01 | 255.15 | 272.16 |
| Linear (Baseline) | 0.34 | 0.69 | 1.04 |

## 3. Visual Quality (RGB Variance per Depth Range)

| Strategy | Near (0-5m)↑ | Mid (5-20m)↑ | Far (20-100m)↑ | Overall↑ |
|----------|--------------|--------------|----------------|----------|
| Paper (Fixed) | 0.099496 | 0.027561 | 0.020372 | 0.066276 |
| Adaptive | 0.099685 | 0.027496 | 0.020372 | 0.066278 |
| Linear (Baseline) | 0.000206 | 0.001932 | 0.054960 | 0.084003 |

## 4. Conclusions

1. **Roundtrip Accuracy**: Linear baseline achieves perfect roundtrip (by design), but offers no visual compression.
2. **Near-range Precision**: Linear (Baseline) performs best at 0-5m range (critical for robotics/AR).
3. **Speed**: Linear (Baseline) is the fastest overall.
4. **Visual Quality**: Linear (Baseline) provides the most visually distinguishable encoding (higher RGB variance = better for diffusion learning).
