# NOISE-STORM

Pressure-test every block of a model with calibrated noise and map what is fragile.

NOISE-STORM adds seeded Gaussian noise to one block write at a time, at multiples of that write own norm, and measures the downstream score degradation. Blocks whose noise collapses the output are fragile; blocks whose noise barely matters are robust.

## What it does

- Per-block noise pressure at calibrated magnitudes.
- Mean and max score degradation per block, ranked by fragility.
- Seeded and reproducible: the same inputs give the same storm.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from noise_storm.storm import storm_report

blocks = [lambda y: y, lambda y: 5.0 * y, lambda y: 0.1 * y]
x = np.array([1.0, 0.0, 0.0, 0.0])
score = lambda out: float(np.linalg.norm(out))

report = storm_report(blocks, x, score)
print(report["most_fragile"])
for row in report["fragility"]:
    print(row["index"], row["mean_degradation"])
```

## How it works

1. Compute each block write on the unperturbed run and its norm.
2. For every block and noise level, add seeded Gaussian noise scaled to that norm.
3. Measure the absolute score change averaged over levels and trials.

Noise placed before an amplifier spreads through the whole downstream stack; noise placed at the end barely moves the output. That asymmetry is the fragility map.

## Honest scope

- Fragility is measured for one input and one score function.
- Blocks must follow stream-additive semantics: state = state + block(state).
- Tests use a tiny synthetic block stack, so no model download is needed.

## License

MIT