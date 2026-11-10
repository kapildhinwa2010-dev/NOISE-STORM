"""Layer-wise noise pressure testing for stream-additive models.

Blocks are stream-additive callables: state = state + block(state).
For every block, NOISE-STORM adds calibrated Gaussian noise to that
block write and measures the downstream score degradation. Blocks
whose noise collapses the output are fragile; blocks whose noise
barely matters are robust. All randomness is seeded, so runs are
reproducible. Pure NumPy.
"""

import numpy as np


def run_stream(blocks, x):
    """Run state through blocks with stream-additive semantics."""
    for b in blocks:
        x = x + b(x)
    return x


def storm_report(blocks, x, score_fn, sigma_fracs=(0.25, 0.5, 1.0), trials=4, seed=0):
    """Pressure-test every block with calibrated noise.

    For block k the noise magnitude is sigma_frac * ||write_k||, where
    write_k is the block write on the unperturbed run. The reported
    degradation is the mean absolute score change over all noise
    levels and trials.
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    x = np.asarray(x, dtype=float)
    baseline = run_stream(blocks, x)
    s0 = float(score_fn(baseline))

    rows = []
    for k, b in enumerate(blocks):
        write = b(x)
        wnorm = float(np.linalg.norm(write))
        degs = []
        for f_i, frac in enumerate(sigma_fracs):
            rng = np.random.default_rng(seed + 1009 * (k + 1) + 13 * f_i)
            for _ in range(trials):
                noise = rng.normal(size=write.shape)
                if wnorm < 1e-12:
                    noise = noise * 0.0
                else:
                    noise = noise / max(float(np.linalg.norm(noise)), 1e-12) * wnorm * frac
                stream = x
                for i, bi in enumerate(blocks):
                    w = bi(stream)
                    if i == k:
                        w = w + noise
                    stream = stream + w
                degs.append(abs(s0 - float(score_fn(stream))))
        rows.append({
            "index": k,
            "write_norm": wnorm,
            "mean_degradation": float(np.mean(degs)),
            "max_degradation": float(np.max(degs)),
        })

    ranked = sorted(rows, key=lambda r: r["mean_degradation"], reverse=True)
    return {
        "baseline_score": s0,
        "fragility": ranked,
        "most_fragile": ranked[0]["index"] if ranked else None,
    }