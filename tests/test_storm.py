import numpy as np
import pytest

from noise_storm.storm import run_stream, storm_report

D = 8


def make_blocks():
    return [
        lambda y: y,
        lambda y: 5.0 * y,
        lambda y: 0.1 * y,
    ]


def score_fn(out):
    return float(np.linalg.norm(out))


def test_run_stream():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    out = run_stream(blocks, x)
    assert out.shape == (D,)
    assert out[0] > 0


def test_storm_fragility_order():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    r = storm_report(blocks, x, score_fn, seed=0)
    assert len(r["fragility"]) == 3
    by_index = {row["index"]: row for row in r["fragility"]}
    assert r["most_fragile"] == 0
    assert r["fragility"][-1]["index"] == 2
    assert by_index[0]["mean_degradation"] > by_index[2]["mean_degradation"]
    assert all(row["mean_degradation"] >= 0 for row in r["fragility"])


def test_storm_is_deterministic():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    r1 = storm_report(blocks, x, score_fn, seed=3)
    r2 = storm_report(blocks, x, score_fn, seed=3)
    assert r1 == r2


def test_zero_write_block():
    blocks = [lambda y: np.zeros(D), lambda y: y]
    x = np.zeros(D)
    x[0] = 1.0
    r = storm_report(blocks, x, score_fn, seed=0)
    by_index = {row["index"]: row for row in r["fragility"]}
    assert by_index[0]["write_norm"] == pytest.approx(0.0)
    assert by_index[0]["mean_degradation"] == pytest.approx(0.0)


def test_empty_blocks_error():
    with pytest.raises(ValueError):
        storm_report([], np.zeros(D), score_fn)