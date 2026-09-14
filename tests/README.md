# Smoke test

Run from the `aura_release` directory:

```bash
python -m pytest -q tests/test_smoke.py
```

The tests use zeros and analytically simple tensors. They verify packaging,
tensor shapes, checkpoint metadata, parameter bounds, and the torque model;
they are not a benchmark and do not claim experimental performance.
