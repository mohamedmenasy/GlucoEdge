# Bundled model provenance

Both models were produced from this repository at commit 361148a, seed 0:
- `venv/bin/python -m training.train --dataset weinstock --classes 5 --epochs 20 --seed 0 --save-checkpoint`
- `venv/bin/python -m conversion.convert --dataset weinstock --classes 5`

Verified against docs/superpowers/specs/2026-07-01-litert-conversion-results.md
(float acc 0.5184 / macro recall 0.5060; int8 acc 0.5866 / macro recall 0.4135).

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `eb96c7e66ab13a41a6bacdcbeb00c0c089d63f67dd1a45c2947a48ff301b08e6` | 17352 |
| `trend_int8.tflite` | `0dc3538704c89f055fb7437e50a5bf42c81a64853a4b527d49c206569f8bea29` | 11976 |

Not a medical device. Trained only on the public GlucoBench benchmark.
