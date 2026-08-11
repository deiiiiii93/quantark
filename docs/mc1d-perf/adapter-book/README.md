# Cross-repo validation harness: the otc-price-adapter trade book

Prices the adapter's real 99-row desk book under two quantark trees and
compares every PV/Greek as IEEE-754 hex. Results are in
`../DECISION-2026-08-10.md` ("Production validation").

These scripts read the adapter repo but never modify it.

## Requirements

A Python environment carrying BOTH the adapter's deps (`pandas`, `openpyxl`,
`tabulate`) and quantark's. The quant-ark `.venv` qualifies once `tabulate` is
installed, and it also has numba — so the accelerated path is exercised. The
adapter's own `.venv` has no numba, which is a useful second configuration:
it exercises the NumPy fallback and must produce the same numbers.

Do NOT install numba into the adapter's venv to "make it fast" — that
silently changes which backend the desk's production runs use.

## Running

```bash
# one tree
cd /Users/fuxinyao/otc-price-adapter
PYTHONPATH=<quantark-tree> <venv>/bin/python book_probe.py \
    --model mc --label PHASE1 --out /tmp/book_mc_phase1.json

# compare two dumps
<venv>/bin/python compare_books.py /tmp/book_mc_base.json /tmp/book_mc_phase1.json

# see where a single row spends its time
PYTHONPATH=<quantark-tree> <venv>/bin/python profile_row.py GJZQ-DCRX7-20260601-OPTION-01
```

`--model` matters: the adapter's shipped config uses `pde` for autocallables,
which never enters the Monte Carlo path. Use `--model mc` to exercise the MC
engines (88 of the 99 rows), and `--model pde` to prove the production config
is undisturbed.

Never run these as heredocs — `python -` cannot parent spawn workers, so
`--workers > 1` dies at bootstrap with an opaque error.

## Timings observed 2026-08-11 (serial, one host)

| Run | Wall |
|---|---|
| full book, MC mode | ~19 min per tree |
| diverse subset ×2 (16 rows), PDE mode | ~2.3 min per tree |
| diverse subset ×1 (9 rows), MC mode | ~16 s per tree |

The full book in PDE mode is ~1.5–2 h per tree (the adapter's own
`freeze_v023_baseline.py` says so); use the subset for PDE checks.
