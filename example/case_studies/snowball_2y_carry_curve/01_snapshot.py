"""Stage 01 — freeze the 2026-11-05 CSI 1000 chain into data/market_20261105.json.

Inputs (the only stage that touches anything outside the repo):
- spot + IM2611/2612/2703/2706 closes + the dashboard's MO synthetic row from the
  index-basis dashboard DB (opened read-only);
- MO2709 call/put settlement prices from the CFFEX daily statistics CSV
  (downloaded, or a local copy via --cffex-csv).

Run:
    python example/case_studies/snowball_2y_carry_curve/01_snapshot.py \
        [--db /Users/fuxinyao/index-dashboard-codex/data/basis.sqlite3] [--cffex-csv PATH]

Fails closed on any missing row; never substitutes another date.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _common():
    name = "snowball_2y_cc_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HERE / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def main(argv=None) -> int:
    c = _common()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=c.DEFAULT_DB)
    parser.add_argument("--cffex-csv", type=Path, default=None,
                        help="local copy of the CFFEX CSV (default: download)")
    parser.add_argument("--out", type=Path, default=c.SNAPSHOT_PATH)
    args = parser.parse_args(argv)

    url = c.cffex_url()
    if args.cffex_csv is not None:
        payload = args.cffex_csv.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (fixed exchange URL)
            payload = response.read()
    chain = c.read_dashboard_chain(args.db)
    snapshot = c.build_snapshot(chain, payload, source_url=url)
    c.write_json(args.out, snapshot)
    print(f"spot {snapshot['spot']:.4f}")
    for f in snapshot["futures"]:
        print(f"  {f['contract']}  {f['expiry']}  close {f['price']:.1f}  settle {f['settle']:.1f}")
    pair = snapshot["option_pair"]
    print(f"  {pair['call_contract']}/{pair['put_contract']}  settle {pair['call_settle']}/{pair['put_settle']}"
          f"  F {snapshot['mo_forward']:.3f}  (dashboard {snapshot['dashboard_syn_fut']['price']:.3f})")
    print(f"  vol {snapshot['vol']['value']:.4%} at K={snapshot['vol']['strike']:g}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
