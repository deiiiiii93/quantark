"""Stage 01 — freeze the public CFETS USD/CNY five-delta composite curve.

Live example:
    .venv/bin/python example/fx_volmodels/01_fetch_cfets_snapshot.py --tag latest

Offline conversion of previously downloaded raw replies:
    .venv/bin/python example/fx_volmodels/01_fetch_cfets_snapshot.py \
        --date 2026-07-20 --time 16:00 --raw-dir /tmp --tag sample

Only this stage accesses the network.  Downstream stages replay the stable JSON
snapshot offline.  The public curve is explicitly labelled composite/non-executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _fx_common import (  # noqa: E402
    PILLAR_DELTA,
    PILLAR_ORDER,
    SCHEMA_VERSION,
    TENOR_ORDER,
    normalise_tenor,
    spot_delta_from_strike,
    strike_from_spot_delta,
    write_json,
)


BASE_URL = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-fx"
REFERER = "https://www.chinamoney.com.cn/english/bmkycvivc/"
CURVE_ENDPOINT = f"{BASE_URL}/FoivltltyCurv"
DELTA_CURRENT_ENDPOINT = f"{BASE_URL}/FodpParam"
DELTA_HISTORY_ENDPOINT = f"{BASE_URL}/FodpParamHis"
METHODOLOGY_URL = "https://www.chinamoney.com.cn/english/bmkycvivc/"
DELTA_URL = "https://www.chinamoney.com.cn/english/bmkycvdpp/"

CURVE_SLICES = {
    "ATM": ("0", "atm"),
    "25C": ("4", "25c"),
    "25P": ("3", "25p"),
    "10C": ("2", "10c"),
    "10P": ("1", "10p"),
}


def _post_json(url: str, params: dict[str, str] | None = None) -> dict:
    query = urllib.parse.urlencode(params or {})
    request = urllib.request.Request(
        f"{url}?{query}" if query else url,
        data=b"",
        method="POST",
        headers={"Referer": REFERER, "User-Agent": "Mozilla/5.0 QuantArk-CFETS-example"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    _validate_payload(payload, url)
    return payload


def _validate_payload(payload: dict, label: str) -> None:
    if str(payload.get("head", {}).get("rep_code")) != "200":
        raise RuntimeError(f"CFETS request failed for {label}: {payload.get('head')}")
    data = payload.get("data")
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"CFETS backend error for {label}: {data['error']}")
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise RuntimeError(f"CFETS response has no records for {label}")


def _payload_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _raw_path(raw_dir: Path, date: str, suffix: str) -> Path:
    return raw_dir / f"cfets_{date}_{suffix}.json"


def _load_raw(raw_dir: Path, date: str, suffix: str) -> dict:
    path = _raw_path(raw_dir, date, suffix)
    # Historical research downloads used ``cfets_delta_DATE.json`` before the
    # suite standardized on ``cfets_DATE_delta.json``.  Accept that one legacy
    # spelling so the public replies can be replayed without renaming evidence.
    if suffix == "delta" and not path.exists():
        legacy = raw_dir / f"cfets_delta_{date}.json"
        if legacy.exists():
            path = legacy
    if not path.exists():
        raise FileNotFoundError(f"missing cached CFETS response: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_payload(payload, str(path))
    return payload


def _fetch_or_load_curves(date: str, quote_time: str, raw_dir: Path | None) -> dict[str, dict]:
    outputs = {}
    for pillar, (surface_code, suffix) in CURVE_SLICES.items():
        if raw_dir is not None:
            outputs[pillar] = _load_raw(raw_dir, date, suffix)
        else:
            outputs[pillar] = _post_json(
                CURVE_ENDPOINT,
                {
                    "lang": "EN",
                    "ccyPair": "USD.CNY",
                    "volatilitySurface": surface_code,
                    "ccyTime": quote_time,
                    "ccyDate": date,
                },
            )
    return outputs


def _fetch_or_load_delta(date: str, raw_dir: Path | None) -> dict:
    if raw_dir is not None:
        return _load_raw(raw_dir, date, "delta")
    return _post_json(
        DELTA_HISTORY_ENDPOINT,
        {
            "init": "false",
            "startDate": date,
            "endDate": date,
            "spotPriceType": "28",
            "rmbRateType": "25",
            "volatilitySurface": "ATM",
            "page": "1",
            "pageSize": "13",
        },
    )


def latest_trade_date() -> str:
    payload = _post_json(DELTA_CURRENT_ENDPOINT)
    date = payload.get("data", {}).get("showDateCN")
    if not date:
        raise RuntimeError("CFETS current delta response did not contain showDateCN")
    return str(date)


def standardise_snapshot(date: str, quote_time: str, curves: dict[str, dict], delta_payload: dict) -> dict:
    _validate_payload(delta_payload, f"delta parameters {date}")
    for pillar, payload in curves.items():
        _validate_payload(payload, f"{pillar} curve {date} {quote_time}")
        for raw_row in payload.get("records", []):
            pair = raw_row.get("ccyPair") or raw_row.get("ccyPairEN")
            if pair is not None and pair != "USD.CNY":
                raise ValueError(f"{pillar} curve contains unexpected currency pair {pair!r}")
    for raw_row in delta_payload.get("records", []):
        pair = raw_row.get("ccyPair")
        if pair is not None and pair != "USD.CNY":
            raise ValueError(f"delta parameters contain unexpected currency pair {pair!r}")
        trade_date = raw_row.get("tradeDateCn")
        if trade_date is not None and trade_date != date:
            raise ValueError(
                f"delta parameters contain trade date {trade_date!r}, expected {date!r}"
            )
    delta_rows = {normalise_tenor(row["tenor"]): row for row in delta_payload.get("records", [])}
    curve_rows = {
        pillar: {normalise_tenor(row["tenor"]): row for row in payload.get("records", [])}
        for pillar, payload in curves.items()
    }
    missing_delta = set(TENOR_ORDER) - set(delta_rows)
    if missing_delta:
        raise ValueError(f"CFETS delta response missing tenors: {sorted(missing_delta)}")

    slices = []
    for tenor in TENOR_ORDER:
        meta = delta_rows[tenor]
        maturity = float(meta["expiryDays"]) / 365.0
        forward = float(meta["strikePrice"])
        domestic_rate = float(meta["ccy2Rate"]) / 100.0
        foreign_rate = float(meta["ccy1Rate"]) / 100.0
        spot = float(meta["spotPrice"])
        pricing_foreign_rate = domestic_rate - math.log(forward / spot) / maturity
        atm_call_delta = float(meta["callDelta"])
        atm_put_delta = float(meta["putDelta"])
        foreign_discount_factor = atm_call_delta - atm_put_delta
        if not 0.0 < foreign_discount_factor <= 1.0:
            raise ValueError(
                f"tenor {tenor}: invalid foreign discount factor from ATM deltas "
                f"{foreign_discount_factor}"
            )
        effective_foreign_rate = -math.log(foreign_discount_factor) / maturity
        quotes_by_pillar = {}
        for pillar in PILLAR_ORDER:
            row = curve_rows[pillar].get(tenor)
            if row is None:
                raise ValueError(f"CFETS curve response missing {pillar} {tenor}")
            bid = float(row["bidVolatilityStr"]) / 100.0
            mid = float(row["midVolatilityStr"]) / 100.0
            ask = float(row["askVolatilityStr"]) / 100.0
            delta = PILLAR_DELTA[pillar]
            strike = forward if delta is None else strike_from_spot_delta(
                forward, mid, maturity, effective_foreign_rate, delta
            )
            if delta is not None:
                recovered = spot_delta_from_strike(
                    forward,
                    strike,
                    mid,
                    maturity,
                    effective_foreign_rate,
                    is_call=delta > 0.0,
                )
                if not math.isclose(recovered, delta, rel_tol=0.0, abs_tol=2e-12):
                    raise ValueError(
                        f"tenor {tenor} {pillar}: spot-delta strike did not round-trip"
                    )
            quotes_by_pillar[pillar] = {
                "pillar": pillar,
                "delta": delta,
                "strike": strike,
                "bid_iv": bid,
                "mid_iv": mid,
                "ask_iv": ask,
                "displayed_spread_is_zero": bool(abs(ask - bid) <= 1e-12),
                "mid_outside_displayed_band": bool(mid < bid or mid > ask),
            }
        quotes = [quotes_by_pillar[pillar] for pillar in PILLAR_ORDER]
        strikes = [quote["strike"] for quote in quotes]
        if any(a >= b for a, b in zip(strikes, strikes[1:])):
            raise ValueError(f"reconstructed strikes are not increasing for {tenor}: {strikes}")
        slices.append(
            {
                "tenor": tenor,
                "expiry_date": meta["expiryDateCn"],
                "delivery_date": None,
                "expiry_days": int(meta["expiryDays"]),
                "maturity": maturity,
                "delivery_maturity": maturity,
                "pricing_v1_delivery_assumption": "expiry_equals_delivery",
                "domestic_rate": domestic_rate,
                "foreign_rate": foreign_rate,
                "pricing_foreign_rate": pricing_foreign_rate,
                "published_forward_basis_bps": (
                    (spot * math.exp((domestic_rate - foreign_rate) * maturity) / forward - 1.0)
                    * 10_000.0
                ),
                "atm_call_delta": atm_call_delta,
                "atm_put_delta": atm_put_delta,
                "foreign_discount_factor_from_deltas": foreign_discount_factor,
                "effective_foreign_rate_for_delta": effective_foreign_rate,
                "spot": spot,
                "forward": forward,
                "swap_points_pips": float(meta["swapRate"]),
                "quotes": quotes,
            }
        )

    spots = {round(float(row["spot"]), 10) for row in slices}
    if len(spots) != 1:
        raise ValueError(f"CFETS delta rows contain inconsistent spots: {sorted(spots)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "CFETS / China Foreign Exchange Trade System public composite curve",
        "source_class": "public_composite_not_executable_history",
        "currency_pair": "USD.CNY",
        "trade_date": date,
        "quote_time": quote_time,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "spot": slices[0]["spot"],
        "delta_convention": {
            "name": "non-premium-adjusted spot delta",
            "call": "exp(-r_f*T) * N(d1)",
            "put": "exp(-r_f*T) * (N(d1)-1)",
            "atm": "ATMF / published CFETS strikePrice",
        },
        "provenance": {
            "curve_endpoint": CURVE_ENDPOINT,
            "delta_endpoint": DELTA_HISTORY_ENDPOINT,
            "methodology_url": METHODOLOGY_URL,
            "delta_parameters_url": DELTA_URL,
            "spot_price_type": "28 / Mean Quote Rate of OTC FX Spot Market",
            "cny_rate_type": "25 / Shibor-Shibor3M IRS closing curve",
            "payload_sha256": {
                "delta": _payload_digest(delta_payload),
                **{pillar: _payload_digest(payload) for pillar, payload in curves.items()},
            },
        },
        "limitations": [
            "Public bid/mid/ask are CFETS composite outputs, not node-level executable quotes.",
            "A displayed zero spread must not be interpreted as an executable zero-spread market.",
            "Official turnover evidence supports the <=1Y core much more strongly than 18M-3Y.",
        ],
        "slices": slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="CFETS trade date YYYY-MM-DD; defaults to latest available")
    parser.add_argument("--time", default="16:00", help="CFETS curve calculation time")
    parser.add_argument("--tag", default="latest", help="stable downstream artifact tag")
    parser.add_argument("--raw-dir", type=Path, help="read cached cfets_DATE_{atm,25c,25p,10c,10p,delta}.json")
    parser.add_argument("--output-dir", type=Path, help="optional artifact directory")
    args = parser.parse_args()

    date = args.date or latest_trade_date()
    curves = _fetch_or_load_curves(date, args.time, args.raw_dir)
    delta = _fetch_or_load_delta(date, args.raw_dir)
    snapshot = standardise_snapshot(date, args.time, curves, delta)

    data_dir = args.output_dir or HERE / "data"
    archive = write_json(data_dir / "snapshots" / f"cfets_usdcny_snapshot_{date.replace('-', '')}.json", snapshot)
    tagged = data_dir / f"cfets_usdcny_snapshot_{args.tag}.json"
    tagged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive, tagged)
    print(tagged)
    print(
        f"{date} {args.time} USD/CNY: {len(snapshot['slices'])} tenors, "
        f"{sum(len(row['quotes']) for row in snapshot['slices'])} five-delta nodes"
    )


if __name__ == "__main__":
    main()
