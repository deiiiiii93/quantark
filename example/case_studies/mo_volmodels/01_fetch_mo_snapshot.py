"""Stage 01 — fetch a CSI 1000 (MO / 中证1000股指期权) option snapshot via AKShare.

RUN WITH THE AKSHARE INTERPRETER (quantark's .venv has no akshare):
    /opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py

Writes data/mo_snapshot_YYYYMMDD.json and data/mo_snapshot_latest.json.
Stages 02-06 replay that snapshot offline under .venv/bin/python.

Data source: Sina (option_cffex_zz1000_*_sina, stock_zh_index_spot_sina). EastMoney
(_em) endpoints are frequently blocked from this network, so we use Sina throughout.
MO options are European, cash-settled on the CSI 1000 index — ideal for Dupire/Heston/SLV.
"""
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

try:
    import akshare as ak
except ImportError:
    sys.exit(
        "akshare not found in this interpreter. Run stage 01 with the anaconda python:\n"
        "  /opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py"
    )

DATA = Path(__file__).resolve().parent / "data"

# Chinese column names in option_cffex_zz1000_spot_sina (verified live 2026).
COL_STRIKE = "行权价"
COL_CALL_LAST, COL_CALL_BID, COL_CALL_ASK, COL_CALL_OI = (
    "看涨合约-最新价", "看涨合约-买价", "看涨合约-卖价", "看涨合约-持仓量",
)
COL_PUT_LAST, COL_PUT_BID, COL_PUT_ASK, COL_PUT_OI = (
    "看跌合约-最新价", "看跌合约-买价", "看跌合约-卖价", "看跌合约-持仓量",
)


def _third_friday(year: int, month: int) -> date:
    fridays = [d for d in range(1, 29) if date(year, month, d).weekday() == 4]
    return date(year, month, fridays[2])


def _expiry_from_code(code: str) -> tuple[str, float]:
    """CFFEX month code 'mo2608' -> (expiry ISO date, T in years, ACT/365)."""
    yy, mm = 2000 + int(code[2:4]), int(code[4:6])
    exp = _third_friday(yy, mm)
    T = max((exp - date.today()).days / 365.0, 1.0 / 365.0)
    return exp.isoformat(), T


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _index_spot() -> float:
    df = ak.stock_zh_index_spot_sina()
    row = df[df["代码"].astype(str).str.contains("000852")]
    if row.empty:
        raise RuntimeError("could not find 000852 (CSI 1000) in stock_zh_index_spot_sina")
    return float(row["最新价"].iloc[0])


def main() -> None:
    spot = _index_spot()
    contracts = ak.option_cffex_zz1000_list_sina()
    months = list(contracts.values())[0] if isinstance(contracts, dict) else list(contracts)

    expiries = []
    any_liquidity = False
    for m in months:
        try:
            df = ak.option_cffex_zz1000_spot_sina(symbol=m)
        except Exception as e:  # noqa: BLE001 — one bad month must not abort the fetch
            print(f"  skip {m}: fetch failed ({e!r})")
            continue
        exp_date, T = _expiry_from_code(m)
        quotes = []
        for _, row in df.iterrows():
            strike = _num(row.get(COL_STRIKE))
            if strike is None or strike <= 0:
                continue
            for kind, c_last, c_bid, c_ask, c_oi in (
                ("C", COL_CALL_LAST, COL_CALL_BID, COL_CALL_ASK, COL_CALL_OI),
                ("P", COL_PUT_LAST, COL_PUT_BID, COL_PUT_ASK, COL_PUT_OI),
            ):
                last = _num(row.get(c_last))
                if last is None or last <= 0:
                    continue
                bid, ask = _num(row.get(c_bid)), _num(row.get(c_ask))
                oi = int(_num(row.get(c_oi)) or 0)
                any_liquidity = any_liquidity or oi > 0
                quotes.append(
                    {
                        "strike": strike,
                        "type": kind,
                        "last": last,
                        "bid": bid,
                        "ask": ask,
                        "volume": oi,  # OI is the available liquidity proxy (no traded-vol column)
                        "oi": oi,
                    }
                )
        if quotes:
            expiries.append({"expiry_date": exp_date, "T_years": T, "quotes": quotes})
            print(f"  {m}: {len(quotes)} quotes, expiry {exp_date}, T={T:.3f}")

    snap = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "market_open": any_liquidity,
        "underlying": {"code": "000852.SH", "spot": spot},
        "expiries": expiries,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    (DATA / f"mo_snapshot_{stamp}.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False))
    (DATA / "mo_snapshot_latest.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False))
    print(f"wrote snapshot: spot={spot:.2f}, {len(expiries)} expiries -> data/mo_snapshot_{stamp}.json")
    if not any_liquidity:
        print("NOTE: no open interest seen (market likely closed) — IVs use last-price mids.")


if __name__ == "__main__":
    main()
