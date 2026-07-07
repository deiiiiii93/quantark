"""Stage 06 — render the full calibration study as self-contained HTML lecture material.

Run: .venv/bin/python example/mo_volmodels/06_lecture.py [--tag latest|sample]

Reads every artifact produced by stages 02-05 (the IV surface, the local-vol / Heston /
SLV reprice results, the calibrated parameters, and the PNG figures), plus optional
stage-07/08/09 exotic and hedging artifacts when present, and weaves them into a single course-style
HTML page: theory, the recovered market numbers, the model math, the calibration
results, and the honest lessons. No external assets (figures are embedded as base64),
so the page opens offline in any browser.
"""
import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def b64(png: Path) -> str:
    if not png.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def fig(png: Path, caption: str) -> str:
    src = b64(png)
    if not src:
        return f"<figure><figcaption>[missing figure: {png.name}]</figcaption></figure>"
    return f"<figure><img src='{src}' alt='{caption}'><figcaption>{caption}</figcaption></figure>"


def bs_flat_baseline(per_expiry) -> float:
    """RMSE of a single-flat-vol 'no smile' model: each expiry priced at its own ATM vol."""
    sq = []
    for p in per_expiry:
        ks = np.array([k for k, _ in p["points"]])
        vs = np.array([v for _, v in p["points"]])
        atm = float(vs[np.argmin(np.abs(ks - p["forward"]))])
        sq.extend((vs - atm).tolist())
    return float(np.sqrt(np.mean(np.square(sq)))) if sq else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()
    t = args.tag
    surface = json.loads((HERE / f"data/mo_iv_surface_{t}.json").read_text())
    lv = json.loads((HERE / f"data/mo_reprice_localvol_{t}.json").read_text())
    he = json.loads((HERE / f"data/mo_calib_heston_{t}.json").read_text())
    slv = json.loads((HERE / f"data/mo_reprice_slv_{t}.json").read_text())

    s0 = surface["s0"]
    pe = surface["per_expiry"]
    fetched = surface.get("fetched_at", "n/a")
    bsflat = bs_flat_baseline(pe)
    smoothing = lv.get("target_smoothing") or he.get("target_smoothing") or slv.get("target_smoothing") or {"method": "none"}
    smooth_method = smoothing.get("method", "none")
    smooth_rmse = smoothing.get("raw_grid_rmse_iv")
    smooth_adj = smoothing.get("calendar_adjusted_nodes", 0)
    smooth_text = (
        f"The model-calibration target is a <b>SABR-smoothed, calendar-projected</b> IV grid "
        f"(beta={smoothing.get('beta', 1.0):.1f}); it differs from the raw rectangular grid by "
        f"<b>{smooth_rmse*100:.2f} vol-points</b> RMSE and adjusted {smooth_adj} total-variance nodes."
        if smooth_method == "sabr_calendar_projected" and smooth_rmse is not None
        else "The model-calibration target is the raw IV grid."
    )

    # Optional barrier-exotic artifact (stage 07); the section is omitted if absent.
    barrier_path = HERE / f"data/mo_barrier_{t}.json"
    barrier = json.loads(barrier_path.read_text()) if barrier_path.exists() else None
    # Optional Snowball exotic artifact (stage 08); omitted if absent.
    snowball_path = HERE / f"data/mo_snowball_{t}.json"
    snowball = json.loads(snowball_path.read_text()) if snowball_path.exists() else None
    # Optional delta-hedging artifact (stage 09); omitted if absent.
    hedge_path = HERE / f"data/mo_hedging_{t}.json"
    hedge = json.loads(hedge_path.read_text()) if hedge_path.exists() else None

    rmse_rows = [
        ("Black-Scholes (flat ATM vol)", bsflat, "no smile — baseline"),
        ("Local Volatility (Dupire)", lv["overall_rmse_iv"], "PDE reprice of the smoothed no-arb target"),
        ("Heston (stochastic vol)", he["overall_rmse_iv"], "5 params, fit to the same smoothed target"),
        ("Heston-SLV (leverage)", slv["overall_rmse_iv"], "smile-consistent dynamics; PDE bias on vanillas"),
    ]

    # comparison CSV
    with (HERE / f"data/comparison_summary_{t}.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "overall_rmse_iv_volpts", "note"])
        for name, r, note in rmse_rows:
            w.writerow([name, f"{r*100:.4f}", note])

    p = he["params"]
    # per-expiry recovered market table
    mkt_rows = "".join(
        f"<tr><td>{e['expiry_date']}</td><td>{e['T']:.3f}</td><td>{e['r']*100:+.2f}%</td>"
        f"<td>{e['q']*100:+.2f}%</td><td>{e['forward']:.1f}</td><td>{len(e['points'])}</td></tr>"
        for e in pe
    )
    # per-expiry RMSE table (LV / Heston / SLV)
    lv_by = {round(x["T"], 3): x["rmse_iv"] for x in lv["per_expiry"]}
    he_by = {round(x["T"], 3): x["rmse_iv"] for x in he["per_expiry"]}
    slv_by = {round(x["T"], 3): x["rmse_iv"] for x in slv["per_expiry"]}
    perT_rows = "".join(
        f"<tr><td>{e['T']:.3f}</td>"
        f"<td>{lv_by.get(round(e['T'],3), float('nan'))*100:.2f}</td>"
        f"<td>{he_by.get(round(e['T'],3), float('nan'))*100:.2f}</td>"
        f"<td>{slv_by.get(round(e['T'],3), float('nan'))*100:.2f}</td></tr>"
        for e in pe
    )
    rmse_table = "".join(
        f"<tr><td>{name}</td><td class='num'>{r*100:.2f}</td><td>{note}</td></tr>"
        for name, r, note in rmse_rows
    )

    feller_ok = he["feller"] >= 1.0

    # ---- Exotic sections (only if stages 07/08 ran) ----
    barrier_section = ""
    barrier_toc = ""
    snowball_block = ""
    hedge_section = ""
    hedge_toc = ""
    if snowball is not None:
        sp_s = snowball["spec"]
        order_s = ["BSM (flat ATM)", "Local Vol", "Heston QE", "SLV", "SLV QE"]

        def _srow(name):
            m = snowball["models"][name]
            mc = m.get("mc_pct_initial", float("nan"))
            se = m.get("mc_stderr", float("nan"))
            notional = sp_s.get("notional", sp_s.get("s0", 1.0)) or 1.0
            se_pct = 100.0 * se / notional if se == se else float("nan")
            pde = m.get("pde_pct_initial")
            pde_txt = "n/a" if pde is None else f"{pde:.3f}%"
            gap = m.get("gap_pct")
            gap_txt = "n/a" if gap is None else f"{gap:.2f}%"
            ko = m.get("ko_probability")
            ko_txt = "n/a" if ko is None else f"{ko*100:.1f}%"
            ok = m.get("cross_check")
            tick = (
                "n/a" if ok is None else
                ("<span style='color:#1a7f37;font-weight:700'>&#10003;</span>" if ok
                 else "<span style='color:#b5432f;font-weight:700'>&#8211;</span>")
            )
            se_txt = f" &plusmn;{se_pct:.3f}%" if se_pct == se_pct else ""
            return (f"<tr><td>{name}</td><td class='num'>{mc:.3f}%{se_txt}</td>"
                    f"<td class='num'>{pde_txt}</td><td class='num'>{gap_txt}</td>"
                    f"<td class='num'>{ko_txt}</td><td style='text-align:center'>{tick}</td></tr>")

        rows_s = "".join(_srow(name) for name in order_s if name in snowball["models"])
        lv_s = snowball["models"].get("Local Vol", {}).get("mc_pct_initial")
        slv_qe_s = snowball["models"].get("SLV QE", {}).get("mc_pct_initial")
        spread_s = (
            slv_qe_s - lv_s
            if lv_s is not None and slv_qe_s is not None
            else float("nan")
        )
        snowball_block = f"""
<h3>Snowball autocallable: same smile, different path dynamics</h3>
<p>Stage 08 prices a <b>standard Snowball</b> (strike {sp_s['strike']:.0f},
KO {sp_s['ko_barrier']:.0f}, KI {sp_s['ki_barrier']:.0f}, {sp_s['ko_monitoring']} KO,
{sp_s['ki_monitoring']} KI, T&nbsp;=&nbsp;{sp_s['T']:.2f}, <b>principal {sp_s.get('principal', 'included')}</b>)
under the same MO surface. The
MC column is expressed as a percentage of initial notional. BSM, Local Vol, Heston QE,
and SLV have PDE diagnostics where a PDE analogue exists; <b>SLV QE is MC-only</b> because
QE is a Monte Carlo variance discretization, not a PDE scheme. The 2D Heston/SLV PDE
rows use a Snowball-aware ADI grid ({sp_s.get('pde_grid_policy', 'critical levels pinned')})
because a single KO-focused grid is too coarse for the KI/downside transition in this
principal-excluded payoff.</p>
<table><thead><tr><th>model</th><th>MC PV (% initial)</th><th>PDE PV</th>
<th>|gap| / MC</th><th>KO probability</th><th>cross-check</th></tr></thead>
<tbody>{rows_s}</tbody></table>
{fig(HERE / f"data/plots/08_snowball_{t}.png", "Snowball PV by model (MC vs PDE where available).")}
<div class="callout key"><b>Autocallable lesson.</b> A Snowball is even more path-dependent than
the single barrier: KO timing, continuous KI, stochastic variance, and leverage all interact.
On this fixture the standalone SLV QE MC row differs from Local Vol by {spread_s:+.2f} percentage
points of initial notional, while preserving the same vanilla smile input. That gap is the economic
content of forward-vol dynamics.</div>
"""

    if barrier is not None:
        sp = barrier["spec"]
        order = ["BSM (flat ATM)", "Local Vol", "Heston", "SLV"]

        def _brow(name):
            m = barrier["models"][name]
            mc = m.get("mc", float("nan")); se = m.get("mc_stderr", float("nan"))
            pde = m.get("pde", float("nan")); gap = m.get("gap", abs(mc - pde))
            ok = m.get("cross_check")
            tick = ("<span style='color:#1a7f37;font-weight:700'>&#10003;</span>" if ok
                    else "<span style='color:#b5432f;font-weight:700'>&#8211;</span>")
            se_txt = f" &plusmn;{se:.3f}" if se == se else ""
            return (f"<tr><td>{name}</td><td class='num'>{mc:.3f}{se_txt}</td>"
                    f"<td class='num'>{pde:.3f}</td><td class='num'>{gap:.3f}</td>"
                    f"<td style='text-align:center'>{tick}</td></tr>")
        rows_b = "".join(_brow(name) for name in order if name in barrier["models"])
        mc_lv = barrier["models"]["Local Vol"]["mc"]
        mc_he = barrier["models"]["Heston"]["mc"]
        spread = (mc_he - mc_lv) / mc_lv * 100 if mc_lv else float("nan")
        toc_tail = (
            " and a Snowball autocallable" if snowball_block else ""
        )
        barrier_toc = (f'<li><a href="#s5">Exotics: where models diverge</a> — an up-and-out call'
                       f'{toc_tail}, MC reference prices with PDE diagnostics</li>')
        barrier_section = f"""
<h2 id="s5">5 &nbsp; Exotics: where the models diverge</h2>
<h3>Barrier option: forward-vol dynamics beyond the terminal smile</h3>
<p>Every model so far was tuned to the <em>same</em> vanilla smile, and on vanillas they largely
agree. A <b>barrier option</b> breaks that tie: its payoff depends on the <em>path</em>, hence on the
<b>forward-volatility dynamics</b> — how volatility evolves after the spot moves — which the terminal
smile alone does not pin down. We price a <b>reverse up-and-out call</b> (strike {sp['strike']:.0f},
barrier {sp['barrier']:.0f} &asymp; 110% spot, {sp['monitoring']} monitoring —
{sp.get('n_obs', 0)} weekly observations, T&nbsp;=&nbsp;{sp['T']:.2f}) under all four models, each by
Monte Carlo and PDE using the new <code>quantark</code> barrier engines. The <b>MC column is the
reference</b> for the model-divergence story. The PDE column is an independent diagnostic: MC and
PDE run on the <em>same</em> {sp.get('n_steps', '?')}-step time grid, so both knock the barrier out on
the identical weekly dates; the MC leg carries its standard error and the <b>cross-check</b> passes
only when |MC&nbsp;&minus;&nbsp;PDE| falls inside max(2%&nbsp;of&nbsp;price,&nbsp;3&nbsp;s.e.) —
agreement is <em>measured</em>, not asserted.</p>
<table><thead><tr><th>model</th><th>MC price (&plusmn;1 s.e.)</th><th>PDE price</th>
<th>|gap|</th><th>cross-check</th></tr></thead>
<tbody>{rows_b}</tbody></table>
{fig(HERE / f"data/plots/07_barrier_{t}.png", "Up-and-out call price by model (MC vs PDE). Same vanilla smile, different barrier prices.")}
<div class="callout key"><b>The payoff of the whole course.</b> Local vol prices the up-and-out close to
flat Black-Scholes, but the <b>stochastic-vol models price it about {spread:+.0f}% differently</b> — a
first-order economic gap produced entirely by dynamics the vanilla smile cannot see. This is why the
model you calibrate is only half the decision: for a path-dependent exotic you must also choose the
<em>dynamics</em>. The PDE diagnostics should be read honestly: after switching the local-vol target
from a floored raw Dupire surface to the SABR-smoothed no-arb grid, the LV row becomes a genuine
MC/PDE cross-check rather than a data-repair warning. The robust conclusion remains the
<b>MC spread across models</b>, now with PDE confirmation where the numerical method is stable.
This is the concrete reason SLV exists: it keeps the market smile
<em>and</em> a realistic, tunable forward-vol process for exactly these products.</div>
{snowball_block}
"""
    elif snowball_block:
        barrier_toc = ('<li><a href="#s5">Exotics: where models diverge</a> — a Snowball '
                       'autocallable under BSM, LV, Heston QE, SLV, and SLV QE</li>')
        barrier_section = f"""
<h2 id="s5">5 &nbsp; Exotics: where the models diverge</h2>
{snowball_block}
"""
    if hedge is not None:
        hs = hedge["spec"]
        order_h = ["BSM (flat vol)", "Local Vol", "Heston", "SLV"]

        def _hrow(name):
            m = hedge["models"][name]
            pnl_pct = 100.0 * m["total_residual_pnl"] / hs["s0"]
            return (
                f"<tr><td>{name}</td>"
                f"<td class='num'>{m['initial_delta']:+.4f}</td>"
                f"<td class='num'>{-m['initial_delta']:+.4f}</td>"
                f"<td class='num'>{m['total_rebalance_units']:.3f}</td>"
                f"<td class='num'>{m['total_residual_pnl']:+.3f} ({pnl_pct:+.3f}%)</td></tr>"
            )

        rows_h = "".join(_hrow(name) for name in order_h if name in hedge["models"])
        lv_turn = hedge["models"].get("Local Vol", {}).get("total_rebalance_units")
        slv_turn = hedge["models"].get("SLV", {}).get("total_rebalance_units")
        turn_spread = (
            slv_turn - lv_turn
            if lv_turn is not None and slv_turn is not None
            else float("nan")
        )
        hedge_toc = (
            '<li><a href="#s6">Delta-neutral hedging</a> — BSM, LV, Heston, and SLV '
            'produce different hedge ratios on the same spot path</li>'
        )
        hedge_section = f"""
<h2 id="s6">6 &nbsp; Delta-neutral hedging: same smile, different hedge behavior</h2>
<p>Stage 09 takes <b>BSM flat vol</b> plus the calibrated <b>LV, Heston, and SLV</b> models and asks a desk question:
if we are long one ATM European call (strike {hs['strike']:.0f}, T&nbsp;=&nbsp;{hs['T']:.2f}) and
rebalance to <b>delta neutral</b> on the same deterministic down-up spot path, how do the hedge
ratios differ? At each rebalance the stock/futures hedge is <code>-model_delta</code>. The residual
PnL is computed as <code>dV - previous_delta * dS</code>; funding, transaction costs, and model
recalibration are deliberately excluded so the chart isolates model-delta behavior. The BSM row
uses one flat ATM implied volatility ({hs.get('flat_vol', float('nan'))*100:.2f}%).</p>
<table><thead><tr><th>model</th><th>initial delta</th><th>initial hedge</th>
<th>total rebalanced units</th><th>cum residual PnL</th></tr></thead>
<tbody>{rows_h}</tbody></table>
{fig(HERE / f"data/plots/09_delta_hedging_{t}.png", "Delta-neutral hedge units and residual PnL under BSM, LV, Heston, and SLV.")}
<div class="callout key"><b>Hedging lesson.</b> A single vanilla smile does not imply a single
hedge book. BSM gives the flat-vol baseline; Local vol gives a deterministic smile-surface delta; Heston changes the hedge through
stochastic-variance convexity; SLV keeps the smile-consistent local response while importing
Heston-style forward-vol dynamics. On this path, SLV rebalances {turn_spread:+.2f} more
underlying units than Local Vol, showing that model choice changes not only price but also
hedge inventory, turnover, and residual PnL under a delta-neutral strategy.</div>
"""
    # Loud banner if this render is the synthetic test fixture, never the real deliverable.
    fixture_banner = "" if t == "latest" else (
        '<div style="background:#b5432f;color:#fff;padding:.7rem 1rem;border-radius:8px;'
        'margin:1rem 0;font-weight:600">⚠ SYNTHETIC TEST FIXTURE (tag=' + t + ') — arbitrage-free '
        'sample data for automated tests, NOT real market data. The real-data lecture is '
        '<code style="color:#fff">mo_volmodels_lecture_latest.html</code> (run stage 01 to refresh).</div>'
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volatility Model Calibration on MO Options — A Practical Lecture</title>
<style>
:root {{ --ink:#1a2230; --muted:#5a6577; --line:#e2e6ee; --accent:#1E3A5F; --accent2:#b5432f;
        --bg:#ffffff; --card:#f7f9fc; --code:#0d1b2a; }}
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;
       color:var(--ink); background:var(--bg); margin:0; line-height:1.62; }}
.wrap {{ max-width:900px; margin:0 auto; padding:2.5rem 1.4rem 5rem; }}
h1 {{ font-size:2rem; line-height:1.2; margin:.2rem 0 .4rem; }}
h2 {{ font-size:1.4rem; margin:2.6rem 0 .6rem; padding-top:1rem; border-top:2px solid var(--line); color:var(--accent); }}
h3 {{ font-size:1.08rem; margin:1.5rem 0 .3rem; color:var(--accent2); }}
.lede {{ color:var(--muted); font-size:1.05rem; }}
.meta {{ font-size:.85rem; color:var(--muted); background:var(--card); border:1px solid var(--line);
         border-radius:8px; padding:.7rem 1rem; margin:1rem 0 0; }}
p, li {{ font-size:.98rem; }}
code {{ font-family:"SF Mono",Menlo,Consolas,monospace; background:#eef1f6; padding:.05rem .3rem; border-radius:4px; font-size:.9em; }}
.eq {{ background:var(--code); color:#e6edf3; border-radius:8px; padding:.8rem 1.1rem; margin:.8rem 0;
       font-family:"SF Mono",Menlo,Consolas,monospace; font-size:.92rem; overflow-x:auto; }}
.eq .c {{ color:#7d8aa0; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.9rem; }}
th,td {{ border:1px solid var(--line); padding:.42rem .6rem; text-align:left; }}
th {{ background:var(--accent); color:#fff; font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr:nth-child(even) td {{ background:var(--card); }}
figure {{ margin:1.2rem 0; text-align:center; }}
figure img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; }}
figcaption {{ font-size:.82rem; color:var(--muted); margin-top:.4rem; }}
.callout {{ border-left:4px solid var(--accent2); background:#fbf2ef; padding:.8rem 1.1rem; border-radius:0 8px 8px 0; margin:1.1rem 0; }}
.callout.key {{ border-color:var(--accent); background:#eef3f9; }}
.callout b {{ color:var(--accent2); }}
.callout.key b {{ color:var(--accent); }}
.toc {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:1rem 1.3rem; margin:1.4rem 0; }}
.toc ol {{ margin:.3rem 0; padding-left:1.2rem; }}
.toc a {{ color:var(--accent); text-decoration:none; }}
.pill {{ display:inline-block; background:var(--accent); color:#fff; border-radius:20px; padding:.12rem .7rem; font-size:.75rem; margin-right:.3rem; }}
footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); font-size:.82rem; color:var(--muted); }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e6edf3; --muted:#9aa7b8; --line:#2b3648; --accent:#8fb4e0; --accent2:#e8927c;
           --bg:#0d131c; --card:#141d2b; --code:#060b12; }}
  code {{ background:#1c2634; }} .callout {{ background:#1b1512; }} .callout.key {{ background:#111a26; }}
}}
</style></head><body><div class="wrap">
{fixture_banner}
<h1>Volatility Model Calibration on MO Index Options</h1>
<p class="lede">A hands-on study: recovering an implied-volatility surface from live CSI&nbsp;1000
(中证1000, <code>000852.SH</code>) option quotes, then calibrating three volatility models —
<b>Dupire Local Volatility</b>, <b>Heston stochastic volatility</b>, and <b>Heston Stochastic-Local
Volatility (SLV)</b> — with an honest account of what each model can and cannot do.</p>
<div class="meta">
Underlying: <b>CSI&nbsp;1000 index (000852.SH)</b> &middot; spot <b>{s0:.1f}</b> &middot;
{len(pe)} expiries (T = {min(e['T'] for e in pe):.2f}–{max(e['T'] for e in pe):.2f} yr) &middot;
snapshot {fetched} &middot; data via AKShare (Sina) &middot; models from <code>quantark.volmodels</code>
</div>

<div class="toc"><b>Contents</b>
<ol>
<li><a href="#s1">Market data &amp; the implied-vol surface</a> — put-call parity, OTM filter, inversion</li>
<li><a href="#s2">Dupire Local Volatility</a> — the forward equation and butterfly arbitrage</li>
<li><a href="#s3">Heston Stochastic Volatility</a> — CIR variance, the Feller condition, identification</li>
<li><a href="#s4">Heston-SLV</a> — the leverage function and which model for which product</li>
{barrier_toc}
{hedge_toc}
<li><a href="#s7">Model comparison &amp; takeaways</a></li>
<li><a href="#s8">Reproducing this study</a></li>
</ol></div>

<h2 id="s1">1 &nbsp; Market data and the implied-vol surface</h2>
<p>MO options are <b>European, cash-settled</b> on the CSI&nbsp;1000 index — the clean setting the three
models assume (no early exercise). But live quotes do not come with a risk-free rate or a dividend
yield attached: we must <b>recover the carry from the prices themselves</b>.</p>

<h3>1.1 &nbsp; Put-call parity: recovering r, the forward, and the carry</h3>
<p>For a given expiry, no-arbitrage forces a <em>model-free</em> linear relation across strikes:</p>
<div class="eq">C(K) &minus; P(K) = DF &middot; (F &minus; K)
&nbsp;&nbsp;<span class="c">// straight line in K: slope = &minus;DF, intercept = DF&middot;F</span></div>
<p>A single least-squares fit of <code>C&minus;P</code> against <code>K</code> yields the discount factor
<code>DF</code> (hence <code>r = &minus;ln(DF)/T</code>) and the forward <code>F</code> (hence the carry
<code>q = r &minus; ln(F/S₀)/T</code>). Here is what the MO chain actually implies:</p>
<table><thead><tr><th>expiry</th><th>T (yr)</th><th>implied r</th><th>implied q (carry)</th><th>forward F</th><th># OTM strikes</th></tr></thead>
<tbody>{mkt_rows}</tbody></table>
<div class="callout"><b>Reading the market.</b> The forward sits <em>well below</em> spot and falls with maturity —
a steep <b>negative basis</b>. The recovered carry <code>q</code> is large (often 10–20%), not because the index
pays such dividends, but because CSI&nbsp;1000 futures trade at a deep discount, driven largely by
structured-product (snowball) hedging flows. Note also that <code>r</code> is <em>weakly identified</em> at short
maturities: since <code>DF≈1</code>, <code>r=&minus;ln(DF)/T</code> amplifies quote noise by <code>1/T</code>. The forward (the
x-intercept) is robust; the rate (the slope) is not.</div>

<h3>1.2 &nbsp; OTM selection and call-equivalent inversion</h3>
<p>Only <b>out-of-the-money</b> options carry clean volatility information — deep-ITM quotes are dominated by
intrinsic value and go stale. We keep OTM puts below the forward and OTM calls above it, then convert each
OTM put to its <b>call-equivalent</b> price via parity, <code>C = P + DF·(F&minus;K)</code>, so a single Black
inverter handles both wings and the two smiles agree at the forward by construction (the no-arbitrage
property Dupire needs). Inverting every survivor gives the market smile:</p>
{fig(HERE / f"data/plots/02_smiles_{t}.png", "Market implied-vol smiles per expiry (OTM, call-equivalent, Black-inverted).")}

<h2 id="s2">2 &nbsp; Dupire Local Volatility</h2>
<p>Dupire (1994) asks: is there a <em>deterministic</em> instantaneous volatility <code>σ<sub>LV</sub>(S,t)</code>
such that a one-factor diffusion <code>dS = (r&minus;q)S dt + σ<sub>LV</sub>(S,t) S dW</code> reproduces the
<em>entire</em> observed surface of European prices? The answer is unique and given by the forward equation
(in implied total variance <code>w = σ²T</code>, log-moneyness <code>y = ln(K/F)</code>):</p>
<div class="eq">σ<sub>LV</sub>²(K,T) = &part;<sub>T</sub>w &divide; [ 1 &minus; (y/w)&part;<sub>y</sub>w
+ ¼(&minus;¼ &minus; 1/w + y²/w²)(&part;<sub>y</sub>w)² + ½&part;<sub>yy</sub>w ]</div>
<p>The denominator is a <b>butterfly no-arbitrage</b> term; the numerator is a <b>calendar no-arbitrage</b>
term. Both must be positive. On real quotes they often are <em>not</em>:</p>
<div class="callout"><b>Butterfly arbitrage in raw data.</b> Building Dupire directly from the raw MO surface
can raise a no-arbitrage rejection at wing and short-dated nodes — the second strike-derivative
<code>&part;<sub>yy</sub>w</code> estimated on a coarse, bid/ask-quantized grid can produce a negative denominator.
The builder <em>refuses</em> to fabricate a local vol there. The pipeline now does the desk-style repair
upstream: fit a SABR smile to each expiry, evaluate a smooth rectangular IV grid, then project total
variance to be non-decreasing in maturity before running Dupire with validation on. {smooth_text}</div>
{fig(HERE / f"data/plots/03_localvol_surface_{t}.png", "Reconstructed Dupire local-volatility surface σ_LV(K,T).")}
<p>Repricing every OTM option through the local-vol PDE and re-inverting gives an overall IV RMSE of
<b>{lv['overall_rmse_iv']*100:.2f} vol-points</b> against that smoothed target. In the continuous,
arbitrage-free limit Dupire reprices the target surface <em>exactly</em>; the remaining gap is finite-grid
PDE error plus the explicit choice to smooth noisy raw quotes before differentiating them.</p>

<h2 id="s3">3 &nbsp; Heston Stochastic Volatility</h2>
<p>Heston (1993) makes variance itself stochastic — a mean-reverting CIR process correlated with spot:</p>
<div class="eq">dS = (r&minus;q)S dt + &radic;v &middot; S dW₁<br>
dv = &kappa;(&theta; &minus; v) dt + &sigma;&radic;v dW₂,
&nbsp;&nbsp; d&lang;W₁,W₂&rang; = &rho; dt</div>
<p>Five parameters shape the whole surface: <code>v₀</code> and <code>&theta;</code> set the level (spot and
long-run variance), <code>&kappa;</code> the term structure of mean reversion, <code>&sigma;</code> the vol-of-vol
(smile convexity), and <code>&rho;</code> the spot/vol correlation (skew). European prices follow from the
characteristic function by Fourier inversion (Lewis form), which is what the fast calibrator uses.</p>
<h3>3.1 &nbsp; Calibration result</h3>
<table><thead><tr><th>v₀</th><th>κ</th><th>θ</th><th>σ (vol-of-vol)</th><th>ρ</th><th>Feller 2κθ/σ²</th></tr></thead>
<tbody><tr><td class="num">{p['v0']:.4f}</td><td class="num">{p['kappa']:.3f}</td><td class="num">{p['theta']:.4f}</td>
<td class="num">{p['sigma']:.3f}</td><td class="num">{p['rho']:+.3f}</td>
<td class="num">{he['feller']:.2f} {'✓' if feller_ok else '(violated)'}</td></tr></tbody></table>
<p>Overall smile fit: <b>{he['overall_rmse_iv']*100:.2f} vol-points</b> across all {len(pe)} expiries against
the same smoothed target — and unlike raw Dupire this is achieved by an <b>arbitrage-free</b> stochastic
model that cannot manufacture butterfly violations.</p>
{fig(HERE / f"data/plots/04_heston_fit_{t}.png", "Heston fit (thin) vs smoothed target smiles (marked) per expiry.")}
<div class="callout key"><b>Two calibration lessons.</b>
(1) <b>Identification.</b> With only a handful of maturities the smile pins the combination
<code>σ²/κ</code>, not <code>κ</code> alone — an unconstrained fit sends <code>κ</code> to any ceiling for
marginal gain. We <em>regularize</em> (cap <code>κ</code>) rather than overfit.
(2) <b>Feller &amp; numerics.</b> The unconstrained fit is deeply Feller-violated
(<code>2κθ ≪ σ²</code>, so variance can hit zero), and that same extreme <code>σ</code> makes the downstream
ADI PDE mis-price by ~2 vol-points. Adding a <b>Feller penalty</b> keeps <code>2κθ/σ² ≈ 1</code>, costing a
little smile fit but making <em>both</em> the calibration and the PDE trustworthy — the calibration
objective and the numerical scheme are not independent choices.</div>

<h2 id="s4">4 &nbsp; Heston-SLV: the leverage function</h2>
<p>Local vol reprices the smile but has unrealistic (deterministic) dynamics; Heston has realistic dynamics
but cannot fit every smile exactly. <b>SLV</b> unifies them by multiplying the Heston vol by a deterministic
<b>leverage</b> <code>L(S,t)</code>:</p>
<div class="eq">dS = (r&minus;q)S dt + L(S,t)&middot;&radic;v &middot; S dW₁
&nbsp;&nbsp;<span class="c">// L chosen so E[v | S=K]·L(K,t)² = σ_LV²(K,t)</span></div>
<p>The calibration condition (Gyöngy) says the SLV model matches the market local variance iff
<code>L(K,t)² = σ<sub>LV</sub>²(K,t) / E[v<sub>t</sub> | S<sub>t</sub>=K]</code>. We solve it by the forward
Fokker-Planck method: evolve the joint (S,v) density one step at a time and read off the conditional
expectation. The resulting leverage ranges <code>L ∈ [{slv['leverage_min']:.2f}, {slv['leverage_max']:.2f}]</code>
(L=1 would be pure Heston):</p>
{fig(HERE / f"data/plots/05_slv_leverage_{t}.png", "Calibrated SLV leverage surface L(S,t).")}
<div class="callout"><b>Which model for which product?</b> The SLV PDE reprices these European vanillas to
<b>{slv['overall_rmse_iv']*100:.2f} vol-points</b> — <em>not</em> better than analytic Heston. That is expected,
not a failure: the SLV PDE carries a finite-grid discretization bias, while Heston vanillas use a
semi-analytical Fourier pricer. <b>SLV's value is not vanilla repricing</b> — Heston already handles vanillas,
exactly and cheaply. SLV earns its keep on
<b>exotics</b> (barriers, autocallables, forward-starting and cliquet structures) where <em>both</em> the market
smile <em>and</em> realistic forward-vol dynamics matter. The leverage surface above is the reusable deliverable.</div>

{barrier_section}
{hedge_section}
<h2 id="s7">7 &nbsp; Model comparison and takeaways</h2>
<p><b>What is IV RMSE?</b> For every quoted strike/maturity node, we first price the vanilla option
under a candidate model, then convert that model price back into a Black implied volatility.
The <b>implied-volatility root mean square error</b> is</p>
<div class="eq">IV RMSE = &radic; mean<sub>K,T</sub> ( σ<sub>model-implied</sub>(K,T) &minus; σ<sub>market target</sub>(K,T) )²</div>
<p>We report it in <b>vol-points</b>: 1.00 vol-point means 1 percentage point of annualized implied
volatility. This is useful because it compares all models in the same market language, independent
of option notional, moneyness, and maturity price scale. A low IV RMSE means the model reproduces
the terminal vanilla smile well; a high IV RMSE means the model is visibly missing the traded smile.</p>
<div class="callout"><b>What IV RMSE can and cannot tell us.</b> It is the right first diagnostic for
separating flat BSM, Heston, Local Vol, and SLV on <em>vanilla smile fit</em>. It does not uniquely
identify the forward-volatility dynamics. Two models can have similar IV RMSE and still produce
different barrier prices, Snowball values, delta hedges, turnover, and residual PnL. That is why
the lecture uses IV RMSE for the vanilla calibration story, then uses exotics and hedging to
differentiate the dynamics.</div>
<table><thead><tr><th>model</th><th>overall IV RMSE (vol-pts)</th><th>note</th></tr></thead>
<tbody>{rmse_table}</tbody></table>
<h3>Per-expiry IV RMSE (vol-points)</h3>
<table><thead><tr><th>T (yr)</th><th>Local Vol</th><th>Heston</th><th>SLV</th></tr></thead>
<tbody>{perT_rows}</tbody></table>
<div class="callout key"><b>The arc of the study.</b>
A flat-vol Black-Scholes leaves a <b>{bsflat*100:.1f} vol-point</b> smile on the table.
<b>Local vol</b> can in principle remove all of it, but only after the raw quotes are turned into a smooth
no-arbitrage target. <b>Heston</b>, arbitrage-free by construction and Feller-regularized for numerical health,
fits that target robustly to a few vol-points. <b>SLV</b> then layers a leverage surface on top so the model is
simultaneously smile-consistent and stochastic — the right tool once you leave vanillas for exotics.
The recurring theme: <em>market data is not arbitrage-free, parameters are not fully identified, and the
calibration objective must respect the numerics that will consume its output.</em></div>

<h2 id="s8">8 &nbsp; Reproducing this study</h2>
<p>Two interpreters are involved: AKShare (data) and quantark (models) do not share one environment.</p>
<div class="eq"><span class="c"># stage 01 fetches live data — AKShare interpreter</span>
/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py

<span class="c"># stages 02-09 replay the snapshot offline — quantark .venv</span>
.venv/bin/python example/mo_volmodels/02_build_iv_surface.py --snapshot latest
.venv/bin/python example/mo_volmodels/03_dupire_localvol.py   --tag latest
.venv/bin/python example/mo_volmodels/04_heston_calibration.py --tag latest
.venv/bin/python example/mo_volmodels/05_slv_calibration.py    --tag latest
.venv/bin/python example/mo_volmodels/07_barrier_exotic.py     --tag latest
.venv/bin/python example/mo_volmodels/08_snowball_exotic.py    --tag latest
.venv/bin/python example/mo_volmodels/09_delta_hedging.py      --tag latest
.venv/bin/python example/mo_volmodels/06_lecture.py           --tag latest</div>
<p><span class="pill">tip</span> A committed synthetic <code>--snapshot sample</code> (arbitrage-free by
construction) drives the automated tests and lets stages 02–09 run with no network.</p>

<footer>Generated by <code>example/mo_volmodels/06_lecture.py</code> from live artifacts (tag: <code>{t}</code>).
Models: <code>quantark.volmodels</code> (Dupire / Heston / SLV). Data: AKShare CSI&nbsp;1000 option chain.
This is educational material, not investment advice.</footer>
</div></body></html>"""

    out = HERE / f"data/mo_volmodels_lecture_{t}.html"
    out.write_text(html)
    print(f"wrote {out}")
    for name, r, _ in rmse_rows:
        print(f"  {name:32s} RMSE = {r*100:6.2f} vol-pts")


if __name__ == "__main__":
    main()
