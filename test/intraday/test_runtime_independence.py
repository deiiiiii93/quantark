"""The intraday runtime prices with modelvalidation unimportable and no evidence anywhere (spec section 11)."""
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent(
    '''
    import builtins, sys
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("quantark.modelvalidation"):
            raise ImportError("modelvalidation is not available in this process")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard
    from datetime import datetime, time, timedelta, timezone
    from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
    from quantark.intraday import Fixing, VarianceProfile, resolve_context, spot_curve, value_intraday
    from quantark.intraday.events import EventKind
    from quantark.intraday.request import IntradayValuationRequest
    from quantark.intraday.session import TradingSession, TradingSessionCalendar
    from quantark.util.calendar import CalendarType, create_calendar
    sys.path.insert(0, "test")
    from intraday.conftest import SHANGHAI, dated_snowball, flat_env

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    sse = TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=cal,
                                 sessions=(TradingSession(time(9, 30), time(11, 30)), TradingSession(time(13, 0), time(15, 0))))
    desk = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    prod = dated_snowball(cal, datetime(2026, 3, 16))
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse, variance_profile=desk))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    for convention in ("point", "desk_bump"):
        req = IntradayValuationRequest(product=prod, pricing_env=flat_env(kos[5].timestamp - timedelta(hours=1), spot=75.2),
                                       session_calendar=sse, variance_profile=desk, fixings=fixings,
                                       greeks=("delta", "gamma", "vega", "theta"), greek_convention=convention)
        res = value_intraday(SnowballQuadEngineV2(), req)
        assert all(g.status == "ok" for g in res.greeks), [(g.name, g.status, g.reason) for g in res.greeks]
    curve = spot_curve(SnowballQuadEngineV2(), req, [60.0, 75.2, 200.0])
    assert [p.status for p in curve] == ["ok", "ok", "ok"], [p.status for p in curve]
    assert not any(m.startswith("quantark.modelvalidation") for m in sys.modules)
    print("RUNTIME_INDEPENDENT")
    '''
)


def test_pricing_succeeds_with_modelvalidation_unimportable_and_no_evidence():
    env = {"PYTHONPATH": str(REPO), "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    proc = subprocess.run([sys.executable, "-c", SCRIPT], cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr
    assert "RUNTIME_INDEPENDENT" in proc.stdout


def test_no_route_accepts_a_certification_flag():
    from quantark.intraday.engines import analytical_barrier, analytical_digital, mc, pde, quad_v2
    for module in (analytical_barrier, analytical_digital, mc, pde, quad_v2):
        for name, cls in inspect.getmembers(module, inspect.isclass):
            fn = getattr(cls, "point_greeks", None)
            if fn is not None:
                assert "certify" not in inspect.signature(fn).parameters, (module.__name__, name)


def test_runtime_package_carries_no_evidence_or_certification_vocabulary():
    import quantark.intraday as pkg
    root = Path(pkg.__file__).parent
    assert not (root / "evidence").exists()
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for word in ("unqualified", "Gate C", "certif"):
            assert word.lower() not in text.lower(), f"{path.relative_to(root)} still says {word!r}"
