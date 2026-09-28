"""Intraday study construction: the daily-KI fixture, the SSE clock, history, checkpoints."""
from datetime import datetime, timedelta, timezone

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.builders import intraday_common as ic
from quantark.modelvalidation.study import CaseSpec
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.timestamp import to_utc

ENV = {"spot": 100.0, "vol": 0.20, "rate": 0.03, "div_yield": 0.01}
PRODUCT = {"initial_date": "2026-03-16", "months": 12, "initial_price": 100.0, "strike": 100.0, "ko_barrier": 103.0,
           "ki_barrier": 75.0, "ko_rate": 0.12, "rebate_rate": 0.12, "contract_multiplier": 1.0,
           "ki_observation": "daily_close"}
CONTEXT = {"valuation": "2026-09-10T14:00:00+08:00", "phase": "before", "profile": "desk", "calendar": "SSE",
           "history_level": 100.0}
SHANGHAI = timezone(timedelta(hours=8))


def test_product_builder_validates_and_returns_the_spec():
    assert ic.build_intraday_snowball_spec(PRODUCT) == PRODUCT
    with pytest.raises(ValidationError, match="ki_observation"):
        ic.build_intraday_snowball_spec({**PRODUCT, "ki_observation": "weekly"})
    with pytest.raises(ValidationError, match="Unknown"):
        ic.build_intraday_snowball_spec({**PRODUCT, "ki_barier": 75.0})


def test_daily_ki_contract_observes_ki_at_every_sse_close_and_ko_monthly():
    product = ic.make_snowball(PRODUCT)
    cal = ic.exchange_calendar("SSE")
    ki = product.barrier_config.ki_observation_schedule.records
    ko = product.barrier_config.ko_observation_schedule.records
    assert len(ko) == 12 and all(r.barrier == 103.0 for r in ko)
    assert all(r.barrier == 75.0 and cal.is_business_day(r.observation_date) for r in ki)
    assert datetime(2026, 3, 16) < ki[0].observation_date and ki[-1].observation_date == product.exercise_date
    assert len(ki) > 200
    monthly = ic.make_snowball({**PRODUCT, "ki_observation": "ko_dates"})
    assert [r.observation_date for r in monthly.barrier_config.ki_observation_schedule.records] == [r.observation_date for r in ko]


def test_settlement_lag_puts_a_settlement_date_on_every_ko_record():
    lagged = ic.make_snowball({**PRODUCT, "settlement_lag_days": 30})
    for record in lagged.barrier_config.ko_observation_schedule.records:
        assert record.settlement_date == record.observation_date + timedelta(days=30)


def test_context_builder_requires_an_aware_valuation_instant():
    assert ic.build_intraday_context_spec(CONTEXT)["profile"] == "desk"
    with pytest.raises(ValidationError, match="offset"):
        ic.build_intraday_context_spec({**CONTEXT, "valuation": "2026-09-10T14:00:00"})
    with pytest.raises(ValidationError, match="profile"):
        ic.build_intraday_context_spec({**CONTEXT, "profile": "bespoke"})


def test_request_resolves_with_confirmed_history_at_the_declared_level():
    request = ic.build_request(ENV, PRODUCT, CONTEXT)
    ctx = resolve_context(request)
    assert ctx.valuation_timestamp == datetime(2026, 9, 10, 14, 0, tzinfo=SHANGHAI)
    assert request.variance_profile.name == "desk" and request.session_calendar.name == "SSE"
    assert request.fixings and all(f.value == 100.0 for f in request.fixings)
    assert all(to_utc(f.timestamp) < to_utc(ctx.valuation_timestamp) for f in request.fixings)
    assert not ctx.provisional and ctx.numerical.lifecycle_state.alive
    nxt = min(to_utc(e.timestamp) for e in ctx.numerical.remaining_events)
    assert nxt == to_utc(datetime(2026, 9, 10, 15, 0, tzinfo=SHANGHAI))


def test_a_fixing_override_above_the_ko_terminates_the_claim():
    context = {**CONTEXT, "fixings": [{"date": "2026-08-17", "level": 104.0}]}
    ctx = resolve_context(ic.build_request(ENV, PRODUCT, context))
    assert ctx.numerical.terminated and ctx.numerical.lifecycle_state.knocked_out


def test_a_checkpoint_covers_earlier_events_so_no_fixings_are_supplied_for_them():
    context = {**CONTEXT, "checkpoint": {"as_of": "2026-09-09", "knocked_in": True, "ki_date": "2026-06-15"}}
    request = ic.build_request(ENV, PRODUCT, context)
    assert request.lifecycle_state.knocked_in and not request.fixings
    ctx = resolve_context(request)
    assert ctx.numerical.knocked_in and not ctx.provisional


def test_arm_merges_case_overrides_and_validates_the_merged_specs():
    arm = ic.IntradayArm(environment_params=ENV, product_params=PRODUCT, context_params=CONTEXT,
                         quantities=("pv",), params={})
    env, product, context = arm.specs(CaseSpec(name="x", environment_params={"spot": 75.2},
                                               context_params={"valuation": "2026-09-10T14:59:59+08:00"}))
    assert env["spot"] == 75.2 and context["valuation"].endswith("14:59:59+08:00") and product == PRODUCT
    with pytest.raises(ValidationError):
        arm.specs(CaseSpec(name="bad", context_params={"profile": "bespoke"}))


def test_a_due_fixing_left_unfixed_is_provisional_and_a_future_one_is_not():
    """A future fixing decides nothing; only an event already due and missing is resolved provisionally."""
    before = resolve_context(ic.build_request({**ENV, "spot": 74.9}, PRODUCT, CONTEXT))
    assert not before.provisional and not before.numerical.knocked_in           # 14:00, the 15:00 fixing is ahead
    after = {**CONTEXT, "valuation": "2026-09-10T15:00:30+08:00", "phase": "after", "unfixed_from": "2026-09-10"}
    ctx = resolve_context(ic.build_request({**ENV, "spot": 74.9}, PRODUCT, after))
    assert ctx.provisional and ctx.numerical.knocked_in
    assert all(f.timestamp.date().isoformat() < "2026-09-10" for f in ctx.request.fixings)


def test_resolved_inputs_carry_the_merged_specs_and_the_quantities():
    arm = ic.IntradayArm(environment_params=ENV, product_params=PRODUCT, context_params=CONTEXT,
                         quantities=("pv", "desk_delta"), params={})
    inputs = arm.resolved_inputs(CaseSpec(name="x", environment_params={"spot": 75.2}))
    assert inputs["environment"]["spot"] == 75.2 and inputs["environment"]["vol"] == 0.20
    assert inputs["context"]["valuation"] == CONTEXT["valuation"] and inputs["quantities"] == ["pv", "desk_delta"]
    assert ic.plain({"d": datetime(2026, 9, 10).date()}) == {"d": "2026-09-10"}


def test_the_fingerprint_covers_the_delegated_numerics_the_calendar_data_and_the_builders():
    """Review R1: the gamma readout lives in base_pde_solver.py, not in the solver class's own module."""
    # Relative to the package, not split at the first "quantark/": CI checks the repository out into
    # .../quantark/quantark, so the first match is the checkout directory itself.
    package = ic._repo_root() / "quantark"
    rel = lambda trees: {p.relative_to(package).as_posix() for p in ic.fingerprint_inputs(*trees)}     # noqa: E731
    pde, common = rel(ic.PDE_TREES), rel(ic.COMMON_TREES)
    assert "asset/equity/engine/pde/base_pde_solver.py" in pde and "asset/equity/engine/pde/snowball_pde_solver.py" in pde
    assert "util/calendar/holidayfile/china_sse.csv" in common
    assert "modelvalidation/builders/intraday_common.py" in common and "intraday/greeks.py" in common
    assert any(name.startswith("asset/equity/product/option/") for name in common)
    assert any(name.startswith("montecarlo/") for name in rel(ic.MC_TREES))
    assert len(ic.implementation_fingerprint(*ic.COMMON_TREES, *ic.PDE_TREES)) == 64
    with pytest.raises(ValidationError, match="does not exist"):
        ic.fingerprint_inputs("quantark/no_such_tree")


def test_the_fingerprint_moves_when_a_delegated_function_changes(tmp_path):
    engine = tmp_path / "quantark" / "engine" / "pde"
    engine.mkdir(parents=True)
    (engine / "solver.py").write_text("from .base import readout\n", encoding="utf-8")
    (engine / "base.py").write_text("def readout(v):\n    return v[1]\n", encoding="utf-8")
    before = ic.implementation_fingerprint("quantark/engine/pde", root=tmp_path)
    assert before == ic.implementation_fingerprint("quantark/engine/pde", root=tmp_path)
    (engine / "base.py").write_text("def readout(v):\n    return v[2]\n", encoding="utf-8")       # the delegated numerics
    assert ic.implementation_fingerprint("quantark/engine/pde", root=tmp_path) != before
