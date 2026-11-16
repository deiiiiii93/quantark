"""Native ADI execution of the autocallable families' value and statistics stages.

The family owns payoffs, rows, coupon memory and event ordering. The ADI core
owns the spot/variance operator. Columns are rolling; operator factorizations
are shared across claims and no time history of knocked-in surfaces is kept.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from quantark.asset.equity.engine.event_stats import StatisticsProvenance
from quantark.asset.equity.engine.pde.core.prepare import OutputRequest, Operation, PreparedProblem
from quantark.asset.equity.engine.pde.grid import Layout, SpatialLayout, TimeLayout
from quantark.util.enum import ADIScheme


class _LinearReadout:
    def price(self, column, spot, layout):
        return float(np.interp(np.log(spot), layout.spatial.x, column))


class _VarianceClaimCorrection:
    """The existing first-passage geometry, shared by all claims per V node."""
    def __init__(self, fp, n_v, n_cols):
        self.fp, self.n_v, self.n_cols = fp, n_v, n_cols

    def step_correction(self, j, s, barrier, d):
        D = d.reshape(len(s), self.n_v, self.n_cols)
        out = np.zeros_like(D)
        for k in range(self.n_v):
            G, sample, distance = self.fp._geometry(float(self.fp._dt[j]),
                float(self.fp._mu[j, k]), float(self.fp._sig2[j, k]), s, barrier)
            if G is not None:
                out[:, k, :] = G[:, None]*(D[sample, k, :]/distance)[None, :]
        return out.reshape(d.shape)

    def add_correction(self, j, s, barrier, v0, v1):
        v0 += self.step_correction(j, s, barrier, v1-v0)


class _VarianceClaimBreach:
    """Allocate the family's KO/seed targets across the variance batch."""
    def __init__(self, stats, n_v):
        self.stats, self.n_v = stats, n_v

    def ko(self, record, block, surface):
        s, L = self.stats, self.stats.layout
        target = np.zeros((block.shape[0], self.n_v, s.n_cols))
        target[:, :, s.column_of_record(record, surface)] = s.breach.unit(record)
        if L.kie is not None:
            target[:, :, L.kie] = block.reshape(target.shape)[:, :, L.kie]
        return target.reshape(block.shape)

    def seed(self, block):
        base = self.stats.breach.seed(np.zeros((block.shape[0], self.stats.n_cols)))
        return np.repeat(base[:, None, :], self.n_v, axis=1).reshape(block.shape)


def local_vol_ki_ever(solver, product, env):
    """KO-independent monitoring even when today's note outcome is decided."""
    from .core.prepare import prepare
    from .core.march import march
    request = OutputRequest(Operation.SESSION_OUTPUTS, price=False, event_stats=True,
                            short_circuit=False)
    p = prepare(solver, product, env, request)
    result = march(p)
    if p.statistics is None:
        return 0.0
    return p.statistics.read(result.states, result.captures, 0.0).ki_ever_probability


def _family(owner, product):
    from .snowball_pde_solver import SnowballPDESolver
    from .phoenix_pde_solver import PhoenixPDESolver
    from .ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    cls = (KOResetSnowballPDESolver if isinstance(product, KnockOutResetSnowballOption)
           else PhoenixPDESolver if isinstance(product, PhoenixOption) else SnowballPDESolver)
    # Preserve overrides and per-operation hooks on existing Snowball/Phoenix
    # facades. KO-reset selects its distinct family and staged maturity rules.
    return cls(params=owner.params) if isinstance(product, KnockOutResetSnowballOption) else owner


def _prepare(owner, product, env, want_stats, streams, fold):
    family = _family(owner, product)
    family._validate_product(product)
    request = OutputRequest(Operation.PRICE, price=True, event_stats=want_stats, streams=streams,
                            fold=fold if fold is None or hasattr(fold, "bind_ko") else None)
    adapter = family.adapter
    family._product_token_memo.clear()
    state = adapter.solve_state(product, env, request)
    if fold is not None and not hasattr(fold, "bind_ko"):
        family._fold = fold
    T = float(product.get_maturity(env))
    market = family.market_snapshot(product, env)
    grid_request = adapter.grid_request(product, market, T, request)
    from .grid import build_time
    from quantark.asset.equity.engine.continuous_ki_support import continuous_coefficient_model
    _, knots = continuous_coefficient_model(env)
    extra = set(grid_request.extra_times) | {x for x in knots if 0 < x < T}
    lev = getattr(owner, 'leverage_surface', None)
    lev_values = getattr(lev, 'leverage_grid', None)
    if lev_values is not None and not np.all(lev_values == lev_values[0]):
        extra.update(float(x) for x in lev.time_grid if 0 < x < T)
    grid_request = replace(grid_request, extra_times=tuple(sorted(extra)))
    records = list(family._get_cached_ko_records(env, product))
    if hasattr(family, '_get_cached_post_ko_records'):
        records.extend(family._get_cached_post_ko_records(env, product))
    event_times = set(grid_request.event_times)
    event_times.update(float(r.observation_time) for r in records
                       if r.observation_time is not None and 0 < r.observation_time < T)
    grid_request = replace(grid_request, event_times=tuple(sorted(event_times)))
    clock = build_time(grid_request, replace(family.grid_binder.config, time_steps=owner.n_t))
    t = np.array(clock.t, copy=True)
    core = owner._make_core(product, env, T, time_grid=t)
    step_of = dict(clock.step_of)
    spatial = SpatialLayout(core.S_grid, core.X_grid, np.diff(core.X_grid),
                            (core.S_grid[0], core.S_grid[-1]), float('nan'))
    layout = Layout(spatial, TimeLayout(t, np.diff(t), step_of, core.N_T, core.N_T),
                    grid_request, family.grid_binder.config.key)
    adapter.observation_maps(product, env, layout, T)
    events = adapter.prepare_events(product, env, layout, T, request)
    prepared = PreparedProblem(family, adapter, product, env, request, None, T, None,
                               state, layout=layout, readout=_LinearReadout(), events=events)
    stats = adapter.statistics_family(prepared, streams) if want_stats else None
    # Per-column first passage uses exactly the ADI spot diffusion at the
    # barrier, including SLV leverage, rather than the strike's implied vol.
    if owner is not family:
        owner._prepare_state(product, env, T, state['ki_continuous'])
    owner._prepare_2d_continuous_ki_correction(core, product)
    fp = owner._ki_fp
    from .snowball_pde_solver import _ContinuousKIFirstPassage
    corrections = [None] * core.N_V
    if fp is not None:
        corrections = [_ContinuousKIFirstPassage(dt=np.diff(t), mu=fp._mu[:, k],
                       sig2=fp._sig2[:, k], is_reverse=product.is_reverse) for k in range(core.N_V)]
    return family, prepared, core, stats, corrections, fp


def _ki_monitor(owner, family, p, core, st):
    """One KO-independent underlying claim, with its own damping schedule."""
    if not p.product.has_ki_barrier:
        return 0.0
    if p.solve_state['knocked_in']:
        return 1.0
    from .snowball_vol_pde_solvers import event_damped_step_keys
    horizon = float(p.product.get_pre_maturity_time(p.env)) if hasattr(p.product, 'get_pre_maturity_time') else core.T
    df_T = p.env.get_discount_factor(core.T)
    ki_keys = {core.N_T-j: None for j in family._ki_observation_indices}
    damped = event_damped_step_keys(st, {'ki': ki_keys}, core.N_T)

    def boundary(U, tau):
        current = core.T-float(tau)
        unit = family._df_between_times(p.env, current, core.T) if current <= horizon+1e-12 else 0.
        if p.product.is_reverse:
            U[0, :], U[-1, :] = 0., unit
        else:
            U[0, :], U[-1, :] = unit, 0.
        return U

    def event(U, tau, j=None, dt=0.):
        current = core.T-float(tau)
        if current > horizon+1e-12:
            return U
        if not p.solve_state['ki_continuous'] and j not in family._ki_observation_indices:
            return U
        barrier = family._ki_barrier if j is None else family._resolve_ki_barrier_at_tidx(j)
        unit = family._df_between_times(p.env, current, core.T)
        hit = np.full_like(U, unit)
        if not p.solve_state['ki_continuous'] and j != 0 and family._use_cell_average_events():
            return family._project_event_values(core.S_grid, barrier, p.product.is_reverse, False, U, hit)
        mask = family._event_nodal_mask(core.S_grid, barrier, p.product.is_reverse, False,
                                       at_valuation=j == 0)
        U[mask, :] = unit
        if p.solve_state['ki_continuous'] and j is not None and 0 < j < core.N_T and current+dt <= horizon+1e-12:
            if owner._ki_fp is not None:
                U += owner._ki_fp.step_correction(j, core.S_grid, barrier, hit-U)
        return U

    U = np.zeros((core.N_S, core.N_V))
    boundary(U, 0.)
    U = event(U, 0., core.N_T)
    previous = core._boundary_hook
    core._boundary_hook = boundary
    try:
        def advance(U, j, tau, dt, terminal=False):
            damping = damped and core.N_T-j in damped
            step = core._douglas_step if terminal or damping or owner.scheme == ADIScheme.DOUGLAS else core._cs_step
            theta = 1. if terminal else float(st.event_theta if damping else st.theta)
            from .adi_columns import step_columns
            edge = boundary(np.zeros_like(U), tau)
            return step_columns(core, U[:, :, None], dt, tau, theta, core.T-tau+.5*dt,
                                np.array([edge[0, 0]]), np.array([edge[-1, 0]]),
                                terminal or damping or owner.scheme == ADIScheme.DOUGLAS,
                                force_scalar=True)[:, :, 0]
        if st.terminal_damping_steps > 0:
            dt = float(p.layout.time.dt[-1])
            U = event(advance(U, core.N_T-1, .5*dt, .5*dt, True), .5*dt)
            U = event(advance(U, core.N_T-1, dt, .5*dt, True), dt, core.N_T-1, dt)
            first = core.N_T-2
        else:
            first = core.N_T-1
        for j in range(first, -1, -1):
            dt = float(p.layout.time.dt[j])
            tau = core.T-float(p.layout.time.t[j])
            U = event(advance(U, j, tau, dt), tau, j, dt)
    finally:
        core._boundary_hook = previous
    return float(core.interpolate(U, np.log(p.env.spot), owner.model_params.v0) / df_T)


def native_adi(owner, product, env, *, event_stats=False, streams=None, npv=None, fold=None, return_surface=False):
    """Return native PV or statistics on the same ADI mesh and event stages."""
    from .snowball_vol_pde_solvers import event_damped_step_keys
    from quantark.asset.equity.engine.pde.route_applicability import check_route_applicability
    from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
    from quantark.asset.equity.engine.pde.core.components import resolve_stepper

    family, p, core, stats, corrections, fp_2d = _prepare(owner, product, env, event_stats, streams, fold)
    check_route_applicability(owner.params, '2D')
    st = resolve_stepper(owner.params, DAMPING_DEFAULTS['2D'])
    maps = {'dt': core.dt, 'ko': {core.N_T - p.layout.time.step_at(t): None for t in p.layout.request.event_times}}
    damped = event_damped_step_keys(st, maps, core.N_T)
    # Family stages expect 1D time grids. These small scratch grids contain the
    # current/next column only in use; ADI claims themselves keep no time axis.
    scratch = p.adapter.allocate(p)
    captures = [{} for _ in range(core.N_V)]
    value_names = tuple(scratch)
    values = {name: np.zeros((core.N_S, core.N_V)) for name in value_names}
    blocks = ({name: np.zeros((core.N_S, core.N_V, stats.n_cols))
               for name in stats.group.surfaces} if stats is not None else {})
    # Terminal payoffs/events depend on spot and product state, not variance.
    # Initialize them once; repeating contract/calendar work per V column can
    # dominate a cold price, especially for deferred coupon claims.
    family._ki_fp = corrections[0]
    p.adapter.terminal(p, scratch, captures[0])
    family._ki_fp = corrections[0]
    for name, grid in scratch.items():
        values[name] = np.repeat(grid[:, -1, None], core.N_V, axis=1)
    if stats is not None:
        view = {name: block[:, 0, :] for name, block in blocks.items()}
        stats.terminal(view)
        family._ki_fp = corrections[0]
        for block in blocks.values():
            block[...] = block[:, 0:1, :]
    value_names = tuple(values)

    def active_claims():
        claims = list(values.items()) + list(blocks.items())
        if not p.solve_state['knocked_in']:
            return claims
        # A carried KI cannot return to the alive regime. Retain its scratch
        # state for the family hooks, but skip the unused ADI solves.
        return [(name, A) for name, A in claims
                if name not in ('alive', 'w0', 'stats_v0') and not name.startswith('v0[')]

    owner._last_native_adi_diagnostics = dict(
        requested_steps=int(owner.n_t), actual_steps=int(core.N_T),
        n_x=int(core.N_S), n_v=int(core.N_V), time_nodes=p.layout.time.t.tolist(),
        statistics_columns=0 if stats is None else int(stats.n_cols),
        active_claims=sum(1 if A.ndim == 2 else A.shape[2] for _, A in active_claims()),
        ki_monitor=bool(stats is not None and stats.want_ki),
        discounting='exact',
        variance_transport='cubic_characteristic' if core._semi_lagrangian_v else core.v_drift_scheme,
    )

    def boundaries(tau, j):
        # Rows are allocated by the actual product family. They do not depend
        # on variance; a single evaluation supplies every ADI variance slice.
        for name in value_names:
            scratch[name][:, j] = values[name][:, 0]
        for group in p.adapter.groups(product, p.layout, p.request):
            p.adapter.boundaries(p, group, scratch, j)
        rows = {name: (scratch[name][0, j], scratch[name][-1, j]) for name in value_names}
        if stats is not None:
            lo, hi = stats.rows(j, None)
            for i, name in enumerate(stats.group.surfaces):
                rows[name] = (lo[i * stats.n_cols:(i + 1) * stats.n_cols],
                              hi[i * stats.n_cols:(i + 1) * stats.n_cols])
        return rows

    def advance(tau, dt, j, terminal_half=False):
        old_time = p.layout.time.t[j]
        try:
            if terminal_half:
                p.layout.time.t[j] = core.T-tau
            rows = boundaries(tau, j)
        finally:
            p.layout.time.t[j] = old_time
        theta = 1. if terminal_half else float(st.event_theta if damped and core.N_T-j in damped else st.theta)
        douglas = terminal_half or (damped and core.N_T-j in damped) or owner.scheme == ADIScheme.DOUGLAS
        from .adi_columns import step_columns
        claims = active_claims()
        packed = np.concatenate([A[:, :, None] if A.ndim == 2 else A for _, A in claims], axis=2)
        low = np.concatenate([np.atleast_1d(rows[name][0]) for name, _ in claims])
        high = np.concatenate([np.atleast_1d(rows[name][1]) for name, _ in claims])
        out = step_columns(core, packed, dt, tau, theta, core.T-tau+.5*dt, low, high, douglas,
                           force_scalar=getattr(owner, 'column_backend', 'auto') == 'scalar')
        c = 0
        for _, A in claims:
            width = 1 if A.ndim == 2 else A.shape[2]
            A[...] = out[:, :, c] if A.ndim == 2 else out[:, :, c:c+width]
            c += width
        return

    def events(j, half=False):
        from .core.adapters.phoenix import PhoenixAdapter
        if j > 0 and not half and not isinstance(p.adapter, PhoenixAdapter):
            # Snowball/reset stages already accept a matrix of columns. Batch
            # V nodes too, so a daily KI projection is built once per date.
            family._ki_fp = fp_2d
            if p.events is not None:
                block = p.events.apply(j, {'alive': values['alive'], 'ki': values['ki']})
                block = p.events.continuous(j, block)
                values['alive'][...], values['ki'][...] = block['alive'], block['ki']
            else:
                seed = family._v0_seed_values
                if seed is not None:
                    family._v0_seed_values = seed[:, None]
                try:
                    family._two_surface_events(values['alive'], values['ki'], core.S_grid, j,
                        core.N_T+1, float(p.layout.time.t[j]), product, env)
                finally:
                    family._v0_seed_values = seed
            if stats is not None:
                family._ki_fp = None if fp_2d is None else _VarianceClaimCorrection(fp_2d, core.N_V, stats.n_cols)
                breach = _VarianceClaimBreach(stats, core.N_V)
                a = blocks['stats_v0'].reshape(core.N_S, -1)
                b = blocks['stats_v1'].reshape(core.N_S, -1)
                if p.events is not None:
                    block = p.events.apply(j, {'alive': a, 'ki': b}, breach=breach)
                    block = p.events.continuous(j, block)
                    a[...], b[...] = block['alive'], block['ki']
                else:
                    family._two_surface_events(a, b, core.S_grid, j, core.N_T+1,
                        float(p.layout.time.t[j]), product, env, breach=breach)
            return
        for k in range(core.N_V):
            family._ki_fp = None if half else corrections[k]
            if half:
                if p.solve_state['ki_continuous']:
                    for a, b in [('alive', 'ki')] + [(f'v0[{m}]', f'v1[{m}]') for m in range(2)]:
                        if a in values:
                            family._ki_columns(values[a][:, k:k+1], values[b][:, k:k+1], core.S_grid,
                                               j, core.N_T+1, product)
                    if stats is not None:
                        family._ki_columns(blocks['stats_v0'][:, k, :], blocks['stats_v1'][:, k, :],
                                           core.S_grid, j, core.N_T+1, product)
                continue
            for name in value_names:
                scratch[name][:, j] = values[name][:, k]
                if j < core.N_T:
                    scratch[name][:, j+1] = values[name][:, k]
            p.adapter.step_events(p, scratch, captures[k], j)
            if j == 0:
                for key in ("_t0_pre_event_cols", "_t0_readout_cols"):
                    cols = getattr(family, key, None)
                    if cols is not None:
                        captures[k][key] = tuple(np.array(c, copy=True) for c in cols)
                vals = getattr(family, "_t0_readout_values", None)
                if vals is not None:
                    captures[k]["_t0_readout_values"] = vals
            for name in value_names:
                values[name][:, k] = scratch[name][:, j]
            if stats is not None:
                view = {name: block[:, k, :] for name, block in blocks.items()}
                stats.step_events(view, captures[k], j)
                for name, block in view.items():
                    blocks[name][:, k, :] = block

    core._opt_is_call = True
    if st.terminal_damping_steps > 0:
        # Preserve the core's two half steps and its boundary monitoring.
        terminal_dt = float(p.layout.time.dt[-1])
        advance(.5*terminal_dt, .5*terminal_dt, core.N_T-1, True)
        events(core.N_T-1, half=True)
        advance(terminal_dt, .5*terminal_dt, core.N_T-1, True)
        events(core.N_T-1)
        first = core.N_T-2
    else:
        first = core.N_T-1
    for j in range(first, -1, -1):
        advance(core.T-float(p.layout.time.t[j]), float(p.layout.time.dt[j]), j)
        events(j)
    # Reduce the variance axis at the actual v0; then use the core's linear
    # spot readout for both value and every probability/cashflow column.
    def reduce(A):
        i = max(0, min(int(np.searchsorted(core.V_grid, owner.model_params.v0))-1, core.N_V-2))
        w = (owner.model_params.v0-core.V_grid[i])/(core.V_grid[i+1]-core.V_grid[i])
        return (1-w)*A[:, i, ...] + w*A[:, i+1, ...]
    reduced = {name: np.column_stack((reduce(A), np.zeros(core.N_S))) for name, A in values.items()}
    # Readout needs column zero; t0 capture is variance-interpolated too.
    cap = {}
    for key in captures[0]:
        if isinstance(captures[0][key], tuple) and key != "_t0_readout_values":
            cap[key] = tuple(reduce(np.stack([c[key][m] for c in captures], axis=1))
                             for m in range(len(captures[0][key])))
    for key in ("_t0_pre_event_cols", "_t0_readout_cols"):
        if key in cap:
            setattr(family, key, cap[key])
    owner._ki_fp = fp_2d
    if "_t0_readout_values" in captures[0]:
        family._t0_readout_values = tuple(float(np.interp(owner.model_params.v0, core.V_grid,
            [c["_t0_readout_values"][m] for c in captures])) for m in range(2))
    # The family stages supply the surface selection, including carried KI.
    if return_surface:
        name = p.adapter._value_surface(1 if p.solve_state['knocked_in'] else 0)
        if "t0_pre_event" in captures[0]:
            sel = 1 if p.solve_state['knocked_in'] else 0
            return core, np.stack([c['t0_pre_event'][sel] for c in captures], axis=1)
        return core, values[name]
    result = SimpleNamespace(states=reduced, captures=cap)
    pv = p.adapter._price(p, p.adapter._columns(p, result))
    if stats is None:
        return None if event_stats else pv
    reduced_blocks = {name: reduce(A) for name, A in blocks.items()}
    report = stats.read(reduced_blocks, cap, float(pv if npv is None else npv))
    if stats.want_ki:
        report = replace(report, ki_ever_probability=_ki_monitor(owner, family, p, core, st))
    decision = family._valuation_decision(product, env)
    if decision.knocked_out:
        from .core.statistics.terminal import knocked_out_statistics
        known_pv = family._get_immediate_ko_payoff(product, env) if npv is None else npv
        known = knocked_out_statistics(family, product, env, p.request, decision, pv=known_pv)
        report = replace(known, ki_ever_probability=report.ki_ever_probability)
    return replace(report, provenance=StatisticsProvenance(owner._price_model, owner._price_model))
