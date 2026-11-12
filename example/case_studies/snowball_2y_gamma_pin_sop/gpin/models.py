"""Hedge models: the true note with a modified barrier schedule, priced with the true note's lifecycle state.

A hedge model supplies a hedge delta (or a gap size) for the ledger; it never supplies a P&L mark, which always
comes from the ``true`` model (spec D3). The ids and shift sizes are read from the pre-registration's ``models``
block; which window each model is priced in follows the technique registry:

==================  ================================================================  ===========
model id            definition                                                        priced when
==================  ================================================================  ===========
``true``            the note                                                          every live day
``surv``            the next KO observation disabled (survived branch)                KO window
``ko_s050`` ..      every KO barrier x (1 + ``ko_dir`` x size)                        KO window
``ko_w100``         every KO barrier x (1 - ``ko_dir`` x 1%), the wrong-way control   KO window
``ki_m100``, ..     KI barrier x (1 + x), x = -1%, -2%; ``ki_p100`` x = +1% (control) KI window
``kin``             the note priced with a copy of the lifecycle state, knocked in    KI window
==================  ================================================================  ===========

``ko_dir`` is the pricing pass's choice of the over-hedge direction for one observation (-1 when redemption costs
the seller, see ``gpin.pricing``). Products are built unaged and numeric exactly as ``s2ts.note.build_note`` builds
the note, except for the per-observation KO barrier list; the pricing pass ages them with the tracker's
``age_copy``, which drops the observations behind the valuation date.

``kin``: ``quantark/asset/equity/engine/quad/v2/adapters.py`` reads the knocked-in state from
``lifecycle_state.knocked_in`` first (``initial_hit``, line 465) and from the product's
``_otc_lifecycle_knocked_in`` only when the state carries none, so the forced state has to be a COPY of the
lifecycle state (a product attribute alone would leave ``kin`` equal to ``true``); the tracker's own state is never
touched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

from s2ts import NOTIONAL
from s2ts.note import KI_PCT, KO_PCT, Terms

from . import load_prereg

#: Barrier multiple (of the strike) that disables a KO observation: no path reaches it.
SURV_OFF_MULTIPLE = 1.0e3

WINDOWS = ("always", "ko", "ki")


@dataclass(frozen=True)
class HedgeModel:
    """One hedge model. ``ko_shift``: signed relative shift of every KO barrier per unit of ``ko_dir`` (the
    wrong-way control carries a negative one); ``ki_shift``: signed relative shift of the KI barrier;
    ``disable_next_ko``: ``surv``; ``force_knocked_in``: ``kin``; ``window``: ``always`` / ``ko`` / ``ki``."""

    id: str
    window: str
    ko_shift: float = 0.0
    ki_shift: float = 0.0
    disable_next_ko: bool = False
    force_knocked_in: bool = False

    @property
    def needs_ko_dir(self) -> bool:
        """True for the models that move the KO barrier in the over-hedge direction."""
        return not is_zero(self.ko_shift)

    def product(self, terms: Terms, s0: float, coupon: float, day: date, ko_dir: Optional[int],
                ki_pct: float = KI_PCT):
        """The unaged numeric note with this model's barrier schedule, valued on ``day`` (``day`` only decides
        which KO observation ``surv`` disables). ``ki_pct``: the book's KI level; ``ki_shift`` is relative to it."""
        return create_standard_snowball(
            initial_price=s0,
            strike=s0,
            maturity=terms.maturity_years,
            contract_multiplier=NOTIONAL / s0,
            ko_barrier=ko_levels(self, terms, day, ko_dir, s0=s0),
            ko_rate=float(coupon),
            ki_barrier=ki_level(self, s0, ki_pct),
            num_observations=len(terms.ko_times),
            is_reverse=False,
            ko_observation_dates=list(terms.ko_times),
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=list(terms.ki_times),
            rebate_rate=float(coupon),
            include_principal=False,
        )

    def lifecycle(self, state: Any) -> Any:
        """The lifecycle state to price with: ``kin`` a deep copy with ``knocked_in`` forced on, every other model
        the state itself. The tracker's own state is never modified."""
        if not self.force_knocked_in:
            return state
        forced = copy.deepcopy(state)
        forced.knocked_in = True
        return forced


def next_ko_index(terms: Terms, day: date) -> Optional[int]:
    """Index of the first KO observation strictly after ``day`` (``day``'s own observation is already
    processed at its close), ``None`` when none is left."""
    for i, d in enumerate(terms.ko_dates):
        if d > day:
            return i
    return None


def ko_levels(model: HedgeModel, terms: Terms, day: date, ko_dir: Optional[int], s0: float = 1.0) -> List[float]:
    """One KO barrier level per observation of the term sheet, as ``build_note`` takes them (past observations
    included; ageing drops them). Levels are in index points for ``s0`` (relative to the strike by default).
    ``surv`` sets the next observation's barrier to ``SURV_OFF_MULTIPLE x s0``."""
    base = KO_PCT * s0
    if model.needs_ko_dir:
        if ko_dir not in (-1, 1):
            raise ValidationError(f"model {model.id} shifts the KO barrier and needs ko_dir -1 or +1, got {ko_dir!r}")
        base = base * (1.0 + ko_dir * model.ko_shift)
    levels = [base] * len(terms.ko_dates)
    if model.disable_next_ko:
        i = next_ko_index(terms, day)
        if i is None:
            raise ValidationError(f"{model.id}: no KO observation after {day}")
        levels[i] = SURV_OFF_MULTIPLE * s0
    return levels


def ki_level(model: HedgeModel, s0: float, ki_pct: float = KI_PCT) -> float:
    """The KI barrier level (index points for ``s0``) of this model in the book whose KI level is ``ki_pct``:
    ``ki_pct x s0 x (1 + ki_shift)``."""
    return ki_pct * s0 * (1.0 + model.ki_shift)


def book_note(coupon: float, s0: float, terms: Terms, ki_pct: float = KI_PCT):
    """The book's note: ``s2ts.note.build_note`` (one scalar KO level) with the KI level ``ki_pct``; the same
    factory call, so at 0.75 it is ``build_note``'s note exactly (``s2ts`` hardcodes 75%)."""
    return create_standard_snowball(
        initial_price=s0,
        strike=s0,
        maturity=terms.maturity_years,
        contract_multiplier=NOTIONAL / s0,
        ko_barrier=KO_PCT * s0,
        ko_rate=float(coupon),
        ki_barrier=ki_pct * s0,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=float(coupon),
        include_principal=False,
    )


def _tag(value: float) -> str:
    """Basis points of ``value`` as the id suffix: 0.005 -> ``050``, 0.03 -> ``300``."""
    return f"{int(round(abs(value) * 1.0e4)):03d}"


def _build_models() -> Dict[str, HedgeModel]:
    cfg = load_prereg()["models"]
    models = [HedgeModel("true", "always"), HedgeModel("surv", "ko", disable_next_ko=True)]
    models += [HedgeModel(f"ko_s{_tag(x)}", "ko", ko_shift=float(x)) for x in cfg["ko_shift_sizes"]]
    models.append(HedgeModel(f"ko_w{_tag(cfg['ko_wrong_size'])}", "ko", ko_shift=-float(cfg["ko_wrong_size"])))
    models += [HedgeModel(f"ki_m{_tag(x)}", "ki", ki_shift=float(x)) for x in cfg["ki_shifts"]]
    models.append(HedgeModel(f"ki_p{_tag(cfg['ki_wrong_shift'])}", "ki", ki_shift=float(cfg["ki_wrong_shift"])))
    models.append(HedgeModel("kin", "ki", force_knocked_in=True))
    return {m.id: m for m in models}


#: Every hedge model, in the order the pricing pass solves them within a window.
MODELS: Dict[str, HedgeModel] = _build_models()
