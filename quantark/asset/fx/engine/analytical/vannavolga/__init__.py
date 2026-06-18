"""
Vanna-Volga FX barrier pricing (engine layer).

Public API:
    - Barrier building blocks: ``one_touch_hit_prob``, ``no_touch_price``,
      ``survival_probability_single``
    - Attenuation: ``gamma_surv``, ``gamma_fet``, ``p_vanna_p_volga_from_gamma``
    - Arbitrage clamps: ``BarrierPrices``, ``enforce_single_barrier_arbitrage``,
      ``enforce_double_barrier_arbitrage``
    - Pricing: ``price_vv_one_touch``, ``VVBarrierResult``, ``BarrierGamma``
"""

from .arbitrage import (
    BarrierPrices,
    clamp_basic,
    enforce_double_barrier_arbitrage,
    enforce_single_barrier_arbitrage,
)
from .attenuation import (
    gamma_fet,
    gamma_surv,
    gamma_surv_single,
    p_vanna_p_volga_from_gamma,
)
from .barrier_bs import (
    no_touch_price,
    one_touch_hit_prob,
    reiner_rubinstein_barrier,
    survival_probability_single,
)
from .vv_barrier import (
    BarrierGamma,
    VVBarrierResult,
    numeric_greeks_ot,
    price_ot_bstv,
    price_vv_one_touch,
)
from .vv_vanilla_barrier import numeric_greeks_barrier, price_vv_barrier
from .vv_barrier_engine import VannaVolgaBarrierEngine

__all__ = [
    "one_touch_hit_prob",
    "no_touch_price",
    "survival_probability_single",
    "reiner_rubinstein_barrier",
    "gamma_surv",
    "gamma_surv_single",
    "gamma_fet",
    "p_vanna_p_volga_from_gamma",
    "BarrierPrices",
    "clamp_basic",
    "enforce_single_barrier_arbitrage",
    "enforce_double_barrier_arbitrage",
    "BarrierGamma",
    "VVBarrierResult",
    "price_ot_bstv",
    "numeric_greeks_ot",
    "price_vv_one_touch",
    "numeric_greeks_barrier",
    "price_vv_barrier",
    "VannaVolgaBarrierEngine",
]
