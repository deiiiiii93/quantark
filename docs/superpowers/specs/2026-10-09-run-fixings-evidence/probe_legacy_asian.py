"""L1: a legacy-schedule Asian (exercise_date + num_observations, no records) carries no past fixings: every
valuation spreads its 12 observations uniformly over the REMAINING life, so ageing never makes one pass."""
from datetime import timedelta

import _fixtures as F
from quantark.asset.equity.product.option.asian_option import AsianOption
from quantark.util.enum import OptionType

legacy = AsianOption(strike=100.0, option_type=OptionType.CALL, exercise_date=F.START + timedelta(days=365),
                     num_observations=12)
for day in (0, 40, 200):
    past, _, future, _, n = legacy.resolve_observations(F.eq_env(day))
    print(f"day {day:>3}: past {len(past)}, future {len(future)} (first at t={future[0]:.6f}, "
          f"i.e. day {day + future[0] * 365:.1f}), total {n}")
