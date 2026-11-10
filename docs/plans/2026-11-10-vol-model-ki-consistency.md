# Native local-volatility and Heston/SLV KI consistency

Scope: the same Snowball, Phoenix, memory Phoenix and absolute KO-reset outcome
semantics as the completed GBM repair, on native MC and PDE model routes.

Use the existing local-volatility 1D value/statistics march. Extend its immediate
KO readout with the independent monitoring claim. MC retains its realized
interval variance bridge and is qualified with seeded paths and refinement.

Use the product family's existing event stages and boundary cash allocation on
the Heston/SLV ADI operator. This is preferable to repeated folded repricing for
production statistics, whose cost scales with the observation count, and to a
flat-volatility surrogate, which changes the model. Keep ADI claims rolling and
reuse directional operator caches. Price-only requests allocate only the value
states. Coupon arrears need two states per KI regime; absolute KO-reset uses the
existing staged pre-maturity seed and pre/post event dispatch.

Native statistics become the 2D default. Explicit refuse and labelled surrogate
modes remain available. Keep raw numerical probabilities unnormalised so a
partition check can expose finite-step discounting error. Verify actual folded
cash price increments, unit-leverage and deterministic-variance reductions,
reverse and overlapping events, carried KI, valuation-date coupons/KO, deferred
coupons, stream pruning and lifecycle wrappers. Qualify non-flat local volatility
and genuine Heston/SLV dynamics with finer PDE and independently seeded MC.
Benchmark cold price calls at matched grids and accuracy; report statistics cost
separately. Preserve the GBM gates and all previous artifacts.
