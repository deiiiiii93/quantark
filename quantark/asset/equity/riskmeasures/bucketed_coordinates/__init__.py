"""Bucketed greek coordinate families (R1b pure code motion).

Each module exposes ``calculate_points(calc, product, pricing_env, engine,
request, mode)`` where ``calc`` is the ``GreeksCalculator`` facade instance;
the facade's ``calculate_bucketed_greeks`` owns the dispatch. Nothing here is
a stable import surface.
"""
