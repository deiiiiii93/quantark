# AI strategy index study

Builds an equal-weighted index of ten China A-share AI stocks and asks whether it suits snowball
products: backtest, KO/KI probabilities under two barrier structures (103% KO / 70% KI and the
original), hedging and replication feasibility, and the q/sigma pricing parameters.

- `ai_index_backtest.py`, `ai_index_wind_backtest.py` — index construction and backtest (AKShare / Wind data)
- `ai_index_snowball_analysis.py` — phases 1-4
- `ai_index_optimized_analysis.py` — higher-KO structure
- `ai_index_phase5_7_analysis.py` — hedging, pricing parameters, methodology
