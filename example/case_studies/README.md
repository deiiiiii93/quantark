# Case studies

Each folder is one self-contained study: its own scripts, committed inputs and (where it has them) README.
They answer a question and keep the answer; they are not API tutorials (those live in the package folders
next to this one, e.g. `example/asset/`, `example/var/`).

| Study | What it asks |
| --- | --- |
| `mo_volmodels/` | **Reference pattern.** Calibrate local vol, Heston and SLV on an option market end to end (flagship; start here to calibrate vol models for a new market) |
| `fx_volmodels/` | The same pattern for FX: a deliberately narrower question on CFETS snapshots |
| `volmodels_improvement_program/` | Research artifacts behind the vol-model solver improvements |
| `snowball_q_term_structure/` | Does a q term structure help hedge a snowball? Includes the bucket-futures-hedge validation |
| `snowball_simulated_paths/` | Realised CSI 1000 history turned into bootstrapped and stress path batches for the snowball hedging study |
| `snowball_volmodel_backtest/` | Term sheet, trading calendar and inception scheduler (locked 3Y seller snowball on 000852.SH) for the vol-model backtest fleet |
| `snowball_rfq_ko_rate/` | How r, q and vol move the quoted KO rate on a snowball RFQ |
| `snowball_greeks_quad_1001/` | Greeks-vs-spot report for snowball variants on QUAD |
| `ppp_dki_snowball_backtest/` | PPP / NPP parachute snowball delta-hedging comparison |
| `ai_index_strategy/` | Is an AI-stock index a suitable snowball underlying? |
| `phoenix_engine_diagnostics/` | Where Phoenix engines disagree, and why |
| `gaussian_quad_comparison/`, `quad_reference_comparison/`, `quad_v2_validation/`, `quad_v2_feasibility/` | QUAD accuracy, reference comparison and the V2 engine's validation and feasibility |
| `engine_benchmarks/` | Timing and convergence checks across engines |
| `intraday_benchmark/` | Intraday valuation latency and peak memory at declared accuracy |
