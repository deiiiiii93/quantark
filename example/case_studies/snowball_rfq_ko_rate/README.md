# Snowball RFQ KO-rate demo

Shows how r, q and vol move the fair KO rate quoted on a Snowball RFQ (standard Snowball, 103% monthly KO,
75% daily KI, 2-year, principal excluded), priced by PDE and checked across Quad, PDE and MC.

- `calculate_*_data.py` — pricing payload and scenario CSV
- `render_*_html.py` — standalone dashboard from a saved payload (`templates/` holds the HTML template)
- `generate_*.py` — calculate + render end to end
- `snowball_rfq_ko_rate_demo_workflow.py` — the full workflow; `snowball_rfq_ko_rate_engine_compare_demo.py` — engine consistency check
