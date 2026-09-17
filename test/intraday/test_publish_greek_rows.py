"""The generated matrix names each certificate's fixture and every evidence source."""
from quantark.intraday import publish


def test_greek_rows_name_their_fixtures_and_every_evidence_source(monkeypatch):
    evidence = {"git_sha": "bbbb", "sources": [{"git_sha": "aaaa", "cells": 10}, {"git_sha": "bbbb", "cells": 154}],
                "demonstrated": [{"product": "SnowballOption", "route": "QuadV2Route", "measure": "point_delta",
                                  "monitoring": "discrete", "profile": "desk", "settings": {"engine": "x.SnowballQuadEngineV2"},
                                  "measure_settings": {}, "horizon_s": 1, "horizon_max_s": 21600, "offsets": ["eq"],
                                  "barriers": ["ki", "ko"], "fixtures": ["snowball_daily_ki"]}]}
    monkeypatch.setattr(publish, "greek_evidence", lambda: evidence)
    text = publish._greek_section()
    assert "(git aaaa, 10 cells; git bbbb, 154 cells)" in text
    assert "| SnowballOption | snowball_daily_ki | QuadV2Route | point_delta |" in text


def test_rows_without_a_fixture_label_render_a_dash(monkeypatch):
    evidence = {"git_sha": "cccc", "demonstrated": [{"product": "P", "route": "R", "measure": "m", "settings": {},
                                                     "horizon_s": 1, "horizon_max_s": 2}]}
    monkeypatch.setattr(publish, "greek_evidence", lambda: evidence)
    text = publish._greek_section()
    assert "(git cccc)" in text and "| P | — | R | m |" in text
