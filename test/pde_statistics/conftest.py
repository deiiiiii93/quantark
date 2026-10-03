import pytest

from quantark.asset.equity.engine.pde.core.adapters.ko_reset import KOResetAdapter
from quantark.asset.equity.engine.pde.core.adapters.phoenix import PhoenixAdapter
from quantark.asset.equity.engine.pde.core.adapters.snowball import SnowballAdapter

ADAPTERS = {"snowball": SnowballAdapter, "phoenix": PhoenixAdapter, "ko_reset": KOResetAdapter}


@pytest.fixture
def value_mode(monkeypatch):
    """Switch families to the value-mode statistics before their stage-2 admission (plan F1)."""
    def switch(*families):
        for fam in families:
            monkeypatch.setattr(ADAPTERS[fam], "statistics_mode", "value")
    return switch
