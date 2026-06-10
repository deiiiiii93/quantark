"""
Factory functions for creating SIMM sensitivity engines.
"""

from typing import Dict, Any, Optional, Type
from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import RiskClass

from quantark.simm.engines.base import SensitivityEngine, BaseSensitivityEngine

# Import engine classes
try:
    from quantark.simm.engines.risk_class.ir_engine import IRSensitivityEngine
except ImportError:
    IRSensitivityEngine = None

try:
    from quantark.simm.engines.risk_class.equity_engine import EquitySensitivityEngine
except ImportError:
    EquitySensitivityEngine = None


def create_engine(
    risk_class: RiskClass,
    config: SIMMConfig,
    **kwargs: Any,
) -> SensitivityEngine:
    """
    Factory function to create a sensitivity engine for the given risk class.

    Args:
        risk_class: The risk class to create an engine for
        config: SIMM configuration settings
        **kwargs: Additional keyword arguments to pass to the engine constructor

    Returns:
        An instance of a SensitivityEngine for the specified risk class

    Raises:
        ValueError: If no engine is available for the specified risk class
    """
    engines: Dict[RiskClass, Type[BaseSensitivityEngine]] = {
        RiskClass.INTEREST_RATE: IRSensitivityEngine,
        RiskClass.EQUITY: EquitySensitivityEngine,
    }

    engine_class = engines.get(risk_class)

    if engine_class is None:
        raise ValueError(
            f"No sensitivity engine available for risk class: {risk_class}. "
            f"Available engines: {list(engines.keys())}"
        )

    if engine_class is None:
        raise ValueError(
            f"Engine class for {risk_class} is not implemented yet"
        )

    try:
        return engine_class(config=config, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Error creating engine for {risk_class}: {e}. "
            f"Check that required arguments are provided in kwargs"
        ) from e


def create_all_engines(
    config: SIMMConfig,
    **kwargs: Any,
) -> Dict[RiskClass, SensitivityEngine]:
    """
    Create all available sensitivity engines.

    Args:
        config: SIMM configuration settings
        **kwargs: Additional keyword arguments to pass to engine constructors

    Returns:
        Dict mapping risk classes to their respective engines

    Note:
        Only engines that are currently implemented will be included.
    """
    engines: Dict[RiskClass, SensitivityEngine] = {}

    # Interest Rate engine
    try:
        engines[RiskClass.INTEREST_RATE] = create_engine(
            RiskClass.INTEREST_RATE, config, **kwargs
        )
    except (ValueError, TypeError):
        # Engine not implemented yet
        pass

    # Equity engine
    try:
        engines[RiskClass.EQUITY] = create_engine(
            RiskClass.EQUITY, config, **kwargs
        )
    except (ValueError, TypeError):
        # Engine not implemented yet
        pass

    return engines


def get_engine_name(engine: SensitivityEngine) -> str:
    """
    Get the name of a sensitivity engine.

    Args:
        engine: The engine to get the name for

    Returns:
        The name of the engine class
    """
    return engine.__class__.__name__


def get_available_engines() -> Dict[str, Type[BaseSensitivityEngine]]:
    """
    Get all available engine classes.

    Returns:
        Dict mapping engine names to engine classes
    """
    engines: Dict[str, Type[BaseSensitivityEngine]] = {}

    if IRSensitivityEngine is not None:
        engines["IR"] = IRSensitivityEngine

    if EquitySensitivityEngine is not None:
        engines["Equity"] = EquitySensitivityEngine

    return engines
