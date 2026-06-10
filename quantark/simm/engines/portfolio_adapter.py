"""
Portfolio to SIMM sensitivity adapter.

This module provides an adapter to convert portfolio positions to SIMM sensitivities,
routing positions to appropriate engines based on risk class.
"""

from typing import Dict, List, Any, Union, Optional
from datetime import datetime

from quantark.simm.config import SIMMConfig
from quantark.simm.sensitivity import SensitivityCollection
from quantark.simm.engines.factory import create_engine, create_all_engines
from quantark.simm.engines.base import SensitivityEngine

# Import portfolio classes
try:
    from quantark.portfolio.equity.portfolio import EquityPortfolio
    from quantark.portfolio.equity.position import EquityPosition
except ImportError:
    EquityPortfolio = None
    EquityPosition = None

try:
    from quantark.portfolio.fi.portfolio import FIPortfolio
    from quantark.portfolio.fi.position import FIPosition
except ImportError:
    FIPortfolio = None
    FIPosition = None

# Import CRIF parser
try:
    from quantark.simm.crif.parser import CRIFParser
except ImportError:
    CRIFParser = None


class SIMMPortfolioAdapter:
    """
    Adapter to convert portfolios to SIMM sensitivity collections.

    This class provides a unified interface to:
    - Convert portfolios to sensitivity collections
    - Route positions to appropriate engines
    - Import/export CRIF format
    - Aggregate results across multiple engines
    """

    def __init__(self, config: SIMMConfig):
        """
        Initialize the portfolio adapter.

        Args:
            config: SIMM configuration settings
        """
        self.config = config
        self._engines_cache: Dict[str, SensitivityEngine] = {}

    def portfolio_to_sensitivities(
        self,
        portfolio: Union[Any, Dict[str, Any]],
        **kwargs: Any,
    ) -> SensitivityCollection:
        """
        Convert a portfolio to SIMM sensitivities.

        This method:
        1. Determines portfolio type (equity, fixed income, or mixed)
        2. Creates appropriate engines for each risk class
        3. Routes positions to correct engines
        4. Aggregates results into a single SensitivityCollection

        Args:
            portfolio: Portfolio object or dict with positions
            **kwargs: Additional arguments (e.g., greeks_calculator for equity engine)

        Returns:
            SensitivityCollection containing all calculated sensitivities
        """
        # Determine portfolio type
        portfolio_type = self._determine_portfolio_type(portfolio)

        # Extract positions and pricing environments
        positions, pricing_environments = self._extract_portfolio_data(portfolio)

        # Create engines for the portfolio type
        engines = self._create_engines_for_portfolio_type(portfolio_type, **kwargs)

        # Calculate sensitivities using each engine
        all_sensitivities = SensitivityCollection()

        for risk_class, engine in engines.items():
            try:
                engine_collection = engine.calculate_sensitivities(
                    positions, pricing_environments, self.config
                )
                all_sensitivities.add_many(engine_collection.sensitivities)
            except Exception as e:
                # Log error but continue with other engines
                print(f"Error calculating sensitivities for {risk_class}: {e}")

        return all_sensitivities

    def crif_to_sensitivities(
        self,
        crif_records: List[Any],
    ) -> SensitivityCollection:
        """
        Import sensitivities from CRIF format.

        Args:
            crif_records: List of CRIFRecord objects

        Returns:
            SensitivityCollection parsed from CRIF
        """
        if CRIFParser is None:
            raise ImportError("CRIF parser not available. Install required dependencies.")

        parser = CRIFParser()
        return parser.parse_to_sensitivities(crif_records)

    def sensitivities_to_crif(
        self,
        sensitivities: SensitivityCollection,
        config: Optional[SIMMConfig] = None,
    ) -> List[Any]:
        """
        Export sensitivities to CRIF format.

        Args:
            sensitivities: SensitivityCollection to export
            config: SIMM configuration (defaults to self.config)

        Returns:
            List of CRIFRecord objects
        """
        if CRIFParser is None:
            raise ImportError("CRIF parser not available. Install required dependencies.")

        if config is None:
            config = self.config

        parser = CRIFParser()
        return parser.sensitivities_to_crif(sensitivities, config)

    def _determine_portfolio_type(self, portfolio: Union[Any, Dict[str, Any]]) -> str:
        """
        Determine the type of portfolio.

        Args:
            portfolio: Portfolio object or dict

        Returns:
            Portfolio type: "equity", "fi", "mixed", or "dict"
        """
        # Check if it's a dict (unstructured portfolio)
        if isinstance(portfolio, dict):
            return "dict"

        # Check class name
        class_name = portfolio.__class__.__name__.upper()

        if EquityPortfolio is not None and isinstance(portfolio, EquityPortfolio):
            return "equity"
        elif FIPortfolio is not None and isinstance(portfolio, FIPortfolio):
            return "fi"

        # Check for equity-related attributes
        if hasattr(portfolio, 'equity_positions') or 'equity_positions' in dir(portfolio):
            return "equity"

        # Check for fixed income-related attributes
        if hasattr(portfolio, 'fi_positions') or 'fi_positions' in dir(portfolio):
            return "fi"

        # Check for general positions attribute
        if hasattr(portfolio, 'positions'):
            # Try to infer type from positions
            positions = getattr(portfolio, 'positions', {})
            if positions:
                first_pos = next(iter(positions.values()))
                if EquityPosition is not None and isinstance(first_pos, EquityPosition):
                    return "equity"
                elif FIPosition is not None and isinstance(first_pos, FIPosition):
                    return "fi"

        # Default to mixed if we can't determine
        return "mixed"

    def _extract_portfolio_data(
        self,
        portfolio: Union[Any, Dict[str, Any]],
    ) -> tuple[List[Any], Dict[str, Any]]:
        """
        Extract positions and pricing environments from portfolio.

        Args:
            portfolio: Portfolio object or dict

        Returns:
            Tuple of (positions_list, pricing_environments_dict)
        """
        positions = []
        pricing_environments = {}

        if isinstance(portfolio, dict):
            # Handle dict-based portfolios
            if 'positions' in portfolio:
                positions = list(portfolio['positions'].values())
            if 'pricing_environments' in portfolio:
                pricing_environments = portfolio['pricing_environments']
        else:
            # Handle portfolio objects
            # Extract positions
            if hasattr(portfolio, 'positions'):
                positions = list(portfolio.positions.values())
            elif hasattr(portfolio, 'equity_positions'):
                positions = list(portfolio.equity_positions.values())
            elif hasattr(portfolio, 'fi_positions'):
                positions = list(portfolio.fi_positions.values())

            # Extract pricing environments
            if hasattr(portfolio, 'pricing_environments'):
                pricing_environments = portfolio.pricing_environments
            elif hasattr(portfolio, 'env'):
                # Single environment for all positions
                if positions:
                    for pos in positions:
                        pricing_environments[pos.underlying] = portfolio.env

        return positions, pricing_environments

    def _create_engines_for_portfolio_type(
        self,
        portfolio_type: str,
        **kwargs: Any,
    ) -> Dict[str, SensitivityEngine]:
        """
        Create appropriate engines for the portfolio type.

        Args:
            portfolio_type: Type of portfolio
            **kwargs: Additional arguments for engines

        Returns:
            Dict mapping risk class names to engines
        """
        engines: Dict[str, SensitivityEngine] = {}

        if portfolio_type == "equity":
            # Create equity and FX engines
            from quantark.simm.taxonomy import RiskClass

            try:
                engines["EQUITY"] = create_engine(
                    RiskClass.EQUITY, self.config, **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not create Equity engine: {e}")

            try:
                engines["FX"] = create_engine(
                    RiskClass.FX, self.config, **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not create FX engine: {e}")

        elif portfolio_type == "fi":
            # Create IR, Credit Q, and Credit NQ engines
            from quantark.simm.taxonomy import RiskClass

            try:
                engines["INTEREST_RATE"] = create_engine(
                    RiskClass.INTEREST_RATE, self.config, **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not create IR engine: {e}")

            try:
                engines["CREDIT_Q"] = create_engine(
                    RiskClass.CREDIT_QUALIFYING, self.config, **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not create Credit Q engine: {e}")

            try:
                engines["CREDIT_NQ"] = create_engine(
                    RiskClass.CREDIT_NON_QUALIFYING, self.config, **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not create Credit NQ engine: {e}")

        elif portfolio_type == "mixed" or portfolio_type == "dict":
            # Create all available engines
            engines = create_all_engines(self.config, **kwargs)

        return engines

    def get_available_engines(self) -> List[str]:
        """
        Get list of available engine names.

        Returns:
            List of engine names
        """
        from quantark.simm.engines.factory import get_available_engines

        engines = get_available_engines()
        return list(engines.keys())

    def get_engine_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities of each engine.

        Returns:
            Dict mapping engine names to lists of supported margin types
        """
        from quantark.simm.taxonomy import RiskClass, MarginType

        capabilities = {}

        try:
            from quantark.simm.taxonomy import RiskClass as RC
            from quantark.simm.engines.risk_class.ir_engine import IRSensitivityEngine
            capabilities["IRSensitivityEngine"] = ["Delta", "Vega", "Curvature"]
        except Exception:
            pass

        try:
            from quantark.simm.engines.risk_class.equity_engine import EquitySensitivityEngine
            capabilities["EquitySensitivityEngine"] = ["Delta", "Vega"]
        except Exception:
            pass

        return capabilities
