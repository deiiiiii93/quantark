"""
Bucket classification for SIMM sensitivity calculations.

This module provides utilities for mapping instruments to SIMM buckets
based on issuer, sector, region, and other classification criteria.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class BucketInfo:
    """Information about a bucket classification."""

    bucket_number: int
    risk_class: str
    classification_type: str  # "equity", "credit", "commodity"
    description: str
    region: Optional[str] = None
    sector: Optional[str] = None


class BucketMapper:
    """
    Maps instruments to SIMM buckets.

    This class provides classification logic for mapping positions
    to their appropriate SIMM buckets for margin calculation.
    """

    # Equity bucket mappings
    # Per SIMM v2.6: 12 buckets (1-12)
    EQUITY_BUCKET_MAPPINGS: Dict[str, BucketInfo] = {
        # Bucket 1: Emerging Markets
        "EMERGING": BucketInfo(1, "EQUITY", "equity", "Emerging Markets"),
        "EMERGING_MARKETS": BucketInfo(1, "EQUITY", "equity", "Emerging Markets"),

        # Bucket 2: Developed Markets - Europe
        "EUROPE": BucketInfo(2, "EQUITY", "equity", "Developed Markets - Europe"),
        "EU": BucketInfo(2, "EQUITY", "equity", "Developed Markets - Europe"),
        "STOXX": BucketInfo(2, "EQUITY", "equity", "Developed Markets - Europe"),
        "EUROPEAN": BucketInfo(2, "EQUITY", "equity", "Developed Markets - Europe"),

        # Bucket 3: North America
        "US": BucketInfo(3, "EQUITY", "equity", "North America"),
        "USA": BucketInfo(3, "EQUITY", "equity", "North America"),
        "SP500": BucketInfo(3, "EQUITY", "equity", "North America"),
        "S&P": BucketInfo(3, "EQUITY", "equity", "North America"),
        "NASDAQ": BucketInfo(3, "EQUITY", "equity", "North America"),

        # Bucket 4: Asia Pacific (excluding Japan)
        "ASIA": BucketInfo(4, "EQUITY", "equity", "Asia Pacific (ex-Japan)"),
        "ASIA_PACIFIC": BucketInfo(4, "EQUITY", "equity", "Asia Pacific (ex-Japan)"),
        "HANG_SENG": BucketInfo(4, "EQUITY", "equity", "Asia Pacific (ex-Japan)"),
        "NIKKEI": BucketInfo(4, "EQUITY", "equity", "Japan", region="JP"),

        # Bucket 5: Technology sector
        "TECH": BucketInfo(5, "EQUITY", "equity", "Technology", sector="Technology"),
        "TECHNOLOGY": BucketInfo(5, "EQUITY", "equity", "Technology", sector="Technology"),
        "AAPL": BucketInfo(5, "EQUITY", "equity", "Technology", sector="Technology"),
        "APPLE": BucketInfo(5, "EQUITY", "equity", "Technology", sector="Technology"),
        "MSFT": BucketInfo(5, "EQUITY", "equity", "Technology", sector="Technology"),
        "GOOGL": BucketInfo(5, "EQUITY", "equity", "Technology", sector="Technology"),

        # Bucket 6: Healthcare/Pharma
        "HEALTHCARE": BucketInfo(6, "EQUITY", "equity", "Healthcare/Pharma", sector="Healthcare"),
        "PHARMA": BucketInfo(6, "EQUITY", "equity", "Healthcare/Pharma", sector="Healthcare"),
        "PHARMACEUTICAL": BucketInfo(6, "EQUITY", "equity", "Healthcare/Pharma", sector="Healthcare"),

        # Bucket 7: Financials
        "FINANCIAL": BucketInfo(7, "EQUITY", "equity", "Financials", sector="Financial"),
        "FINANCIALS": BucketInfo(7, "EQUITY", "equity", "Financials", sector="Financial"),
        "JPMORGAN": BucketInfo(7, "EQUITY", "equity", "Financials", sector="Financial"),
        "BANK": BucketInfo(7, "EQUITY", "equity", "Financials", sector="Financial"),

        # Bucket 8: Consumer
        "CONSUMER": BucketInfo(8, "EQUITY", "equity", "Consumer", sector="Consumer"),
        "RETAIL": BucketInfo(8, "EQUITY", "equity", "Consumer", sector="Consumer"),
        "WALMART": BucketInfo(8, "EQUITY", "equity", "Consumer", sector="Consumer"),

        # Bucket 9: Industrials
        "INDUSTRIAL": BucketInfo(9, "EQUITY", "equity", "Industrials", sector="Industrial"),
        "INDUSTRIALS": BucketInfo(9, "EQUITY", "equity", "Industrials", sector="Industrial"),

        # Bucket 10: Energy
        "ENERGY": BucketInfo(10, "EQUITY", "equity", "Energy", sector="Energy"),
        "OIL": BucketInfo(10, "EQUITY", "equity", "Energy", sector="Energy"),
        "EXXON": BucketInfo(10, "EQUITY", "equity", "Energy", sector="Energy"),
        "CHEVRON": BucketInfo(10, "EQUITY", "equity", "Energy", sector="Energy"),

        # Bucket 11: Indices, funds, ETFs
        "ETF": BucketInfo(11, "EQUITY", "equity", "Indices, funds, ETFs"),
        "INDEX": BucketInfo(11, "EQUITY", "equity", "Indices, funds, ETFs"),
        "SPY": BucketInfo(11, "EQUITY", "equity", "Indices, funds, ETFs"),
        "QQQ": BucketInfo(11, "EQUITY", "equity", "Indices, funds, ETFs"),
        "VTI": BucketInfo(11, "EQUITY", "equity", "Indices, funds, ETFs"),
        "FUND": BucketInfo(11, "EQUITY", "equity", "Indices, funds, ETFs"),

        # Bucket 12: Volatility indices
        "VIX": BucketInfo(12, "EQUITY", "equity", "Volatility indices"),
        "VOLATILITY": BucketInfo(12, "EQUITY", "equity", "Volatility indices"),
    }

    # Credit bucket mappings (simplified)
    # Credit Qualifying (CQ): 12 buckets + residual
    # Credit Non-Qualifying (CNQ): 2 buckets
    CREDIT_BUCKET_MAPPINGS: Dict[str, BucketInfo] = {
        # Bucket 1: Sovereigns
        "US_TREASURY": BucketInfo(1, "CREDIT", "credit", "Sovereign", region="US"),
        "UST": BucketInfo(1, "CREDIT", "credit", "Sovereign", region="US"),
        "GERMAN_BUND": BucketInfo(1, "CREDIT", "credit", "Sovereign", region="DE"),
        "UK_GILT": BucketInfo(1, "CREDIT", "credit", "Sovereign", region="UK"),
        "JGB": BucketInfo(1, "CREDIT", "credit", "Sovereign", region="JP"),

        # Bucket 2: Financials (IG)
        "JPMORGAN": BucketInfo(2, "CREDIT", "credit", "Financials (IG)", sector="Financial"),
        "BANK_OF_AMERICA": BucketInfo(2, "CREDIT", "credit", "Financials (IG)", sector="Financial"),
        "WELLS_FARGO": BucketInfo(2, "CREDIT", "credit", "Financials (IG)", sector="Financial"),

        # Bucket 4: Consumer (IG)
        "WALMART": BucketInfo(4, "CREDIT", "credit", "Consumer (IG)", sector="Consumer"),
        "PROCTER_GAMBLE": BucketInfo(4, "CREDIT", "credit", "Consumer (IG)", sector="Consumer"),
        "P&G": BucketInfo(4, "CREDIT", "credit", "Consumer (IG)", sector="Consumer"),

        # Bucket 5: Technology (IG)
        "APPLE": BucketInfo(5, "CREDIT", "credit", "Technology (IG)", sector="Technology"),
        "MICROSOFT": BucketInfo(5, "CREDIT", "credit", "Technology (IG)", sector="Technology"),

        # CNQ Buckets (high yield or non-rated)
        "HY_CORPORATE": BucketInfo(2, "CNQ", "credit", "High Yield Corporate"),
        "NR_CORPORATE": BucketInfo(2, "CNQ", "credit", "Non-Rated Corporate"),
    }

    # Commodity bucket mappings
    # Per SIMM v2.6: 17 commodity buckets
    COMMODITY_BUCKET_MAPPINGS: Dict[str, BucketInfo] = {
        # Energy
        "OIL": BucketInfo(1, "COMMODITY", "commodity", "Crude Oil"),
        "BRENT": BucketInfo(1, "COMMODITY", "commodity", "Brent Crude"),
        "WTI": BucketInfo(1, "COMMODITY", "commodity", "WTI Crude"),
        "GAS": BucketInfo(2, "COMMODITY", "commodity", "Natural Gas"),
        "NATGAS": BucketInfo(2, "COMMODITY", "commodity", "Natural Gas"),
        "GASOLINE": BucketInfo(3, "COMMODITY", "commodity", "Gasoline"),
        "HEATING_OIL": BucketInfo(4, "COMMODITY", "commodity", "Heating Oil"),

        # Metals
        "GOLD": BucketInfo(5, "COMMODITY", "commodity", "Gold"),
        "SILVER": BucketInfo(6, "COMMODITY", "commodity", "Silver"),
        "PLATINUM": BucketInfo(7, "COMMODITY", "commodity", "Platinum"),
        "PALLADIUM": BucketInfo(8, "COMMODITY", "commodity", "Palladium"),
        "COPPER": BucketInfo(9, "COMMODITY", "commodity", "Copper"),
        "ALUMINUM": BucketInfo(10, "COMMODITY", "commodity", "Aluminum"),

        # Agriculture
        "WHEAT": BucketInfo(11, "COMMODITY", "commodity", "Wheat"),
        "CORN": BucketInfo(12, "COMMODITY", "commodity", "Corn"),
        "SOYBEANS": BucketInfo(13, "COMMODITY", "commodity", "Soybeans"),
        "COFFEE": BucketInfo(14, "COMMODITY", "commodity", "Coffee"),
        "SUGAR": BucketInfo(15, "COMMODITY", "commodity", "Sugar"),
        "COCOA": BucketInfo(16, "COMMODITY", "commodity", "Cocoa"),
        "COTTON": BucketInfo(17, "COMMODITY", "commodity", "Cotton"),
    }

    def __init__(self):
        """Initialize the bucket mapper."""
        self.custom_mappings: Dict[str, Dict[str, BucketInfo]] = {
            "equity": {},
            "credit": {},
            "commodity": {},
        }

    def classify_equity_bucket(self, issuer: str) -> int:
        """
        Classify an equity issuer to a SIMM bucket.

        Args:
            issuer: Issuer identifier (e.g., ticker, name)

        Returns:
            Bucket number (1-12)
        """
        issuer_upper = issuer.upper()

        # Check custom mappings first
        if issuer_upper in self.custom_mappings["equity"]:
            return self.custom_mappings["equity"][issuer_upper].bucket_number

        # Check built-in mappings
        for key, info in self.EQUITY_BUCKET_MAPPINGS.items():
            if key in issuer_upper:
                return info.bucket_number

        # Default to bucket 8 (Large Developed - Financials)
        # This is a reasonable default for unclassified developed market equities
        return 8

    def classify_credit_bucket(
        self,
        issuer: str,
        is_qualifying: bool,
    ) -> Tuple[int, str]:
        """
        Classify a credit issuer to a SIMM bucket.

        Args:
            issuer: Issuer identifier
            is_qualifying: True for Credit Qualifying (IG), False for CNQ (HY/NR)

        Returns:
            Tuple of (bucket_number, credit_type)
        """
        issuer_upper = issuer.upper()

        # Check custom mappings first
        if issuer_upper in self.custom_mappings["credit"]:
            info = self.custom_mappings["credit"][issuer_upper]
            return info.bucket_number, info.risk_class.lower()

        # Check built-in mappings
        for key, info in self.CREDIT_BUCKET_MAPPINGS.items():
            if key in issuer_upper:
                return info.bucket_number, info.risk_class.lower()

        # Default classification
        if is_qualifying:
            # Default IG corporate
            return 3, "credit"  # Assume corporate IG bucket
        else:
            # Default HY/NR
            return 2, "cnq"

    def classify_commodity_bucket(self, commodity: str) -> int:
        """
        Classify a commodity to a SIMM bucket.

        Args:
            commodity: Commodity identifier

        Returns:
            Bucket number (1-17)
        """
        commodity_upper = commodity.upper()

        # Check custom mappings first
        if commodity_upper in self.custom_mappings["commodity"]:
            return self.custom_mappings["commodity"][commodity_upper].bucket_number

        # Check built-in mappings
        for key, info in self.COMMODITY_BUCKET_MAPPINGS.items():
            if key in commodity_upper:
                return info.bucket_number

        # Default to bucket 1 (Crude Oil)
        # This is a reasonable default for unclassified energy commodities
        return 1

    def register_custom_mapping(
        self,
        classification_type: str,
        identifier: str,
        bucket_info: BucketInfo,
    ) -> None:
        """
        Register a custom bucket mapping.

        Args:
            classification_type: "equity", "credit", or "commodity"
            identifier: The identifier to map (e.g., ticker, issuer)
            bucket_info: Bucket information
        """
        if classification_type not in self.custom_mappings:
            raise ValueError(f"Unknown classification type: {classification_type}")

        identifier_upper = identifier.upper()
        self.custom_mappings[classification_type][identifier_upper] = bucket_info

    def get_bucket_info(
        self,
        classification_type: str,
        identifier: str,
    ) -> Optional[BucketInfo]:
        """
        Get bucket information for a given identifier.

        Args:
            classification_type: "equity", "credit", or "commodity"
            identifier: The identifier to look up

        Returns:
            BucketInfo if found, None otherwise
        """
        identifier_upper = identifier.upper()

        # Check custom mappings first
        if identifier_upper in self.custom_mappings[classification_type]:
            return self.custom_mappings[classification_type][identifier_upper]

        # Check built-in mappings
        if classification_type == "equity":
            for key, info in self.EQUITY_BUCKET_MAPPINGS.items():
                if key in identifier_upper:
                    return info
        elif classification_type == "credit":
            for key, info in self.CREDIT_BUCKET_MAPPINGS.items():
                if key in identifier_upper:
                    return info
        elif classification_type == "commodity":
            for key, info in self.COMMODITY_BUCKET_MAPPINGS.items():
                if key in identifier_upper:
                    return info

        return None

    def is_qualifying_credit(self, issuer: str) -> bool:
        """
        Determine if a credit issuer is Qualifying (IG) or Non-Qualifying (HY/NR).

        This is a simplified heuristic based on common patterns.
        Production code would use actual credit ratings.

        Args:
            issuer: Issuer identifier

        Returns:
            True if Qualifying (IG), False if Non-Qualifying (HY/NR)
        """
        # Sovereigns are typically considered Qualifying
        sovereign_keywords = ["TREASURY", "GILT", "BUND", "JGB", "SOVEREIGN"]
        issuer_upper = issuer.upper()

        for keyword in sovereign_keywords:
            if keyword in issuer_upper:
                return True

        # Major investment grade issuers (simplified)
        ig_keywords = ["APPLE", "MICROSOFT", "GOOGLE", "WALMART", "PROCTER"]
        for keyword in ig_keywords:
            if keyword in issuer_upper:
                return True

        # Default to IG for major financial institutions
        if "BANK" in issuer_upper or "JPM" in issuer_upper:
            return True

        # Everything else is HY/NQ
        return False
