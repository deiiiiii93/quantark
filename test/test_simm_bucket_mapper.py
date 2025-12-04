"""
Tests for SIMM bucket classification.
"""

import pytest
from simm.engines.classification.bucket_mapper import BucketMapper, BucketInfo


class TestBucketMapper:
    """Test the BucketMapper class."""

    def test_init(self):
        """Test initialization."""
        mapper = BucketMapper()
        assert mapper is not None
        assert len(mapper.custom_mappings) == 3

    def test_classify_equity_bucket_apple(self):
        """Test classifying AAPL to bucket 5 (Technology)."""
        mapper = BucketMapper()
        bucket = mapper.classify_equity_bucket("AAPL")
        assert bucket == 5  # Technology

    def test_classify_equity_bucket_spy(self):
        """Test classifying SPY to bucket 11 (ETF)."""
        mapper = BucketMapper()
        bucket = mapper.classify_equity_bucket("SPY")
        assert bucket == 11  # ETF/Index

    def test_classify_equity_bucket_unknown(self):
        """Test classifying unknown issuer defaults to bucket 8."""
        mapper = BucketMapper()
        bucket = mapper.classify_equity_bucket("UNKNOWN_ISSUER")
        assert bucket == 8  # Default bucket

    def test_classify_credit_bucket_sovereign(self):
        """Test classifying sovereign to bucket 1."""
        mapper = BucketMapper()
        bucket, credit_type = mapper.classify_credit_bucket("US_TREASURY", is_qualifying=True)
        assert bucket == 1
        assert credit_type == "credit"

    def test_classify_credit_bucket_apple_ig(self):
        """Test classifying Apple as Credit Qualifying."""
        mapper = BucketMapper()
        bucket, credit_type = mapper.classify_credit_bucket("APPLE", is_qualifying=True)
        assert bucket == 5  # Technology bucket
        assert credit_type == "credit"

    def test_is_qualifying_credit_sovereign(self):
        """Test that sovereigns are considered Qualifying."""
        mapper = BucketMapper()
        assert mapper.is_qualifying_credit("US_TREASURY") is True
        assert mapper.is_qualifying_credit("GERMAN_BUND") is True

    def test_is_qualifying_credit_major_issuer(self):
        """Test that major issuers are considered Qualifying."""
        mapper = BucketMapper()
        assert mapper.is_qualifying_credit("APPLE") is True
        assert mapper.is_qualifying_credit("MICROSOFT") is True

    def test_is_qualifying_credit_bank(self):
        """Test that banks are considered Qualifying."""
        mapper = BucketMapper()
        assert mapper.is_qualifying_credit("JPMORGAN") is True
        assert mapper.is_qualifying_credit("BANK_OF_AMERICA") is True

    def test_is_qualifying_credit_default_hy(self):
        """Test default classification as HY."""
        mapper = BucketMapper()
        # Unknown issuer without clear patterns should default to HY
        assert mapper.is_qualifying_credit("UNKNOWN_CORPORATE") is False

    def test_classify_commodity_bucket_oil(self):
        """Test classifying Oil to bucket 1."""
        mapper = BucketMapper()
        bucket = mapper.classify_commodity_bucket("OIL")
        assert bucket == 1

    def test_classify_commodity_bucket_gold(self):
        """Test classifying Gold to bucket 5."""
        mapper = BucketMapper()
        bucket = mapper.classify_commodity_bucket("GOLD")
        assert bucket == 5

    def test_classify_commodity_bucket_unknown(self):
        """Test classifying unknown commodity defaults to bucket 1."""
        mapper = BucketMapper()
        bucket = mapper.classify_commodity_bucket("UNKNOWN_COMMODITY")
        assert bucket == 1

    def test_register_custom_mapping(self):
        """Test registering custom bucket mappings."""
        mapper = BucketMapper()

        custom_bucket = BucketInfo(
            bucket_number=99,
            risk_class="EQUITY",
            classification_type="equity",
            description="Custom Equity Bucket",
        )

        mapper.register_custom_mapping("equity", "CUSTOM_TICKER", custom_bucket)

        # Check that custom mapping is registered
        bucket = mapper.classify_equity_bucket("CUSTOM_TICKER")
        assert bucket == 99

    def test_get_bucket_info(self):
        """Test getting bucket information."""
        mapper = BucketMapper()

        # Check existing mapping
        info = mapper.get_bucket_info("equity", "AAPL")
        assert info is not None
        assert info.bucket_number == 5
        assert info.classification_type == "equity"

        # Check non-existent mapping
        info = mapper.get_bucket_info("equity", "UNKNOWN_TICKER")
        assert info is None

    def test_equity_bucket_mappings(self):
        """Test that all expected equity bucket mappings exist."""
        mapper = BucketMapper()

        # Test major ETFs
        assert mapper.classify_equity_bucket("SPY") == 11
        assert mapper.classify_equity_bucket("QQQ") == 11
        assert mapper.classify_equity_bucket("VTI") == 11

        # Test major US stocks
        assert mapper.classify_equity_bucket("MSFT") == 5
        assert mapper.classify_equity_bucket("GOOGL") == 5

        # Test indices
        assert mapper.classify_equity_bucket("INDEX") == 11

    def test_credit_bucket_mappings(self):
        """Test that all expected credit bucket mappings exist."""
        mapper = BucketMapper()

        # Test sovereigns
        bucket, credit_type = mapper.classify_credit_bucket("US_TREASURY", is_qualifying=True)
        assert bucket == 1
        assert credit_type == "credit"

        # Test financials
        bucket, credit_type = mapper.classify_credit_bucket("JPMORGAN", is_qualifying=True)
        assert bucket == 2
        assert credit_type == "credit"

    def test_commodity_bucket_mappings(self):
        """Test that all expected commodity bucket mappings exist."""
        mapper = BucketMapper()

        # Test energy
        assert mapper.classify_commodity_bucket("BRENT") == 1
        assert mapper.classify_commodity_bucket("WTI") == 1
        assert mapper.classify_commodity_bucket("NATURAL_GAS") == 2

        # Test metals
        assert mapper.classify_commodity_bucket("GOLD") == 5
        assert mapper.classify_commodity_bucket("SILVER") == 6
        assert mapper.classify_commodity_bucket("COPPER") == 9

        # Test agriculture
        assert mapper.classify_commodity_bucket("WHEAT") == 11
        assert mapper.classify_commodity_bucket("CORN") == 12
