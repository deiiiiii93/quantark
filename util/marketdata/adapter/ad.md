adapter to fetch market data from different sources
for example: Bloomberg, Wind
Each source may provide files or streaming api with specified fields and formats.
As we don't know exact info about how Bloomberg and Wind data is like at this moment, we just create a mock adapter here, and leave placeholder and abstract apis for real sources.

The data acquired via Adapter will be fed into option hedging strategy backtest, which we will implement later.