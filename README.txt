Gold Pattern Explorer — Volatility Breakout Strategy

Included files:
- app.py: strategy version based on the app.py you supplied.
- requirements.txt: Python dependencies.

Deployment:
1. Download and extract this ZIP.
2. Upload app.py and requirements.txt to the ROOT of a new GitHub repository.
   (Rename gold_pattern_explorer_strategy_app.py to app.py first, or use the app.py file inside the ZIP.)
3. Deploy that repository using Streamlit Community Cloud, selecting app.py.

Preserved:
- Existing XAUS automatic XAU/USD hourly data fetching.
- Existing CSV upload fallback.
- Existing Historical Matching section.

Strategy rules:
- Volatility = High - Low.
- Each completed hourly candle is compared only with the previous 24 candles.
- Bullish signal: wait for a retracement to 38.2% from the candle high, buy; stop at the candle low; target at 127.2% extension above the high.
- Bearish signal: wait for a retracement to 38.2% from the candle low, sell; stop at the candle high; target at 127.2% extension below the low.
- Default: wait up to 12 candles for entry; do not enter on the signal candle; one position at a time.
- If stop and target are both touched in the same hourly candle, stop is assumed first (conservative assumption).
- Backtest results exclude spread, commission and slippage and do not guarantee future profitability.
