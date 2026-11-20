# Bitcoin Volatility Forecasting: HAR vs GARCH vs a Naive Benchmark

Predicting Bitcoin's price direction is close to impossible. Predicting **how much** it will move is a different story, because volatility clusters: rough days follow rough days.

This project builds a daily volatility measure from 5-minute prices and tests whether two standard volatility models beat the simplest possible forecast: *tomorrow = today*.

## What it does
1. **Data:** 5-minute BTC/USDT candles, 2018–2025, from Binance's public archive ([data.binance.vision](https://data.binance.vision)).
2. **Realized variance:** daily RV = sum of squared 5-minute log returns; incomplete days are dropped rather than understated.
3. **Forecasts (next day, strictly out of sample, 2023–2025):**
   - **Naive:** tomorrow's RV = today's RV
   - **HAR** (Corsi, 2009): daily, weekly and monthly RV components, rolling 1,000-day OLS
   - **GARCH(1,1)** with Student-t errors (Bollerslev, 1986) on daily close-to-close returns
4. **Evaluation:** MSE and QLIKE losses (Patton, 2011), Diebold–Mariano tests with the Harvey–Leybourne–Newbold correction, and a calm vs turbulent day split.

## Results
Out-of-sample, next-day forecasts, 1,095 days (Jan 2023 – Dec 2025). Lower loss is better.

| Model | MSE | vs naive | DM p-value | QLIKE | vs naive | DM p-value |
|---|---|---|---|---|---|---|
| Naive | 101.18 | – | – | 0.579 | – | – |
| HAR | 68.11 | −32.7% | 0.017 | 0.376 | −35.1% | 0.001 |
| GARCH(1,1) | 71.33 | −29.5% | 0.062 | 0.438 | −24.3% | 0.003 |

QLIKE on calm vs turbulent days (split at the median of realized variance):

| Model | Calm days | Turbulent days |
|---|---|---|
| Naive | 0.362 | 0.796 |
| HAR | 0.505 | 0.246 |
| GARCH(1,1) | 0.533 | 0.343 |

**Takeaways**
- Daily returns have almost no memory (average autocorrelation over 30 lags: 0.002), while daily realized variance does (0.145): volatility clusters.
- Both models beat "tomorrow = today" on average; HAR, a simple regression on intraday-based volatility, had the lowest loss on both measures.
- The naive forecast was best on days that turned out calm. The models earned their advantage on turbulent days.

![Calm vs turbulent days](figures/regime_comparison.png)

![Forecast vs reality, 2025](figures/forecast_vs_reality_2025.png)

Full tables: `results.csv` and `results_by_regime.csv`.

## Run it
Open `Bitcoin_Volatility_Forecasting.ipynb` in Google Colab or Jupyter and run all cells (or `python btc_volatility.py`). The first run downloads about 840k rows from Binance's public archive; the models then run in about a minute.

```
pip install -r requirements.txt
```

## References
- Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review, 39*(4), 885–905.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics, 31*(3), 307–327.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics, 7*(2), 174–196.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting, 13*(2), 281–291.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics, 160*(1), 246–256.
