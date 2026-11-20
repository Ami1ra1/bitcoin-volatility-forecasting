# %% [markdown]
# # Can we forecast how much Bitcoin will move tomorrow?
# **Realized volatility from 5-minute data: HAR vs GARCH vs a naive benchmark (BTC/USDT, 2018–2025)**
#
# Price direction is close to unpredictable, but the *size* of the moves is not:
# volatility clusters. This notebook builds a daily volatility measure from
# 5-minute prices and asks a simple question:
# **do standard volatility models actually beat "tomorrow = today"?**
#
# Steps
# 1. Download public 5-minute BTC/USDT candles from Binance (data.binance.vision)
# 2. Build daily realized variance (RV) from 5-minute log returns
# 3. Forecast next-day RV with three methods, strictly out of sample:
#    - **Naive**: tomorrow's RV = today's RV
#    - **HAR** (Corsi, 2009): daily, weekly and monthly RV averages, rolling OLS
#    - **GARCH(1,1)** with Student-t errors (Bollerslev, 1986) on daily close-to-close returns
# 4. Compare with MSE and QLIKE (Patton, 2011) and test differences with the
#    Diebold–Mariano test (Harvey–Leybourne–Newbold correction)
#
# Runs on Google Colab or locally: `pip install numpy pandas scipy arch matplotlib requests`

# %%
import io, os, time, zipfile
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model

SYMBOL = "BTCUSDT"
YEARS = range(2018, 2026)
DATA_FILE = f"{SYMBOL}_5m_2018_2025.csv"
MIN_BARS = 259          # a full day has 288 five-minute bars; keep days with >= 90%
WINDOW = 1000           # rolling estimation window (days)
OOS_START = "2023-01-01"  # out-of-sample evaluation period starts here
GARCH_REFIT = 20        # re-estimate GARCH parameters every 20 days
os.makedirs("figures", exist_ok=True)

# %% [markdown]
# ## 1. Data: 5-minute candles from Binance's public archive

# %%
BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_month(symbol, year, month, retries=3):
    url = f"{BASE}/{symbol}/5m/{symbol}-5m-{year}-{month:02d}.zip"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z, z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=COLS)
            if not str(df.iloc[0, 0]).isdigit():   # some files carry a header row
                df = df.iloc[1:]
            return df[["open_time", "close"]].astype(float)
        except Exception:
            time.sleep(2)
    return None


def download(symbol):
    from concurrent.futures import ThreadPoolExecutor
    months = [(y, m) for y in YEARS for m in range(1, 13)]
    frames, missing = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, ((y, m), f) in enumerate(zip(months, ex.map(lambda ym: fetch_month(symbol, *ym), months)), 1):
            (frames.append(f) if f is not None else missing.append(f"{y}-{m:02d}"))
            print(f"\r{k}/{len(months)} months downloaded", end="")
    print(f"\nMissing months: {missing or 'none'}")
    if len(frames) < 0.9 * len(months):
        raise RuntimeError("Too many months failed: data.binance.vision may be unreachable from here.")
    df = pd.concat(frames, ignore_index=True)
    # Binance switched from milliseconds to microseconds in 2025
    t = df["open_time"]
    df["timestamp"] = pd.to_datetime(np.where(t > 1e14, t / 1000, t), unit="ms")
    return df[["timestamp", "close"]].drop_duplicates("timestamp").sort_values("timestamp")


if not os.path.exists(DATA_FILE):
    download(SYMBOL).to_csv(DATA_FILE, index=False)

px = pd.read_csv(DATA_FILE)
px["timestamp"] = pd.to_datetime(px["timestamp"], format="mixed")  # some rows have milliseconds, some don't
px = px.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")["close"].astype(float)
print(f"{len(px):,} five-minute prices from {px.index.min()} to {px.index.max()}")

# %% [markdown]
# ## 2. Daily realized variance
# $RV_t = \sum_{i=1}^{M} r_{t,i}^2$, with $r_{t,i}$ the i-th 5-minute log return of day t (in %).
# Days with too many missing bars are set to missing rather than understated.

# %%
r5 = 100 * np.log(px).diff()
r5 = r5[r5.index.to_series().diff() == pd.Timedelta(minutes=5)]  # skip returns across data gaps
# a candle stamped t closes at t+5min, so each day's 288 returns sum to its close-to-close return
day = r5.index.floor("D")
rv = (r5 ** 2).groupby(day).sum()
bars = r5.groupby(day).size()
rv[bars < MIN_BARS] = np.nan
rv = rv.asfreq("D")

close_daily = px.resample("D").last()
ret_daily = 100 * np.log(close_daily).diff()   # close-to-close return, for GARCH

print("Days:", len(rv), "| dropped (incomplete):", int(rv.isna().sum()))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(np.sqrt(rv), lw=0.6, color="#1f4e79")
ax.set_title("Bitcoin daily realized volatility (from 5-minute returns)")
ax.set_ylabel("daily volatility, %")
fig.tight_layout(); fig.savefig("figures/01_realized_volatility.png", dpi=200); plt.show()

# Volatility clustering: returns have ~no memory, squared returns do
from statsmodels.tsa.stattools import acf
ac_ret = acf(ret_daily.dropna(), nlags=30)[1:]
ac_rv = acf(rv.dropna(), nlags=30)[1:]
print("Avg autocorrelation, lags 1-30 | returns: %.3f | realized variance: %.3f"
      % (ac_ret.mean(), ac_rv.mean()))

# %% [markdown]
# ## 3. Out-of-sample forecasts of tomorrow's RV
# Every forecast made on day t uses only information available at the end of day t.

# %%
# --- HAR features
X = pd.DataFrame({
    "d": rv,
    "w": rv.rolling(7, min_periods=6).mean(),
    "m": rv.rolling(30, min_periods=24).mean(),
})
y_next = rv.shift(-1)                      # target: RV of day t+1
dates = rv.index
fc = pd.DataFrame(index=dates, columns=["Naive", "HAR", "GARCH"], dtype=float)
fc["Naive"] = rv

Xv = np.column_stack([np.ones(len(X)), X.values])
yv = y_next.values
for i, t in enumerate(dates):
    if t < pd.Timestamp(OOS_START) or np.isnan(Xv[i]).any():
        continue
    # training pairs (X_s, RV_{s+1}) with s+1 <= t, i.e. s <= t-1
    lo = max(0, i - WINDOW)
    Xi, yi = Xv[lo:i], yv[lo:i]
    ok = ~np.isnan(Xi).any(axis=1) & ~np.isnan(yi)
    beta, *_ = np.linalg.lstsq(Xi[ok], yi[ok], rcond=None)
    fc.loc[t, "HAR"] = max(Xv[i] @ beta, 1e-6)   # keep variance forecasts positive

# --- GARCH(1,1), Student-t, rolling window, parameters re-estimated every GARCH_REFIT days
ret = ret_daily.dropna()
params, k = None, 0
for t in ret.index[ret.index >= pd.Timestamp(OOS_START)]:
    sample = ret.loc[:t].iloc[-WINDOW:]
    am = arch_model(sample, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    if params is None or k % GARCH_REFIT == 0:
        params = am.fit(disp="off", show_warning=False).params
    k += 1
    if k % 100 == 0:
        print(f"\rGARCH forecasts: {k}", end="")
    fc.loc[t, "GARCH"] = am.fix(params).forecast(horizon=1, reindex=False).variance.values[-1, 0]

res = fc.assign(actual=y_next).loc[OOS_START:].dropna()
print("Evaluation days:", len(res), "|", res.index.min().date(), "to", res.index.max().date())

# %% [markdown]
# ## 4. Evaluation
# - **MSE** punishes big misses on extreme days.
# - **QLIKE** $= \frac{RV}{\hat h} - \ln\frac{RV}{\hat h} - 1$ is the standard loss for variance
#   forecasts: robust to noise in the RV proxy (Patton, 2011) and harsh on under-predicting risk.
# - **Diebold–Mariano**: is the loss difference against the naive benchmark real or luck?

# %%
def qlike(y, f):
    return y / f - np.log(y / f) - 1


def dm_test(loss_a, loss_b, h=1):
    """H0: equal accuracy. Negative stat => A has lower loss. HLN-corrected, Newey-West variance."""
    d = np.asarray(loss_a - loss_b); n = len(d)
    lags = max(h - 1, int(n ** (1 / 3)))
    dc = d - d.mean(); v = dc @ dc / n
    for j in range(1, lags + 1):
        v += 2 * (1 - j / (lags + 1)) * (dc[j:] @ dc[:-j]) / n
    stat = d.mean() / np.sqrt(v / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


models = ["Naive", "HAR", "GARCH"]
L = {"MSE": {m: (res["actual"] - res[m]) ** 2 for m in models},
     "QLIKE": {m: qlike(res["actual"], res[m]) for m in models}}

rows = []
for m in models:
    row = {"model": m}
    for name in ("MSE", "QLIKE"):
        row[name] = L[name][m].mean()
        row[f"{name} vs naive %"] = 100 * (L[name][m].mean() / L[name]["Naive"].mean() - 1)
        if m != "Naive":
            s, p = dm_test(L[name][m], L[name]["Naive"])
            row[f"DM {name}"], row[f"p {name}"] = s, p
    rows.append(row)
table = pd.DataFrame(rows).set_index("model").round(4)
print(table.to_string())
table.to_csv("results.csv")

# Calm vs turbulent days (split at the median of realized variance)
hi = res["actual"] > res["actual"].median()
regime = pd.DataFrame({m: [L["QLIKE"][m][~hi].mean(), L["QLIKE"][m][hi].mean()] for m in models},
                      index=["calm days", "turbulent days"]).T.round(3)
print("\nQLIKE by regime:\n", regime.to_string())
regime.to_csv("results_by_regime.csv")

# %% [markdown]
# ## 5. Charts

# %%
last = res.loc["2025-01-01":].copy()
last.index = last.index + pd.Timedelta(days=1)   # res is indexed by forecast origin t; plot on target day t+1
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(np.sqrt(last["actual"]), color="#9aa5b1", lw=0.8, label="Actual (realized)")
ax.plot(np.sqrt(last["HAR"]), color="#1f4e79", lw=1.2, label="HAR forecast")
ax.plot(np.sqrt(last["GARCH"]), color="#e07b00", lw=1.2, label="GARCH forecast")
ax.set_title("Next-day Bitcoin volatility: forecasts vs reality (2025)")
ax.set_ylabel("daily volatility, %"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/02_forecasts_2025.png", dpi=200); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, name in zip(axes, ["MSE", "QLIKE"]):
    vals = table[f"{name} vs naive %"]
    ax.barh(vals.index, vals.values, color=["#9aa5b1", "#1f4e79", "#e07b00"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"{name}: % change vs naive (lower = better)")
fig.tight_layout(); fig.savefig("figures/03_loss_vs_naive.png", dpi=200); plt.show()
