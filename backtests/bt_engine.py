# backtests/bt_engine.py
import argparse, glob
import pandas as pd
import backtrader as bt
from pathlib import Path
import os, pandas as pd

def load_symbol_df_from_r2(symbol: str, year: int):
    import s3fs
    fs = s3fs.S3FileSystem(
        key=os.getenv("R2_ACCESS_KEY_ID"),
        secret=os.getenv("R2_SECRET_ACCESS_KEY"),
        client_kwargs={"endpoint_url": os.getenv("R2_ENDPOINT")},
    )
    key = f"s3://{os.getenv('R2_BUCKET','quant-ml')}/eod/symbol={symbol}/year={year}/*.parquet"
    with fs.open(key, "rb") as f:
        df = pd.read_parquet(f)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[["open","high","low","adjClose","volume"]].dropna()


# ---------- Data adapter: read our Parquet into Backtrader ----------
class ParquetData(bt.feeds.PandasData):
    params = (
        ("datetime", "date"),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "adjClose"),   # use adjusted close if present
        ("volume", "volume"),
        ("openinterest", None),
    )

def load_symbol_df(symbol: str) -> pd.DataFrame:
    # fallback: prefer processed feature file, else raw eod file
    p1 = Path(f"data/processed/feat_{symbol}.parquet")
    p2 = Path(f"data/raw/eod_{symbol}.parquet")
    fp = p1 if p1.exists() else p2
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet for {symbol}: {p1} or {p2}")
    df = pd.read_parquet(fp)
    # standardize required cols
    ren = {}
    if "adjClose" not in df.columns and "close" in df.columns:
        ren["close"] = "adjClose"
    if "date" not in df.columns:
        # some APIs use 'time'/'timestamp'
        for c in ("time", "timestamp", "Date"):
            if c in df.columns:
                ren[c] = "date"
                break
    if ren:
        df = df.rename(columns=ren)
    # index must be datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    # Use Regular Trading Hours only if a 'close' session flag exists — skip now
    return df[["open", "high", "low", "adjClose", "volume"]].dropna()

# ---------- Strategy: bar-close entries, risk-based sizing, ATR stop ----------
class ProbOrMACross(bt.Strategy):
    params = dict(
        risk_pct=0.01,          # % of equity risked per trade
        atr_len=14,
        atr_mult=2.2,           # stop distance = ATR * mult
        ma_fast=10,
        ma_slow=20,
        prob_threshold=0.55,    # if using ML probs (optional)
        use_probs=False,        # start False; flip True when you have CSV probs
        probs_csv=None,         # CSV with columns: date, prob (0..1)
        slippage_warn=True,
    )

    def __init__(self):
        self.price = self.datas[0].close
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_len)
        self.ma_fast = bt.ind.SMA(self.data.close, period=self.p.ma_fast)
        self.ma_slow = bt.ind.SMA(self.data.close, period=self.p.ma_slow)
        self.crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)

        # load optional probabilities aligned by datetime
        self.prob_map = None
        if self.p.use_probs and self.p.probs_csv:
            probs = pd.read_csv(self.p.probs_csv, parse_dates=["date"])
            probs = probs.set_index("date").sort_index()
            self.prob_map = probs["prob"]

    def _current_prob(self):
        if not self.prob_map:
            return None
        ts = pd.Timestamp(self.datas[0].datetime.datetime(0)).tz_localize(None)
        # find last known prob at or before current bar
        idx = self.prob_map.index.searchsorted(ts, side="right") - 1
        if idx >= 0:
            return float(self.prob_map.iloc[idx])
        return None

    def next(self):
        if not self.position:
            # Signal: ML prob -> else MA cross
            p = self._current_prob() if self.p.use_probs else None
            long_signal = (p is not None and p >= self.p.prob_threshold) or (self.crossover > 0)

            if long_signal and self.atr[0] > 0:
                cash = self.broker.getcash()
                eq   = self.broker.getvalue()
                risk_dollars = eq * self.p.risk_pct
                stop_dist = self.atr[0] * self.p.atr_mult
                if stop_dist <= 0:
                    return
                size = int(risk_dollars / stop_dist)
                # ensure at least 1 share and don’t exceed cash
                if size <= 0:
                    size = max(1, int(cash / self.price[0] * 0.95))
                else:
                    size = min(size, int(cash / self.price[0] * 0.95))
                if size > 0:
                    self.buy(size=size)

        else:
            # Exit if MA cross down OR price trails by ATR*mult from max
            # Simple trailing stop using highest close since entry
            if not hasattr(self, "max_close"):
                self.max_close = self.price[0]
            self.max_close = max(self.max_close, self.price[0])
            trail_stop = self.max_close - self.atr[0] * self.p.atr_mult

            exit_signal = (self.crossover < 0) or (self.price[0] < trail_stop)
            if exit_signal:
                self.close()

# ---------- Runner ----------
def run(symbol: str, cash: float, commission_per_share: float, slip_pct: float, fromdate=None, todate=None, **kwargs):
    df = load_symbol_df(symbol)

    cerebro = bt.Cerebro(stdstats=False)
    dfeed = ParquetData(dataname=df, fromdate=fromdate, todate=todate)
    cerebro.adddata(dfeed, name=symbol)

    cerebro.addstrategy(ProbOrMACross, **kwargs)

    # Broker: commissions + slippage
    # Commission per share (e.g., 0.0038 USD/share)
    comminfo = bt.CommInfoBase(
        commission=commission_per_share,
        stocklike=True,
        percabs=False
    )
    cerebro.broker.addcommissioninfo(comminfo)

    # Realistic slippage ~ 5 bps (adjust as needed)
    cerebro.broker.set_slippage_perc(slip_pct)

    cerebro.broker.setcash(cash)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=True, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, tann=252, _name='rets')

    res = cerebro.run()
    strat = res[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    dd = strat.analyzers.dd.get_analysis()
    rets = strat.analyzers.rets.get_analysis()
    print(f"\n=== {symbol} Results ===")
    print(f"Start Cash: {cash:,.2f} | End Equity: {cerebro.broker.getvalue():,.2f}")
    print(f"Sharpe: {sharpe:.3f} | MaxDrawdown: {dd.max.drawdown:.2f}% | CAGR: {rets.get('rnorm100', 0):.2f}%")
    t = strat.analyzers.trades.get_analysis()
    if t and 'won' in t and 'lost' in t:
        wins = t['won'].get('total', 0)
        losses = t['lost'].get('total', 0)
        print(f"Trades: {wins+losses} | Win/Loss: {wins}/{losses}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NVDA")
    p.add_argument("--cash", type=float, default=10000)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--atr_mult", type=float, default=2.2)
    p.add_argument("--slip_pct", type=float, default=0.0005)   # 5 bps
    p.add_argument("--comm_per_sh", type=float, default=0.0038)
    p.add_argument("--use_probs", action="store_true")
    p.add_argument("--probs_csv", default=None)
    args = p.parse_args()

    run(
        symbol=args.symbol,
        cash=args.cash,
        commission_per_share=args.comm_per_sh,
        slip_pct=args.slip_pct,
        atr_mult=args.atr_mult,
        risk_pct=args.risk,
        use_probs=args.use_probs,
        probs_csv=args.probs_csv
    )
