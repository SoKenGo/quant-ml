import os, argparse, json, numpy as np, pandas as pd
import backtrader as bt
from datetime import datetime
from dotenv import load_dotenv
from .feeds import load_eod_from_r2, load_probs_from_r2
from src.utils.r2_client import _fs, s3url

load_dotenv(override=True)

class PerShareCommission(bt.CommInfoBase):
    params = dict(commission=0.0038, stocklike=True, percabs=False)
    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission

# --- Custom Pandas feed with prob_up as a proper line ---
class PandasProb(bt.feeds.PandasData):
    lines = ('prob_up',)
    params = (
        ('prob_up', -1),
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'),
        ('close', 'close'), ('volume', 'volume'), ('openinterest', None),
    )

class ProbSignalStrategy(bt.Strategy):
    params = dict(
        symbol="TSLA",
        entry_thr=0.55,
        exit_thr=0.50,
        long_only=True,
        risk_pct=0.01,
        atr_period=14,
        atr_mult=2.0,
        max_holding=10,       # bars
        use_next_open=True,
    )
    def __init__(self):
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.holding_bars = 0
        self.trades_log = []
        # Use data line directly (robust)
        self.prob_line = self.data.prob_up
        # Execution convention
        if not self.p.use_next_open:
            self.cerebro.broker.set_coc(True)  # cheat-on-close
        # Debug counters
        self._sig_count = 0

    def next(self):
        prob = float(self.prob_line[0]) if not np.isnan(self.prob_line[0]) else np.nan
        if np.isnan(prob):
            return

        pos = self.getposition()
        cash = self.broker.getcash()
        value = self.broker.getvalue()

        atr = max(float(self.atr[0]), 1e-6)
        stop_dist = self.p.atr_mult * atr
        risk_dollars = self.p.risk_pct * value
        shares_risk = int(risk_dollars / stop_dist) if stop_dist > 0 else 0
        price = float(self.data.close[0])

        if self.p.long_only:
            if not pos.size:
                if prob >= self.p.entry_thr:
                    self._sig_count += 1
                    size = max(1, shares_risk)
                    if size * price > cash:
                        size = int(cash // price)
                    if size > 0:
                        self.buy(size=size)
                        self.holding_bars = 0
            else:
                self.holding_bars += 1
                if (prob <= self.p.exit_thr) or (self.holding_bars >= self.p.max_holding):
                    self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_log.append(dict(
                close_datetime=bt.num2date(self.data.datetime[0]).strftime("%Y-%m-%d"),
                pnl=float(trade.pnl),
                pnlcomm=float(trade.pnlcomm),
                size=int(trade.size),
            ))

def _metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if eq.empty: return {}
    ret = eq.pct_change().dropna()
    ann = 252
    cagr = (eq.iloc[-1]/eq.iloc[0])**(ann/len(eq)) - 1
    sharpe = (ret.mean()*ann) / (ret.std(ddof=1)*np.sqrt(ann)) if ret.std(ddof=1)>0 else np.nan
    rollmax = eq.cummax()
    dd = (eq/rollmax - 1.0).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(dd)}

def _push_dir_to_r2(local_dir: str, r2_prefix: str):
    fs = _fs()
    for fname in os.listdir(local_dir):
        lp = os.path.join(local_dir, fname)
        if not os.path.isfile(lp): 
            continue
        key = f"{r2_prefix.rstrip('/')}/{fname}"
        with open(lp, "rb") as fin, fs.open(s3url(key), "wb") as fout:
            fout.write(fin.read())

def run(symbol: str, years: list[int], out_root: str, run_id: str,
        commission_per_share: float, slippage_bps: float,
        entry_thr: float, exit_thr: float, risk_pct: float,
        atr_mult: float, max_holding: int, use_next_open: bool,
        push_to_r2: bool):

    eod = load_eod_from_r2(symbol, years)
    probs = load_probs_from_r2(symbol)
    if eod.empty:
        raise RuntimeError(f"No EOD found for {symbol}")
    if probs.empty:
        raise RuntimeError(f"No OOS probs found for {symbol}")

    prob = probs["prob_up"].reindex(eod.index).astype(float)
    # forward-fill single gaps (shouldn’t happen often)
    prob = prob.ffill()

    # Debug: how many bars exceed entry?
    sig_bars = int((prob >= entry_thr).sum())
    print(f"[{symbol}] bars>=entry({entry_thr:.2f}): {sig_bars}/{len(prob)}")

    # Build dataframe for feed
    df = eod.copy()
    df["prob_up"] = prob

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(
        ProbSignalStrategy,
        symbol=symbol,
        entry_thr=entry_thr,
        exit_thr=exit_thr,
        risk_pct=risk_pct,
        atr_mult=atr_mult,
        max_holding=max_holding,
        use_next_open=use_next_open,
    )

    data = PandasProb(dataname=df)
    cerebro.adddata(data)

    broker = cerebro.getbroker()
    comminfo = PerShareCommission(commission=commission_per_share)
    broker.addcommissioninfo(comminfo)
    # If set_slippage_perc not available, ignore
    if hasattr(broker, "set_slippage_perc"):
        broker.set_slippage_perc(slippage_bps / 1e4)
    broker.setcash(100000.0)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

    res = cerebro.run()
    strat = res[0]

    outdir = os.path.join(out_root, symbol, run_id)
    os.makedirs(outdir, exist_ok=True)

    rets = pd.Series(strat.analyzers.timereturn.get_analysis(), dtype=float)
    equity = (1.0 + rets.fillna(0)).cumprod()*100000.0
    equity.index = pd.to_datetime(equity.index)
    equity_df = pd.DataFrame({"equity": equity})
    equity_df.to_csv(os.path.join(outdir, "equity.csv"), index_label="date")

    trades = pd.DataFrame(getattr(strat, "trades_log", []))
    if trades.empty:
        trades = pd.DataFrame(columns=["close_datetime","pnl","pnlcomm","size"])
    trades.to_csv(os.path.join(outdir, "trades.csv"), index=False)

    dd = strat.analyzers.dd.get_analysis()
    metrics = _metrics_from_equity(equity_df["equity"])
    metrics.update({
        "symbol": symbol,
        "run_id": run_id,
        "commission_per_share": commission_per_share,
        "slippage_bps": slippage_bps,
        "entry_thr": entry_thr,
        "exit_thr": exit_thr,
        "risk_pct": risk_pct,
        "atr_mult": atr_mult,
        "max_holding": max_holding,
        "use_next_open": use_next_open,
        "signal_bars": int(sig_bars),
        "trades": int(len(trades)),
    })
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if push_to_r2:
        _push_dir_to_r2(outdir, f"{out_root}/{symbol}/{run_id}")
        print(f"Pushed backtest to R2 at s3://quant-ml/{out_root}/{symbol}/{run_id}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--out-root", default="backtests/out")
    p.add_argument("--run-id", default=None)
    p.add_argument("--commission-per-share", type=float, default=0.0038)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--entry-thr", type=float, default=0.55)
    p.add_argument("--exit-thr", type=float, default=0.50)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--atr-mult", type=float, default=2.0)
    p.add_argument("--max-holding", type=int, default=10)
    p.add_argument("--use-next-open", action="store_true")
    p.add_argument("--push-to-r2", action="store_true")
    args = p.parse_args()

    run_id = args.run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run(
        symbol=args.symbol, years=args.years, out_root=args.out_root, run_id=run_id,
        commission_per_share=args.commission_per_share, slippage_bps=args.slippage_bps,
        entry_thr=args.entry_thr, exit_thr=args.exit_thr, risk_pct=args.risk_pct,
        atr_mult=args.atr_mult, max_holding=args.max_holding, use_next_open=args.use_next_open,
        push_to_r2=args.push_to_r2
    )
