# backtests/bt_engine_h1.py
# 1H event-driven backtest with next-open fills, ATR risk, regime gates, and R2 I/O.
import os, argparse, json, warnings
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import backtrader as bt
import yaml
import pyarrow.parquet as pq
from dotenv import load_dotenv

# 项目已有的 R2 客户端
from src.utils.r2_client import _fs, s3url

load_dotenv(override=True)
warnings.filterwarnings("ignore")

# ============== IO helpers ==============
def _read_parquet_rel(key: str) -> pd.DataFrame:
    """读取相对路径（会自动拼接桶名）"""
    fs = _fs()
    with fs.open(s3url(key), "rb") as f:
        return pq.read_table(f).to_pandas()

def _read_parquet_url(url: str) -> pd.DataFrame:
    """读取完整 s3:// URL（无需再拼接桶名）"""
    fs = _fs()
    with fs.open(url, "rb") as f:
        return pq.read_table(f).to_pandas()

def _list_parquet_urls(prefix_rel: str) -> List[str]:
    """
    列出某个前缀下的所有 .parquet 完整 URL。
    prefix_rel: 例如 'h1/symbol=TSLA/year=2024' 或 'h1/symbol=TSLA/year=2024/month=01'
    """
    fs = _fs()
    urls = []
    try:
        roots = fs.ls(s3url(prefix_rel))  # 返回 s3://bucket/... 形式
    except Exception:
        return urls
    for p in roots:
        if p.endswith(".parquet"):
            urls.append(p)
        else:
            # 尝试列出子目录（例如 month=MM/ 下的文件）
            try:
                subs = fs.ls(p)
                urls += [q for q in subs if q.endswith(".parquet")]
            except Exception:
                pass
    return urls

def _load_h1_from_r2(symbol: str, years: List[int]) -> pd.DataFrame:
    parts=[]
    for y in years:
        base=f"h1/symbol={symbol}/year={y}"
        urls=_list_parquet_urls(base)
        if not urls:
            for mm in range(1,13):
                urls += _list_parquet_urls(f"{base}/month={mm:02d}")
        for url in urls:
            try:
                df = _read_parquet_url(url)
                parts.append(df)
            except Exception as e:
                warnings.warn(f"[H1] failed to read {url}: {e}")
    if not parts:
        raise RuntimeError(f"[H1] no data for {symbol} years={years}")
    df = pd.concat(parts, ignore_index=True)

    def pick(names):
        for n in names:
            if n in df.columns: return n
        return None
    col_date = pick(["date","datetime","timestamp","time"])
    col_o    = pick(["open","o","Open"])
    col_h    = pick(["high","h","High"])
    col_l    = pick(["low","l","Low"])
    col_c    = pick(["close","adjClose","Adj Close","c","Close","adj_close","price"])
    col_v    = pick(["volume","v","Volume","vol"])

    need = dict(date=col_date, open=col_o, high=col_h, low=col_l, close=col_c, volume=col_v)
    miss=[k for k,v in need.items() if v is None]
    if miss:
        raise RuntimeError(f"[H1] {symbol} missing required columns: {miss}; have={list(df.columns)}")

    out = pd.DataFrame({
        "date":   pd.to_datetime(df[col_date], errors="coerce", utc=True).dt.tz_convert(None),
        "open":   pd.to_numeric(df[col_o], errors="coerce"),
        "high":   pd.to_numeric(df[col_h], errors="coerce"),
        "low":    pd.to_numeric(df[col_l], errors="coerce"),
        "close":  pd.to_numeric(df[col_c], errors="coerce"),
        "volume": pd.to_numeric(df[col_v], errors="coerce"),
    }).dropna(subset=["date","open","high","low","close"]) \
     .sort_values("date").drop_duplicates("date").set_index("date")
    return out


    def pick(names):
        for n in names:
            if n in df.columns: return n
        return None
    col_date = pick(["date","datetime","timestamp","time"])
    col_o    = pick(["open","o","Open"])
    col_h    = pick(["high","h","High"])
    col_l    = pick(["low","l","Low"])
    col_c    = pick(["close","adjClose","Adj Close","c","Close","adj_close","price"])
    col_v    = pick(["volume","v","Volume","vol"])

    need = dict(date=col_date, open=col_o, high=col_h, low=col_l, close=col_c, volume=col_v)
    miss=[k for k,v in need.items() if v is None]
    if miss:
        raise RuntimeError(f"[H1] {symbol} missing required columns: {miss}; have={list(df.columns)}")

    out = pd.DataFrame({
        "date":   pd.to_datetime(df[col_date], utc=False).astype("datetime64[ns]"),
        "open":   pd.to_numeric(df[col_o], errors="coerce"),
        "high":   pd.to_numeric(df[col_h], errors="coerce"),
        "low":    pd.to_numeric(df[col_l], errors="coerce"),
        "close":  pd.to_numeric(df[col_c], errors="coerce"),
        "volume": pd.to_numeric(df[col_v], errors="coerce"),
    }).dropna(subset=["open","high","low","close"]).sort_values("date").drop_duplicates("date").set_index("date")
    return out

def _load_probs_from_r2_h1(symbol: str) -> Optional[pd.Series]:
    """
    优先顺序：probs.parquet（与你的训练器 --calibration 对应），其次 probs_sigmoid、probs_none。
    兼容列名：'p' / 'prob' / 'prob_up'
    """
    base = f"models/xgb_oos_h1/symbol={symbol}"
    for fname in ["probs.parquet", "probs_sigmoid.parquet", "probs_none.parquet"]:
        key = f"{base}/{fname}"
        try:
            df = _read_parquet_rel(key)
            # 统一列名
            col = "p"
            if "p" not in df.columns:
                if "prob" in df.columns: col = "prob"
                elif "prob_up" in df.columns: col = "prob_up"
                else: continue
            s = pd.Series(df[col].astype(float).values,
                          index=pd.to_datetime(df["date"], utc=False).astype("datetime64[ns]"),
                          name="prob_up").sort_index()
            return s
        except Exception:
            continue
    warnings.warn(f"[PROB] no 1H OOS probs found for {symbol} under {base}")
    return None

def _load_hmm_regime_h1() -> Optional[pd.Series]:
    """
    尝试读取 1H 的 HMM regime（若无则返回 None，回测时放行）
    期望列：date, regime_bull（1=多头, 0=空头）
    """
    try:
        df = _read_parquet_rel("models/regime/qqq_hmm_h1.parquet")
        s = pd.Series(df["regime_bull"].astype(int).values,
                      index=pd.to_datetime(df["date"], utc=False).astype("datetime64[ns]"),
                      name="regime_bull").sort_index()
        return s
    except Exception as e:
        warnings.warn(f"[HMM] 1H regime file not available ({e}); pass-through.")
        return None

def _load_param_cfg(path: Optional[str], symbol: str) -> Dict[str, Any]:
    cfg = {"entry_thr": 0.55, "exit_thr": 0.50, "risk_pct": 0.005, "atr_mult": 2.0}
    if not path: return cfg
    try:
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        cfg.update(y.get("default") or {})
        cfg.update((y.get("overrides") or {}).get(symbol) or {})
    except Exception as e:
        warnings.warn(f"[PARAM] failed to read {path}: {e}")
    return cfg

# ============== Backtrader feed ==============
class PandasFeed(bt.feeds.PandasData):
    params = (("datetime", None), ("open", "open"), ("high", "high"),
              ("low", "low"), ("close", "close"), ("volume", "volume"),
              ("openinterest", None),)
    # 建议在 adddata 时传 timeframe=Minutes, compression=60

# ============== Strategy ==============
class ProbATRStrategyH1(bt.Strategy):
    params = dict(
        entry_thr=0.55, exit_thr=0.50,
        risk_pct=0.005, atr_mult=2.0, atr_len=14,
        max_holding=48,                 # 以“bar”为单位
        probs=None,                     # pd.Series (1H 概率，时间戳为 bar close)
        qqq_gate=None,                  # pd.Series(bool/int) 与 1H 时间对齐
        hmm_gate=None,                  # pd.Series(bool/int)
        dd_halt_pct=0.02,               # 按“自然日”触发的当日熔断
        atr_vol_cap=0.12,               # ATR/price 上限，过高波动拒绝开仓
    )

    def __init__(self):
        self.price = self.datas[0].close
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_len)

        # 状态
        self.cur_day = None              # 用于“当日熔断”重置
        self.bars_in_pos = 0
        self.max_close = None

        # 计数
        self.dd_halts = 0
        self.skipped_high_atr = 0
        self.regime_skips = 0
        self.regime_hmm_skips = 0

        # 记录
        self.prev_day_close_equity = None
        self.halt_today = False
        self.halt_counted_today = False
        self.equity_log: List[tuple] = []
        self.trade_log: List[dict] = []

        # 信号序列
        self.prob_series = self.p.probs if isinstance(self.p.probs, pd.Series) else None
        self.qqq_gate = self.p.qqq_gate if isinstance(self.p.qqq_gate, pd.Series) else None
        self.hmm_gate = self.p.hmm_gate if isinstance(self.p.hmm_gate, pd.Series) else None

    # ----- helpers -----
    @staticmethod
    def _bt_ts(data0) -> pd.Timestamp:
        # backtrader 的 datetime 数字转 Python dt，再转 naive pandas TS
        ts = bt.num2date(data0.datetime[0])
        return pd.Timestamp(ts).tz_localize(None)

    def _get_prob(self, ts: pd.Timestamp):
        if self.prob_series is None:
            return None
        idx = self.prob_series.index.searchsorted(ts, side="right") - 1
        if idx >= 0:
            return float(self.prob_series.iloc[idx])
        return None

    def _gate_ok(self, ts: pd.Timestamp):
        ok = True
        if self.qqq_gate is not None:
            i = self.qqq_gate.index.searchsorted(ts, side="right") - 1
            if i >= 0 and int(self.qqq_gate.iloc[i]) == 0:
                ok = False
        if self.hmm_gate is not None:
            i = self.hmm_gate.index.searchsorted(ts, side="right") - 1
            if i >= 0 and int(self.hmm_gate.iloc[i]) == 0:
                ok = False
        return ok

    # ----- order/trade hooks -----
    def notify_order(self, order):
        if order.status in [order.Completed]:
            ts = self._bt_ts(self.datas[0])
            action = "BUY" if order.isbuy() else "SELL"
            self.trade_log.append({
                "ts": ts.isoformat(),
                "action": action,
                "size": int(order.executed.size),
                "price": float(order.executed.price),
                "value": float(order.executed.value),
                "commission": float(order.executed.comm),
            })

    def notify_trade(self, trade):
        if trade.isclosed:
            ts = self._bt_ts(self.datas[0])
            self.trade_log.append({
                "ts": ts.isoformat(),
                "action": "CLOSE",
                "size": int(trade.size),
                "price": float(trade.price),
                "pnl": float(trade.pnl),
                "pnlcomm": float(trade.pnlcomm),
            })

    # ----- main event -----
    def next(self):
        ts = self._bt_ts(self.datas[0])
        day = ts.normalize()  # 自然日 00:00:00

        # 日切换：只用于“当日熔断”变量重置
        if self.cur_day is None or day != self.cur_day:
            self.cur_day = day
            self.halt_today = False
            self.halt_counted_today = False
            # 用“昨日收盘权益”阈值来判断当日熔断
            # 若你要严格以前一日最后一根 1H bar 的权益作为“昨日收盘”，可在外部聚合
            if self.equity_log:
                # equity_log[-1] 是上一 bar 的权益；此处近似为“昨日收盘”
                self.prev_day_close_equity = self.equity_log[-1][1]

        # 逐 bar 更新“在仓 bar 数”
        self.bars_in_pos = (self.bars_in_pos + 1) if self.position.size else 0

        # 当日熔断：与“昨日收盘权益”比较
        cur_equity = self.broker.getvalue()
        if self.prev_day_close_equity is not None:
            thr = self.prev_day_close_equity * (1.0 - float(self.p.dd_halt_pct))
            if cur_equity < thr:
                self.halt_today = True
                if not self.halt_counted_today:
                    self.dd_halts += 1
                    self.halt_counted_today = True

        p = self._get_prob(ts)

        # ENTRY（bar close 发出，下根 bar open 成交）
        if not self.position:
            if p is not None and p >= float(self.p.entry_thr):
                if not self._gate_ok(ts):
                    # 计数
                    if self.qqq_gate is not None:
                        i = self.qqq_gate.index.searchsorted(ts, side="right") - 1
                        if i >= 0 and int(self.qqq_gate.iloc[i]) == 0:
                            self.regime_skips += 1
                    if self.hmm_gate is not None:
                        i = self.hmm_gate.index.searchsorted(ts, side="right") - 1
                        if i >= 0 and int(self.hmm_gate.iloc[i]) == 0:
                            self.regime_hmm_skips += 1
                    return
                if self.halt_today:
                    return
                # ATR 波动率上限
                if self.atr[0] > 0 and (self.atr[0] / max(1e-9, self.price[0])) > float(self.p.atr_vol_cap):
                    self.skipped_high_atr += 1
                    return

                # 风险头寸 sizing
                equity = self.broker.getvalue()
                risk_dollars = equity * float(self.p.risk_pct)
                stop_dist = float(self.atr[0]) * float(self.p.atr_mult)
                if stop_dist <= 0:
                    return
                size = int(max(1, risk_dollars / stop_dist))
                cash_cap = int(self.broker.getcash() / max(1e-9, self.price[0]) * 0.95)
                size = max(1, min(size, cash_cap))
                if size > 0:
                    self.buy(size=size)  # 将在“下一根” 1H bar 的 open 成交
                    self.max_close = self.price[0]
        else:
            # EXIT: trailing ATR / prob 退出 / 最长持有
            self.max_close = max(self.max_close, self.price[0]) if self.max_close is not None else self.price[0]
            trail_stop = self.max_close - float(self.atr[0]) * float(self.p.atr_mult)
            exit_prob = (p is not None and p <= float(self.p.exit_thr))
            exit_trail = self.price[0] < trail_stop
            exit_time = self.bars_in_pos >= int(self.p.max_holding)
            if exit_prob or exit_trail or exit_time:
                self.close()

        # 记录逐 bar equity（以 close 近似）
        self.equity_log.append((ts.isoformat(timespec="seconds"), float(self.broker.getvalue())))

    def get_metrics_counts(self) -> Dict[str, int]:
        return dict(
            dd_halts=int(self.dd_halts),
            skipped_high_atr=int(self.skipped_high_atr),
            regime_skips=int(self.regime_skips),
            regime_hmm_skips=int(self.regime_hmm_skips),
        )

# ============== Runner ==============
def run(symbol: str,
        years: List[int],
        out_root: str,
        run_id: str,
        commission_per_share: float,
        slippage_bps: float,
        entry_thr: Optional[float],
        exit_thr: Optional[float],
        risk_pct: Optional[float],
        atr_mult: Optional[float],
        max_holding: int,
        use_param_cfg: Optional[str],
        qqq_regime: bool,
        hmm_regime: bool,
        qqq_symbol: str,
        dd_halt_pct: float,
        atr_vol_cap: float,
        push_to_r2: bool,
        use_next_open: bool = True):

    # 1) 参数合并（YAML → CLI 显式覆盖）
    cfg = _load_param_cfg(use_param_cfg, symbol)
    if entry_thr is not None:  cfg["entry_thr"] = entry_thr
    if exit_thr is not None:   cfg["exit_thr"] = exit_thr
    if risk_pct is not None:   cfg["risk_pct"] = risk_pct
    if atr_mult is not None:   cfg["atr_mult"] = atr_mult

    # 2) 数据（1H）
    px = _load_h1_from_r2(symbol, years)
    data = PandasFeed(dataname=px)
    # 指定时间框架为 60 分钟，有助于分析器和内部逻辑
    data._timeframe = bt.TimeFrame.Minutes
    data._compression = 60

    # 3) 辅助序列
    probs = _load_probs_from_r2_h1(symbol)

    qqq_gate = None
    if qqq_regime:
        try:
            # 优先用 1H QQQ 计算 EMA12/EMA24 上穿作为 regime_up
            qqq = _load_h1_from_r2(qqq_symbol, years)
            ema12 = qqq["close"].ewm(span=12, adjust=False).mean()
            ema24 = qqq["close"].ewm(span=24, adjust=False).mean()
            cond = (ema12 > ema24)
            qqq_gate = pd.Series(cond.astype(int).values, index=qqq.index)
        except Exception as e:
            warnings.warn(f"[QQQ] 1H regime gate fallback pass-through ({e})")

    hmm_gate = _load_hmm_regime_h1() if hmm_regime else None

    # 4) Broker & 回测器
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_coc(False)  # 禁止同bar成交，保证 next open 成交
    cerebro.adddata(data, name=symbol)
    cerebro.addstrategy(
        ProbATRStrategyH1,
        entry_thr=float(cfg["entry_thr"]),
        exit_thr=float(cfg["exit_thr"]),
        risk_pct=float(cfg["risk_pct"]),
        atr_mult=float(cfg["atr_mult"]),
        max_holding=int(max_holding),
        probs=probs, qqq_gate=qqq_gate, hmm_gate=hmm_gate,
        dd_halt_pct=float(dd_halt_pct),
        atr_vol_cap=float(atr_vol_cap),
    )

    # 成本
    comminfo = bt.CommInfoBase(commission=float(commission_per_share), stocklike=True, percabs=False)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.set_slippage_perc(float(slippage_bps) / 10000.0)

    # analyzers（tann ≈ 每年 1H bars 数，取 252*6.5 ≈ 1638）
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=True, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, tann=1638, _name='rets')

    # 5) 运行
    res = cerebro.run()
    strat = res[0]

    # 6) 导出
    outdir = os.path.join(out_root, symbol, run_id)
    os.makedirs(outdir, exist_ok=True)

    # equity.csv
    eq_df = pd.DataFrame(strat.equity_log, columns=["ts","equity"])
    eq_df.to_csv(os.path.join(outdir, "equity.csv"), index=False)

    # trades.csv
    tr_df = pd.DataFrame(strat.trade_log)
    if tr_df.empty:
        tr_df = pd.DataFrame(columns=["ts","action","size","price","value","commission","pnl","pnlcomm"])
    tr_df.to_csv(os.path.join(outdir, "trades.csv"), index=False)

    # metrics.json
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    dd = strat.analyzers.dd.get_analysis()
    rets = strat.analyzers.rets.get_analysis()
    metrics = dict(
        symbol=symbol,
        cagr=float(rets.get("rnorm100", 0.0))/100.0 if rets else None,
        sharpe=float(sharpe) if sharpe is not None else None,
        maxdd_pct=float(dd.max.drawdown) if dd else None,
    )
    metrics.update(strat.get_metrics_counts())
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== {symbol} 1H Results ({run_id}) ===")
    print(f"CAGR={metrics['cagr']}, Sharpe={metrics['sharpe']}, MaxDD={metrics['maxdd_pct']}%")
    print(f"Counts: dd_halts={metrics['dd_halts']}, skipped_high_atr={metrics['skipped_high_atr']}, "
          f"regime_skips={metrics['regime_skips']}, regime_hmm_skips={metrics['regime_hmm_skips']}")

    # 7) push to R2
    if push_to_r2:
        fs = _fs()
        base = f"backtests_h1/out/{symbol}/{run_id}"
        for fname in ["equity.csv","trades.csv","metrics.json"]:
            lp = os.path.join(outdir, fname)
            with open(lp, "rb") as fin, fs.open(s3url(f"{base}/{fname}"), "wb") as fout:
                fout.write(fin.read())
        print(f"Pushed to R2: s3://{os.getenv('R2_BUCKET','quant-ml')}/{base}/")

# ============== CLI ==============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out-root", default="backtests_h1/out")
    ap.add_argument("--run-id", required=True)

    ap.add_argument("--commission-per-share", type=float, default=0.0038)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--entry-thr", type=float, default=None)
    ap.add_argument("--exit-thr", type=float, default=None)
    ap.add_argument("--risk-pct", type=float, default=None)
    ap.add_argument("--atr-mult", type=float, default=None)
    ap.add_argument("--max-holding", type=int, default=48)  # 48×1H ≈ 2个交易日

    ap.add_argument("--use-param-cfg", default=None, help="YAML path with default & overrides")

    # 风控
    ap.add_argument("--dd-halt-pct", type=float, default=0.02)
    ap.add_argument("--atr-vol-cap", type=float, default=0.12)

    # Regime
    ap.add_argument("--qqq-regime", action="store_true")
    ap.add_argument("--hmm-regime", action="store_true")
    ap.add_argument("--qqq-symbol", default="QQQ")

    # 兼容参数（不影响行为）
    ap.add_argument("--use-next-open", action="store_true")
    ap.add_argument("--push-to-r2", action="store_true")

    args = ap.parse_args()
    run(
        symbol=args.symbol, years=args.years, out_root=args.out_root, run_id=args.run_id,
        commission_per_share=args.commission_per_share, slippage_bps=args.slippage_bps,
        entry_thr=args.entry_thr, exit_thr=args.exit_thr, risk_pct=args.risk_pct, atr_mult=args.atr_mult,
        max_holding=args.max_holding, use_param_cfg=args.use_param_cfg,
        qqq_regime=args.qqq_regime, hmm_regime=args.hmm_regime, qqq_symbol=args.qqq_symbol,
        dd_halt_pct=args.dd_halt_pct, atr_vol_cap=args.atr_vol_cap,
        push_to_r2=args.push_to_r2, use_next_open=args.use_next_open
    )
