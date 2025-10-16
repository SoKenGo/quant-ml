import os, json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
from pandas.errors import EmptyDataError

def load_metrics(run_id, out_root="backtests/out"):
    rows=[]
    for sym_path in sorted(glob(f"{out_root}/*/{run_id}")):
        symbol = sym_path.split("/")[-2]
        mpath = os.path.join(sym_path, "metrics.json")
        ep = os.path.join(sym_path, "equity.csv")
        tp = os.path.join(sym_path, "trades.csv")
        if not (os.path.exists(mpath) and os.path.exists(ep)):
            continue
        with open(mpath) as f: m=json.load(f)
        eq = pd.read_csv(ep, parse_dates=["date"]).set_index("date")["equity"].dropna()

        # read trades safely
        if os.path.exists(tp) and os.path.getsize(tp) > 0:
            try:
                tr = pd.read_csv(tp)
            except EmptyDataError:
                tr = pd.DataFrame(columns=["pnlcomm"])
        else:
            tr = pd.DataFrame(columns=["pnlcomm"])
        if "pnlcomm" not in tr.columns:
            tr["pnlcomm"] = pd.Series(dtype=float)

        ret = eq.pct_change().dropna()
        ann=252
        cagr = (eq.iloc[-1]/eq.iloc[0])**(ann/len(eq)) - 1
        sharpe = (ret.mean()*ann) / (ret.std(ddof=1)*np.sqrt(ann)) if ret.std(ddof=1)>0 else np.nan
        rollmax = eq.cummax()
        maxdd = (eq/rollmax - 1.0).min()
        ntr = len(tr)
        wins = int((tr["pnlcomm"] > 0).sum()) if ntr>0 else 0
        losses = ntr - wins
        hit = wins / ntr if ntr>0 else np.nan
        pf = (tr.loc[tr["pnlcomm"]>0,"pnlcomm"].sum() /
              abs(tr.loc[tr["pnlcomm"]<0,"pnlcomm"].sum())) if (tr["pnlcomm"]<0).any() else np.nan
        rows.append(dict(symbol=symbol, CAGR=cagr, Sharpe=sharpe, MaxDD=maxdd,
                         Trades=ntr, HitRate=hit, ProfitFactor=pf))
    return pd.DataFrame(rows)

def plot_rolling(symbol, run_id, out_root="backtests/out", out_dir=None, window=126):
    ep = f"{out_root}/{symbol}/{run_id}/equity.csv"
    if not os.path.exists(ep): return
    eq = pd.read_csv(ep, parse_dates=["date"]).set_index("date")["equity"].dropna()
    ret = eq.pct_change().dropna()
    rsh = (ret.rolling(window).mean() / ret.rolling(window).std(ddof=1)) * np.sqrt(252)
    rdd = (eq / eq.cummax() - 1.0)
    out_dir = out_dir or f"{out_root}/{symbol}/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(); eq.plot(); plt.title(f"{symbol} Equity"); plt.tight_layout(); plt.savefig(f"{out_dir}/equity.png"); plt.close()
    plt.figure(); rsh.plot(); plt.title(f"{symbol} Rolling Sharpe ({window}d)"); plt.tight_layout(); plt.savefig(f"{out_dir}/rolling_sharpe.png"); plt.close()
    plt.figure(); rdd.plot(); plt.title(f"{symbol} Drawdown"); plt.tight_layout(); plt.savefig(f"{out_dir}/drawdown.png"); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", default="backtests/out")
    ap.add_argument("--summary-csv", default=None)
    args=ap.parse_args()
    df = load_metrics(args.run_id, args.out_root).sort_values("Sharpe", ascending=False)
    os.makedirs("backtests/analysis", exist_ok=True)
    out_csv = args.summary_csv or f"backtests/analysis/summary_{args.run_id}.csv"
    df.to_csv(out_csv, index=False)
    for sym in df["symbol"]:
        plot_rolling(sym, args.run_id, args.out_root)
    print("Saved:", out_csv)
    print(df.head(12).to_string(index=False))

if __name__=="__main__":
    main()
