import os, argparse, json, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def metrics_from_equity(equity_csv: str):
    eq = pd.read_csv(equity_csv, parse_dates=["date"]).set_index("date")["equity"].dropna()
    if eq.empty: return {}
    ret = eq.pct_change().dropna()
    ann = 252
    cagr = (eq.iloc[-1]/eq.iloc[0])**(ann/len(eq)) - 1
    sharpe = (ret.mean()*ann) / (ret.std(ddof=1)*np.sqrt(ann)) if ret.std(ddof=1)>0 else np.nan
    rollmax = eq.cummax()
    maxdd = (eq/rollmax - 1.0).min()
    # turnover approx from trades if present
    return dict(CAGR=cagr, Sharpe=sharpe, MaxDD=maxdd)

def plot_equity(equity_csv: str, out_png: str, title: str):
    eq = pd.read_csv(equity_csv, parse_dates=["date"]).set_index("date")["equity"]
    plt.figure()
    eq.plot(title=title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def run(out_root: str, run_id: str, out_csv: str):
    rows=[]
    for sym in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, sym, run_id, "equity.csv")
        if os.path.exists(p):
            m = metrics_from_equity(p)
            rows.append(dict(symbol=sym, run_id=run_id, **{k: float(v) for k,v in m.items()}))
            plot_equity(p, os.path.join(out_root, sym, run_id, "equity.png"), f"{sym} — {run_id}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"Saved summary → {out_csv}")
    else:
        print("No results found.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="backtests/out")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-csv", default="backtests/summary.csv")
    args = ap.parse_args()
    run(args.out_root, args.run_id, args.out_csv)
