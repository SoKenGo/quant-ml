# backtests/portfolio_h1_report.py
import os, glob, json, argparse
import pandas as pd

def load_equity(run_root="backtests_h1/out", symbol=None):
    rows=[]
    for sym_dir in glob.glob(os.path.join(run_root, symbol or "*")):
        sym=os.path.basename(sym_dir)
        for run_dir in glob.glob(os.path.join(sym_dir,"*")):
            eq_path=os.path.join(run_dir,"equity.csv")
            if not os.path.exists(eq_path): continue
            df=pd.read_csv(eq_path)
            df["date"]=pd.to_datetime(df["date"])
            df=df.set_index("date").sort_index()
            df.rename(columns={"equity": sym}, inplace=True)
            rows.append(df[[sym]])
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="backtests_h1/out")
    ap.add_argument("--symbols", nargs="+", default=["TSLA","RKLB","NVDA"])
    args=ap.parse_args()

    eqs=[]
    for s in args.symbols:
        parts=load_equity(args.root, s)
        if parts:
            # pick the latest run by folder name
            parts.sort(key=lambda d: d.index.max())
            eqs.append(parts[-1])

    if not eqs:
        print("No equities found.")
        return

    aligned=pd.concat(eqs, axis=1).ffill().dropna(how="all")
    # equal-weight portfolio
    port = aligned.pct_change().mean(axis=1).add(1).cumprod()
    out=pd.DataFrame({"portfolio":port})
    print(out.tail(10))
    out.to_csv("backtests_h1/portfolio_equity.csv", index=True)
    print("Wrote backtests_h1/portfolio_equity.csv")

if __name__=="__main__":
    main()
