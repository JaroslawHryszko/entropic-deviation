#!/usr/bin/env python
# stats.py
import argparse, pandas as pd, scipy.stats as ss, statsmodels.api as sm

def ar1(phi_series):
    return phi_series.autocorr(lag=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="output of ed.py")
    ap.add_argument("--out", default="results_Ftests.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []

    # F1: mean ED != 0  (one-sample t-test)
    t,p = ss.ttest_1samp(df["ED_mean"], 0.0)
    rows.append(("F1", p))

    # F2: temperature effect (one-way ANOVA)
    groups = [g["ED_mean"].values for _,g in df.groupby("temp")]
    f,p = ss.f_oneway(*groups)
    rows.append(("F2", p))

    # F3: size slope  (needs >1 model sizes; placeholder)
    rows.append(("F3", float("nan")))

    # F4: corr ED vs temp
    r,p = ss.pearsonr(df["ED_mean"], df["temp"])
    rows.append(("F4", p))

    # F5: AR(1) coeff
    phi = ar1(df["ED_mean"])
    p = ss.ttest_1samp([phi], 0.0).pvalue
    rows.append(("F5", p))

    # F6: corr ED vs length
    r,p = ss.pearsonr(df["ED_mean"], df["seq_len"])
    rows.append(("F6", p))

    # F7: domain uniformity (needs domain col)
    if "domain" in df.columns:
        k,p = ss.kruskal(*[g["ED_mean"] for _,g in df.groupby("domain")])
    else:
        p = float("nan")
    rows.append(("F7", p))

    # F8: rank-independence placeholder
    rows.append(("F8", float("nan")))

    out = pd.DataFrame(rows, columns=["TestID","p_value"])
    out.to_csv(args.out, index=False)
    print("→", args.out)

if __name__ == "__main__":
    main()
