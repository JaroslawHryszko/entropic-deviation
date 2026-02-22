#!/usr/bin/env python3
"""
calculate_metrics.py — Statistical falsification tests (F1-F8) for ED results.
"""
import argparse, os
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm


def main():
    ap = argparse.ArgumentParser(
        description="Run falsification tests F1-F8 on ED results"
    )
    ap.add_argument("csv", help="Path to ED results CSV")
    ap.add_argument("--out", default="results_FT.csv", help="Output CSV for test results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []

    # F1: mean ED != 0  (one-sample t-test)
    t, p = ss.ttest_1samp(df["ED_mean"], 0.0)
    rows.append(("F1", p))

    # F2: temperature effect (one-way ANOVA)
    groups = [g["ED_mean"].values for _, g in df.groupby("temp")]
    f, p = ss.f_oneway(*groups)
    rows.append(("F2", p))

    # F2: Post-hoc Tukey HSD
    if df["temp"].nunique() > 2:
        try:
            tuk = sm.stats.multicomp.pairwise_tukeyhsd(df["ED_mean"], df["temp"])
            tuk_df = pd.DataFrame(
                data=tuk.summary().data[1:], columns=tuk.summary().data[0]
            )
            out_dir = os.path.dirname(args.out) or "."
            base = os.path.basename(args.csv)
            tuk_df.to_csv(os.path.join(out_dir, f"f2_posthoc_{base}"), index=False)
        except Exception as e:
            print("F2 post-hoc Tukey HSD failed:", e)

    # F3: size slope (OLS, needs >1 model sizes)
    if "model_size" in df.columns and df["model_size"].dropna().nunique() > 1:
        clean = df[["ED_mean", "model_size"]].dropna()
        mdl = sm.OLS(clean["ED_mean"], sm.add_constant(clean["model_size"])).fit()
        p = mdl.pvalues["model_size"]
    else:
        p = float("nan")
    rows.append(("F3", p))

    # F4: corr ED vs temp
    r, p = ss.pearsonr(df["ED_mean"], df["temp"])
    rows.append(("F4", p))

    # F5: AR(1) across checkpoints
    phis = []
    for _, g in df.groupby("chkpt_id"):
        if len(g) > 1:
            phi = g["ED_mean"].autocorr(lag=1)
            if pd.notna(phi):
                phis.append(phi)
    if len(phis) > 1:
        p = ss.ttest_1samp(phis, 0.0).pvalue
    else:
        p = float("nan")
    rows.append(("F5", p))

    # F6: corr ED vs length
    if "seq_len" in df.columns and df["seq_len"].nunique() > 1:
        r, p = ss.pearsonr(df["ED_mean"], df["seq_len"])
    else:
        p = float("nan")
    rows.append(("F6", p))

    # F7: domain uniformity (Kruskal-Wallis)
    if "domain" in df.columns and df["domain"].nunique() > 1:
        k, p = ss.kruskal(*[g["ED_mean"].values for _, g in df.groupby("domain")])
    else:
        p = float("nan")
    rows.append(("F7", p))

    # F8: rank-independence (OLS slope)
    if "rank" in df.columns and df["rank"].nunique() > 1:
        f8_data = df[["ED_mean", "rank"]].dropna()
        mdl = sm.OLS(f8_data["ED_mean"], sm.add_constant(f8_data["rank"])).fit()
        p = mdl.pvalues["rank"]
    else:
        p = float("nan")
    rows.append(("F8", p))

    out = pd.DataFrame(rows, columns=["TestID", "p_value"])
    out.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
