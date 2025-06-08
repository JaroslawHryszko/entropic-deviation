#!/usr/bin/env python
# stats.py
import argparse, pandas as pd, scipy.stats as ss, statsmodels.api as sm

def ar1(phi_series):
    return phi_series.autocorr(lag=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results_FT.csv")
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
    # --- F2: Post-hoc Tukey HSD ---
    if df["temp"].nunique() > 2:
        try:
            tuk = sm.stats.multicomp.pairwise_tukeyhsd(df["ED_mean"], df["temp"])
            tuk_df = pd.DataFrame(data=tuk.summary().data[1:], columns=tuk.summary().data[0])
            tuk_df.to_csv(f"f2_posthoc_{args.csv}", index=False)
        except Exception as e:
            print("F2 post-hoc Tukey HSD failed:", e)


    # F3: size slope  (needs >1 model sizes; placeholder)
    # ----------------- F3  (OLS slope) -----------------
    ### --- PATCH F3 ---
    # F3: dependency on model size  (needs >1 distinct sizes)
    if df["model_size"].nunique() > 1:
        mdl = sm.OLS(df["ED_mean"], sm.add_constant(df["model_size"])).fit()
        p = mdl.pvalues["model_size"]
    else:
        p = float("nan")
    rows.append(("F3", p))
    ### --- END PATCH ---

    # F4: corr ED vs temp
    r,p = ss.pearsonr(df["ED_mean"], df["temp"])
    rows.append(("F4", p))

    # ----------------- F5  (AR(1) across checkpoints) -----------------
    ### --- PATCH F5 ---
    phis = []
    for _, g in df.groupby("chkpt_id"):
        if len(g) > 1:
            phis.append(g["ED_mean"].autocorr(lag=1))
    if len(phis) > 1:
        p = ss.ttest_1samp(phis, 0.0).pvalue
    else:
        p = float("nan")
    rows.append(("F5", p))
    ### --- END PATCH ---


    # F6: corr ED vs length
    if df["seq_len"].nunique() > 1:
        r, p = ss.pearsonr(df["ED_mean"], df["seq_len"])
    else:
        p = float("nan")
    rows.append(("F6", p))


    # F7: domain uniformity (needs domain col)
    if "domain" in df.columns:
        k,p = ss.kruskal(*[g["ED_mean"] for _,g in df.groupby("domain")])
    else:
        p = float("nan")
    rows.append(("F7", p))

    # ----------------- F8  (rank-independence) -----------------
    if "rank" in df.columns and df["rank"].nunique() > 1:
        f8_data = df[["ED_mean", "rank"]].dropna()
        mdl = sm.OLS(f8_data["ED_mean"], sm.add_constant(f8_data["rank"])).fit()
        p = mdl.pvalues["rank"]
    else:
        p = float("nan")
    rows.append(("F8", p))
    ### --- END PATCH ---


    out = pd.DataFrame(rows, columns=["TestID","p_value"])
    out.to_csv(args.out, index=False)
    print("→", args.out)

if __name__ == "__main__":
    main()
