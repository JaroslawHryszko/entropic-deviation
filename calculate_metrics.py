#!/usr/bin/env python3
"""
calculate_metrics.py — Statistical falsification tests (F1-F8) for ED results.
"""
import argparse, os, logging
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm


def setup_logger(log_file=None):
    logger = logging.getLogger("ed_metrics")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def main():
    ap = argparse.ArgumentParser(
        description="Run falsification tests F1-F8 on ED results"
    )
    ap.add_argument("csv", help="Path to ED results CSV")
    ap.add_argument("--out", default="results_FT.csv", help="Output CSV for test results")
    ap.add_argument("--log", default=None, help="Log file path")
    args = ap.parse_args()

    logger = setup_logger(args.log)
    logger.info(f"Loading ED results from {args.csv}")

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        logger.error(f"Failed to read {args.csv}: {e}")
        return
    logger.info(f"Loaded {len(df)} rows, {df.columns.tolist()}")

    rows = []

    # F1: mean ED != 0  (one-sample t-test)
    try:
        t, p = ss.ttest_1samp(df["ED_mean"], 0.0)
        rows.append(("F1", p))
        logger.info(f"F1 (t-test ED≠0): t={t:.4f}, p={p:.2e}")
    except Exception as e:
        logger.error(f"F1 failed: {e}")
        rows.append(("F1", float("nan")))

    # F2: temperature effect (one-way ANOVA)
    try:
        groups = [g["ED_mean"].values for _, g in df.groupby("temp")]
        f, p = ss.f_oneway(*groups)
        rows.append(("F2", p))
        logger.info(f"F2 (ANOVA temp): F={f:.4f}, p={p:.2e}")
    except Exception as e:
        logger.error(f"F2 failed: {e}")
        rows.append(("F2", float("nan")))

    # F2: Post-hoc Tukey HSD
    if df["temp"].nunique() > 2:
        try:
            tuk = sm.stats.multicomp.pairwise_tukeyhsd(df["ED_mean"], df["temp"])
            tuk_df = pd.DataFrame(
                data=tuk.summary().data[1:], columns=tuk.summary().data[0]
            )
            out_dir = os.path.dirname(args.out) or "."
            base = os.path.basename(args.csv)
            posthoc_path = os.path.join(out_dir, f"f2_posthoc_{base}")
            tuk_df.to_csv(posthoc_path, index=False)
            logger.info(f"F2 post-hoc Tukey HSD saved to {posthoc_path}")
        except Exception as e:
            logger.error(f"F2 post-hoc Tukey HSD failed: {e}")

    # F3: size slope (OLS, needs >1 model sizes)
    try:
        if "model_size" in df.columns and df["model_size"].dropna().nunique() > 1:
            clean = df[["ED_mean", "model_size"]].dropna()
            mdl = sm.OLS(clean["ED_mean"], sm.add_constant(clean["model_size"])).fit()
            p = mdl.pvalues["model_size"]
            logger.info(f"F3 (OLS size): slope={mdl.params['model_size']:.2e}, p={p:.2e}")
        else:
            p = float("nan")
            logger.info("F3 (OLS size): skipped (single model size)")
    except Exception as e:
        logger.error(f"F3 failed: {e}")
        p = float("nan")
    rows.append(("F3", p))

    # F4: corr ED vs temp
    try:
        r, p = ss.pearsonr(df["ED_mean"], df["temp"])
        rows.append(("F4", p))
        logger.info(f"F4 (Pearson ED~temp): r={r:.4f}, p={p:.2e}")
    except Exception as e:
        logger.error(f"F4 failed: {e}")
        rows.append(("F4", float("nan")))

    # F5: AR(1) across checkpoints
    try:
        phis = []
        group_col = "chkpt_id" if "chkpt_id" in df.columns else None
        if group_col:
            for _, g in df.groupby(group_col):
                if len(g) > 1:
                    phi = g["ED_mean"].autocorr(lag=1)
                    if pd.notna(phi):
                        phis.append(phi)
        if len(phis) > 1:
            p = ss.ttest_1samp(phis, 0.0).pvalue
            logger.info(f"F5 (AR(1)): mean_phi={sum(phis)/len(phis):.4f}, p={p:.2e}")
        else:
            p = float("nan")
            logger.info("F5 (AR(1)): skipped (insufficient groups)")
    except Exception as e:
        logger.error(f"F5 failed: {e}")
        p = float("nan")
    rows.append(("F5", p))

    # F6: corr ED vs length
    try:
        if "seq_len" in df.columns and df["seq_len"].nunique() > 1:
            r, p = ss.pearsonr(df["ED_mean"], df["seq_len"])
            logger.info(f"F6 (Pearson ED~len): r={r:.4f}, p={p:.2e}")
        else:
            p = float("nan")
            logger.info("F6 (Pearson ED~len): skipped (uniform seq_len)")
    except Exception as e:
        logger.error(f"F6 failed: {e}")
        p = float("nan")
    rows.append(("F6", p))

    # F7: domain uniformity (Kruskal-Wallis)
    try:
        if "domain" in df.columns and df["domain"].nunique() > 1:
            k, p = ss.kruskal(*[g["ED_mean"].values for _, g in df.groupby("domain")])
            logger.info(f"F7 (Kruskal-Wallis domain): H={k:.4f}, p={p:.2e}")
        else:
            p = float("nan")
            logger.info("F7 (Kruskal-Wallis domain): skipped (single domain)")
    except Exception as e:
        logger.error(f"F7 failed: {e}")
        p = float("nan")
    rows.append(("F7", p))

    # F8: rank-independence (OLS slope)
    try:
        if "rank" in df.columns and df["rank"].nunique() > 1:
            f8_data = df[["ED_mean", "rank"]].dropna()
            mdl = sm.OLS(f8_data["ED_mean"], sm.add_constant(f8_data["rank"])).fit()
            p = mdl.pvalues["rank"]
            logger.info(f"F8 (OLS rank): slope={mdl.params['rank']:.2e}, p={p:.2e}")
        else:
            p = float("nan")
            logger.info("F8 (OLS rank): skipped (single rank)")
    except Exception as e:
        logger.error(f"F8 failed: {e}")
        p = float("nan")
    rows.append(("F8", p))

    out = pd.DataFrame(rows, columns=["TestID", "p_value"])
    out.to_csv(args.out, index=False)
    logger.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
