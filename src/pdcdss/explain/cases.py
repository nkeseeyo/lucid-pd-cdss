"""RQ3 case generator: real, leakage-free cases for the explanation experiment.

Each case is a {prediction, risk band, top contributing feature families} triple drawn
from the actual voice detector on UCI #470:

  * the probability (and therefore the risk band) is an out-of-fold, subject-level
    prediction, so the case is leakage-free;
  * the top contributing features come from the model's SHAP attributions for that
    subject, mapped from raw acoustic identifiers to plain-language family descriptions
    (a patient cannot read 'tqwt_entropy_log_dec_27', so the explanation layer must not
    be handed it either).

A fixed spread across the low/moderate/high bands and both true classes is selected so
the explanation comparison covers easy, borderline and misclassified cases.

Run:  python -m pdcdss.explain.cases
Output: results/tables/rq3_cases.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pdcdss.config import N_SPLITS, RANDOM_SEED, RESULTS_TABLES
from pdcdss.speech.eda import feature_family
from pdcdss.speech.experiments import load_uci470
from pdcdss.speech.shap_analysis import _shap_values, _xgb

# risk bands match the rule-based recommender thresholds (0.4 / 0.7)
LOW, HIGH = 0.4, 0.7

# raw acoustic family -> plain-language description for a lay reader
PLAIN: dict[str, str] = {
    "TQWT (tunable-Q wavelet)": "fine-grained irregularity in the voice signal",
    "Wavelet transform (WT)": "irregularity in the voice signal",
    "MFCC / cepstral": "the spectral shape and dynamics of speech (articulation)",
    "Vocal fold (GQ/GNE/VFER/IMF)": "vocal-fold vibration quality",
    "Formant / intensity": "vowel resonance and loudness",
    "Baseline (jitter/shimmer/...)": "pitch and amplitude stability (jitter and shimmer)",
    "Demographic": "a demographic factor",
    "Other": "other acoustic measures",
}


def _band(p: float) -> str:
    return "low" if p < LOW else ("moderate" if p < HIGH else "high")


def _prediction_text(p: float) -> str:
    return f"Estimated probability of Parkinson's disease: {round(100 * p)}%"


def build_cases(n_per_band: int = 3) -> pd.DataFrame:
    X, y, groups, feats = load_uci470()
    feats = list(feats)
    fam = {f: feature_family(f) for f in feats}
    groups = np.asarray(groups)
    y = np.asarray(y)

    # out-of-fold (leakage-free) probabilities AND held-out SHAP from the SAME fold
    # model, so each subject's estimate and its "why" are coherent and leakage-free
    oof = np.full(len(y), np.nan)
    sv_oof = np.full((len(y), len(feats)), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    for tr, te in cv.split(X, y, groups):
        m = _xgb().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        sv_oof[te] = _shap_values(m, X[te])  # signed, toward PD (class 1)

    sub = pd.DataFrame({"group": groups, "y": y, "prob": oof})
    per_sub = sub.groupby("group").agg(y=("y", "first"), prob=("prob", "mean"))
    per_sub["band"] = per_sub["prob"].map(_band)

    # mean signed held-out SHAP per subject per feature -> top contributing families
    sv_df = pd.DataFrame(sv_oof, columns=feats)
    sv_df["group"] = groups
    sub_shap = sv_df.groupby("group").mean()

    def top_families(g: int, k: int = 3) -> str:
        row = sub_shap.loc[g]
        order = row.abs().sort_values(ascending=False).index
        seen, items = set(), []
        for f in order:
            family = PLAIN[fam[f]]
            if family in seen:
                continue
            seen.add(family)
            direction = "raising the estimate" if row[f] > 0 else "lowering the estimate"
            items.append(f"{family} ({direction})")
            if len(items) == k:
                break
        return "; ".join(items)

    # deterministic spread: per band, pick subjects nearest the band centre, mixing
    # true PD and true HC where both exist
    centres = {"low": 0.15, "moderate": 0.55, "high": 0.9}
    chosen = []
    for band, centre in centres.items():
        pool = per_sub[per_sub["band"] == band].copy()
        if pool.empty:
            continue
        pool["dist"] = (pool["prob"] - centre).abs()
        pool = pool.sort_values(["dist", "prob"])
        picked, labels_seen = [], set()
        # first pass: prefer covering both true labels
        for g, r in pool.iterrows():
            if int(r["y"]) not in labels_seen:
                picked.append(g)
                labels_seen.add(int(r["y"]))
            if len(picked) >= n_per_band:
                break
        # fill remaining slots by closeness to centre
        for g, _ in pool.iterrows():
            if len(picked) >= n_per_band:
                break
            if g not in picked:
                picked.append(g)
        chosen.extend((band, g) for g in picked[:n_per_band])

    rows = []
    for i, (band, g) in enumerate(chosen, start=1):
        r = per_sub.loc[g]
        rows.append({
            "case_id": f"C{i:02d}",
            "subject": int(g),
            "true_label": "PD" if int(r["y"]) == 1 else "control",
            "pred_prob": round(float(r["prob"]), 3),
            "risk_band": band,
            "prediction": _prediction_text(float(r["prob"])),
            "top_features": top_families(g),
            "correct": (int(r["y"]) == 1) == (float(r["prob"]) >= LOW),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_cases()
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out = RESULTS_TABLES / "rq3_cases.csv"
    df.to_csv(out, index=False)
    print(f"Built {len(df)} RQ3 cases (bands: {dict(df['risk_band'].value_counts())})")
    for _, r in df.iterrows():
        print(f"  {r['case_id']} [{r['risk_band']:8s}] {r['true_label']:7s} "
              f"p={r['pred_prob']:.2f} | {r['top_features']}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
