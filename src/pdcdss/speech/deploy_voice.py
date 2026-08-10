"""Train, evaluate and serialise the DEPLOYED voice model (eGeMAPS, IPVS).

This is the model the web app uses on live audio: an uploaded recording is turned into
eGeMAPS features by the same extractor used here (pdcdss.speech.audio_features), then an
XGBoost classifier returns a Parkinson's probability. The model is evaluated subject-level
(grouped on the individual) so the deployed pipeline gets the same anti-leakage discipline
as the UCI #470 research benchmark.

Six speakers are excluded from training entirely and scored afterwards as an external
check. Their recordings ship with the project as the demonstration set, and a
demonstration proves nothing if its clips were also training data: a correct answer could
be memory rather than prediction. Holding the demonstration speakers out makes every demo
run a genuine unseen-subject prediction. They are identified by an anonymous hash of the
subject key, so no speaker name appears here.

Run:  python -m pdcdss.speech.deploy_voice
Outputs:
  models/deployed_voice.joblib          fitted pipeline + eGeMAPS feature list
  results/tables/ipvs_deploy_eval.csv    subject-level evaluation metrics
  results/tables/ipvs_demo_holdout.csv   per-subject scores for the held-out demo speakers
"""
from __future__ import annotations

import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pdcdss.config import (
    MODELS_DIR,
    N_SPLITS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_TABLES,
)

IPVS_CSV = PROCESSED_DIR / "ipvs_egemaps.csv"
MODEL_OUT = MODELS_DIR / "deployed_voice.joblib"
META_COLS = {"label", "group", "subject", "file"}

#: SHA-256 prefixes of the subject keys whose recordings ship as the demonstration set
#: (three Parkinson's, three control). Never trained on; scored after the final fit.
DEMO_HOLDOUT_TAGS = frozenset({
    "426692cf", "d6841d4b", "cb78ecf4",      # Parkinson's
    "0b0d375c", "0050bf99", "0fe88d65",      # healthy control
})


def _tag(subject: str) -> str:
    return hashlib.sha256(subject.encode()).hexdigest()[:8]


def _xgb() -> XGBClassifier:
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.8, eval_metric="logloss",
                         random_state=RANDOM_SEED, n_jobs=-1)


def _pipe() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", _xgb())])


def main() -> None:
    if not IPVS_CSV.exists():
        raise SystemExit(f"{IPVS_CSV} not found. Run build_ipvs_features first.")
    df = pd.read_csv(IPVS_CSV)
    feats = [c for c in df.columns if c not in META_COLS]

    demo = df["subject"].map(_tag).isin(DEMO_HOLDOUT_TAGS)
    held = df[demo]
    df = df[~demo]
    X = df[feats].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    groups = df["subject"].to_numpy()

    print(f"IPVS eGeMAPS: {len(df)} recordings, {df['subject'].nunique()} subjects, "
          f"{len(feats)} features, class balance {np.bincount(y).tolist()}")
    print(f"demo holdout: {len(held)} recordings from {held['subject'].nunique()} subjects, "
          f"excluded from every fit below")

    # subject-level evaluation (no individual split across folds)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for tr, te in cv.split(X, y, groups):
        pipe = _pipe().fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append({
            "accuracy": accuracy_score(y[te], pred),
            "balanced_acc": balanced_accuracy_score(y[te], pred),
            "sensitivity": recall_score(y[te], pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "f1": f1_score(y[te], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[te], proba),
            "mcc": matthews_corrcoef(y[te], pred),
        })
    res = pd.DataFrame(rows)
    summary = res.mean().to_frame("mean").join(res.std().to_frame("std"))
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_TABLES / "ipvs_deploy_eval.csv")
    print("\nSubject-level evaluation (mean over folds):")
    for m in ("balanced_acc", "sensitivity", "specificity", "roc_auc", "mcc"):
        print(f"  {m:13s} {res[m].mean():.3f} ± {res[m].std():.3f}")

    # final model on every non-demo subject, serialised for the API
    final = _pipe().fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "features": feats}, MODEL_OUT)
    print(f"\nSaved deployed model -> {MODEL_OUT}")
    print(f"Saved evaluation -> {RESULTS_TABLES / 'ipvs_deploy_eval.csv'}")

    # score the held-out demonstration speakers: genuinely unseen subjects
    proba_h = final.predict_proba(held[feats].to_numpy(dtype=float))[:, 1]
    holdout = (held.assign(tag=held["subject"].map(_tag), probability=proba_h)
                   .groupby("tag")
                   .agg(label=("label", "first"), recordings=("file", "count"),
                        mean_probability=("probability", "mean"),
                        min_probability=("probability", "min"),
                        max_probability=("probability", "max"))
                   .round(3)
                   .sort_values("label", ascending=False))
    holdout.to_csv(RESULTS_TABLES / "ipvs_demo_holdout.csv")
    print("\nHeld-out demo speakers (never trained on):")
    print(holdout.to_string())
    correct = int(((holdout["mean_probability"] >= 0.5).astype(int)
                   == holdout["label"]).sum())
    print(f"\n{correct}/{len(holdout)} held-out subjects on the correct side of 0.5")


if __name__ == "__main__":
    main()
