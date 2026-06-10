"""Reusable credit-risk pipeline: features, model, SHAP drivers, explanations.

Shared by the notebook (analysis) and app.py (Streamlit demo) so both use one
source of truth for feature engineering and scoring.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

SEED = 42
DATA = Path(__file__).parent / "data"

NUMERIC = ["revenue_m", "ebitda_margin", "debt_ratio", "interest_coverage",
           "cash_ratio", "years_in_operation", "employee_count", "revenue_growth"]
ENGINEERED = ["debt_to_coverage", "margin_x_growth"]
# country dropped: one-hot country carried ~28% of XGBoost importance but zero
# univariate signal (an OHE-cardinality artifact). Dropping it leaves CV AUC
# unchanged (0.766) and removes the overfit. Sector keeps real signal.
CATEGORICAL = ["sector"]
TARGET = "defaulted"

# Tier cutoffs derived from the 70th / 90th percentile of out-of-fold PD on the
# training set (data-driven, not hand-picked). On training they map to default
# rates of ~9% (Low) / ~27% (Medium) / ~56% (High = riskiest decile).
MED_CUT = 0.16
HIGH_CUT = 0.34
BASE_RATE = 0.174  # training default rate — reference line on PD gauges

FEATURE_LABELS = {
    "revenue_m": "revenue", "ebitda_margin": "EBITDA margin", "debt_ratio": "debt ratio",
    "interest_coverage": "interest coverage", "cash_ratio": "cash ratio",
    "years_in_operation": "years in operation", "employee_count": "headcount",
    "revenue_growth": "revenue growth", "debt_to_coverage": "debt-to-coverage ratio",
    "margin_x_growth": "profitable-growth profile",
}

NEG = ["dependence on a few key accounts", "exposed to volatile commodity prices",
       "short operating history", "regulatory uncertainty", "high customer concentration",
       "margin compression", "declining revenue", "legacy debt burden",
       "weak liquidity position", "tight cash position"]
POS = ["recurring subscription revenue", "long-term government contracts",
       "strong market position", "resilient demand profile", "diversified customer base",
       "stable supplier relationships", "high-quality management team"]


# --------------------------------------------------------------------------- #
# Data + features
# --------------------------------------------------------------------------- #
def load_train() -> pd.DataFrame:
    return (pd.read_csv(DATA / "train_companies.csv")
            .merge(pd.read_csv(DATA / "train_narratives.csv"), on="company_id")
            .merge(pd.read_csv(DATA / "train_outcomes.csv"), on="company_id"))


def load_scoring() -> pd.DataFrame:
    return pd.read_csv(DATA / "scoring_companies.csv")


def build_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Model matrix. Pass `columns` to align an inference frame to the train layout."""
    x = df.copy()
    # coerce numerics (a single-row .T frame arrives as object dtype)
    x[NUMERIC] = x[NUMERIC].apply(pd.to_numeric, errors="coerce")
    x["debt_to_coverage"] = x["debt_ratio"] / (x["interest_coverage"] + 0.1)
    x["margin_x_growth"] = x["ebitda_margin"] * x["revenue_growth"]
    X = pd.concat([x[NUMERIC + ENGINEERED].astype(float),
                   pd.get_dummies(x[CATEGORICAL])], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _base_xgb() -> XGBClassifier:
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=SEED, n_jobs=4)


def make_model() -> CalibratedClassifierCV:
    """Platt-calibrated XGBoost — calibrated PD, not just a ranking score."""
    return CalibratedClassifierCV(_base_xgb(), method="sigmoid", cv=3)


def train():
    """Fit the calibrated model + a raw explainer. Returns (model, explainer, columns)."""
    train_df = load_train()
    X = build_features(train_df)
    y = train_df[TARGET].values
    model = make_model().fit(X, y)
    explainer = _base_xgb().fit(X, y)
    return model, explainer, list(X.columns)


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
def tier_of(p: float) -> str:
    return "Low" if p < MED_CUT else ("Medium" if p < HIGH_CUT else "High")


def _label(col: str) -> str:
    if col.startswith("sector_"):
        return f"{col[7:]} sector"
    if col.startswith("country_"):
        return f"operations in {col[8:]}"
    return FEATURE_LABELS.get(col, col)


def risk_drivers(explainer, X_row: pd.DataFrame, raw_row: pd.Series, k: int = 4):
    """Top-k signed SHAP drivers for one company. Returns list of (text, signed_value)."""
    contrib = explainer.get_booster().predict(
        xgb.DMatrix(X_row), pred_contribs=True)[0, :-1]
    s = pd.Series(contrib, index=X_row.columns)
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    out = []
    for col in s.index[:k]:
        direction = "raises" if s[col] > 0 else "lowers"
        if col in NUMERIC + ENGINEERED and col in raw_row.index:
            text = f"{_label(col)} ({raw_row[col]:.2f}) {direction} risk"
        else:
            text = f"{_label(col)} {direction} risk"
        out.append((text, float(s[col])))
    return out


def narrative_cue(text: str) -> str:
    low = str(text).lower()
    neg = [p for p in NEG if p in low]
    pos = [p for p in POS if p in low]
    bits = []
    if neg:
        bits.append("flagged concerns: " + ", ".join(neg[:2]))
    if pos:
        bits.append("positives: " + ", ".join(pos[:2]))
    return "; ".join(bits) if bits else "no strong qualitative signals"


def template_explanation(name, pd_prob, drivers, cue) -> str:
    driver_txt = "; ".join(d[0] for d in drivers[:3])
    return (f"{name} has an estimated PD of {pd_prob:.1%} ({tier_of(pd_prob)} risk). "
            f"Key drivers: {driver_txt}. Narrative — {cue}.")


def llm_explanation(name, sector, country, pd_prob, drivers, cue) -> str | None:
    """Groq (OpenAI-compatible) rewrite. Returns None if no key or on error."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        prompt = (
            "You are a credit risk analyst. In 2-3 concise sentences, explain this "
            "company's default risk. Be specific and reference the drivers. Do not "
            "invent numbers.\n\n"
            f"Company: {name} ({sector}, {country})\n"
            f"Estimated PD: {pd_prob:.1%} — {tier_of(pd_prob)} risk tier\n"
            f"Model risk drivers: {'; '.join(d[0] for d in drivers)}\n"
            f"Narrative cue: {cue}\n\nRisk assessment:")
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=160, temperature=0.3,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()
    except Exception:
        return None
