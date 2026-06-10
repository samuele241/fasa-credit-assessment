# fasa-credit-assessment
Fasanara Credit Risk Prediction Challenge

## Set up
1. Fork this repository to your own GitHub account.
2. Clone your forked repository to your local machine.
3. Create a new branch for your work.
4. Install any necessary dependencies

### Dependencies
- Python 3.12 or higher
- Set up

```bash
pip install uv
uv sync
```
- The above commnand will install the required dependencies listed in `pyproject.toml` to a virtual environment.

## Solution

**Goal:** estimate Probability of Default (PD) for SME borrowers, score the unseen companies, and produce analyst-style explanations.

**Approach.** Structured financials drive the model; the narrative text was tested (phrase flags, TF-IDF, sentence embeddings) and added *no* cross-validated lift on this synthetic data — so it is used only to enrich the explanations. The model is a **Platt-calibrated XGBoost**: shallow trees + low learning rate to avoid overfitting 1k rows, and sigmoid calibration so the output is a genuine probability (mean PD ≈ true base rate), not just a ranking score.

**Performance (5-fold CV):** AUC **0.766** · Gini **0.531** · KS **0.428** · Brier **0.122** — in the usual band for production credit scorecards, with well-calibrated probabilities. Top drivers (debt ratio, EBITDA margin, interest coverage, revenue growth) are economically sensible.

**Explanations.** For each scored company, the model extracts the top risk drivers using TreeSHAP (signed feature contributions that show which financials pushed the PD up or down). These drivers, together with qualitative signals extracted from the business description, are passed as structured context to an LLM (`llama-3.3-70b-versatile` via Groq's OpenAI-compatible API). The LLM is prompted to act as a credit analyst and rewrite the drivers into 2–3 fluent sentences, grounded strictly in the provided data — it receives no free-form context and is instructed not to invent numbers. If no API key is present, a deterministic template produces the same structured output as a fallback.

Optional: set `GROQ_API_KEY` in `.env` to enable LLM explanations (otherwise the template is used).

### Files
- `notebook.ipynb` — end-to-end analysis: EDA, modelling, text experiment, scoring, explanations.
- `credit_pipeline.py` — reusable features / model / SHAP / explanation logic.
- `app.py` — Streamlit app: scored portfolio + live PD scorer.
- `outputs/predictions.csv` — `company_id, predicted_default_probability, risk_rating, explanation`.
