"""Fasanara — Credit Risk Analyst. Streamlit console.

Two views:
  • Portfolio — scored companies as a credit-desk console, with a per-company dossier.
  • Live scorer — enter a borrower's financials, get a calibrated PD in real time.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import credit_pipeline as cp

load_dotenv()
st.set_page_config(page_title="Fasanara — Credit Risk Desk",
                   page_icon="▮", layout="wide")

BASE_RATE = cp.BASE_RATE   # training default rate — reference line on every gauge
HIGH_CUT = cp.HIGH_CUT     # data-driven tier cutoffs (OOF p70/p90), shown on the gauge
MED_CUT = cp.MED_CUT
TIER_KEY = {"Low": "low", "Medium": "med", "High": "risk"}

# --------------------------------------------------------------------------- #
# Aesthetic: institutional credit desk. Warm paper, ink serif, mono data layer,
# hairline rules instead of cards, one oxblood red reserved for risk.
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  :root{
    --paper:oklch(0.971 0.008 75); --panel:oklch(0.986 0.006 75);
    --ink:oklch(0.27 0.012 60);    --muted:oklch(0.53 0.012 65);
    --faint:oklch(0.70 0.010 68);  --line:oklch(0.89 0.008 70);
    --risk:oklch(0.47 0.135 25);   --low:oklch(0.52 0.065 155);
    --med:oklch(0.66 0.110 75);
    --serif:Georgia,'Times New Roman',serif;
    --mono:'SF Mono',SFMono-Regular,ui-monospace,'Courier New',monospace;
  }
  .stApp{ background:var(--paper); color:var(--ink); font-family:var(--serif); }
  #MainMenu,footer,header{ visibility:hidden; }
  .block-container{ padding:1.4rem 2.2rem 3rem; max-width:1240px; }
  h1,h2,h3,h4{ font-family:var(--serif); color:var(--ink); letter-spacing:-.01em; }

  /* masthead */
  .desk{ display:flex; align-items:baseline; justify-content:space-between;
         border-bottom:2px solid var(--ink); padding-bottom:10px; }
  .desk .mark{ font-size:30px; font-weight:700; letter-spacing:-.015em; }
  .desk .mark b{ color:var(--risk); font-weight:700; }
  .desk .meta{ font-family:var(--mono); font-size:11px; letter-spacing:.18em;
               text-transform:uppercase; color:var(--muted); }

  /* stat ribbon — hairline separated, no boxes */
  .ribbon{ display:flex; gap:0; margin:14px 0 6px; }
  .ribbon .s{ flex:1; padding:2px 22px; border-left:1px solid var(--line); }
  .ribbon .s:first-child{ border-left:none; padding-left:0; }
  .ribbon .v{ font-size:27px; font-weight:700; font-variant-numeric:tabular-nums; }
  .ribbon .k{ font-family:var(--mono); font-size:10.5px; letter-spacing:.13em;
              text-transform:uppercase; color:var(--muted); margin-top:1px; }

  /* section label */
  .lbl{ font-family:var(--mono); font-size:10.5px; letter-spacing:.18em;
        text-transform:uppercase; color:var(--muted); margin:6px 0 8px;
        border-bottom:1px solid var(--line); padding-bottom:5px; }

  /* console list */
  .head{ display:flex; font-family:var(--mono); font-size:10px; letter-spacing:.12em;
         text-transform:uppercase; color:var(--faint); padding:0 0 6px; }
  .crow{ display:flex; align-items:center; padding:9px 0; border-top:1px solid var(--line);
         font-size:14px; }
  .crow:hover{ background:var(--panel); }
  .c-co{ flex:2.4; } .c-co .id{ font-family:var(--mono); font-size:11px; color:var(--faint); }
  .c-co .nm{ font-weight:600; }
  .c-sec{ flex:1.7; font-family:var(--mono); font-size:11.5px; color:var(--muted); }
  .c-pd{ flex:2.2; } .c-rt{ flex:.9; text-align:right; }

  /* pd meter */
  .meter{ position:relative; height:6px; background:oklch(0.92 0.006 72);
          border-radius:0; overflow:visible; }
  .meter .fill{ position:absolute; left:0; top:0; bottom:0; }
  .meter .tk{ position:absolute; top:-2px; bottom:-2px; width:1px; background:var(--ink); opacity:.35; }
  .meter .tk.hi{ background:var(--risk); opacity:.6; }
  .mval{ font-family:var(--mono); font-size:12px; font-variant-numeric:tabular-nums;
         margin-top:3px; display:inline-block; }

  /* tier chip */
  .chip{ font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase;
         padding:2px 9px; border:1px solid currentColor; border-radius:1px; }
  .chip.low{ color:var(--low); } .chip.med{ color:var(--med); } .chip.risk{ color:var(--risk); }

  /* dossier */
  .dossier{ background:var(--panel); border:1px solid var(--line); padding:20px 22px; }
  .dossier .co{ font-size:22px; font-weight:700; line-height:1.1; }
  .dossier .sub{ font-family:var(--mono); font-size:11.5px; color:var(--muted);
                 letter-spacing:.05em; margin-top:2px; }

  /* big gauge */
  .gauge{ margin:16px 0 4px; }
  .gauge .top{ display:flex; align-items:baseline; justify-content:space-between; }
  .gauge .pd{ font-size:38px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1; }
  .gauge .track{ position:relative; height:10px; background:oklch(0.92 0.006 72); margin:10px 0 4px; }
  .gauge .fill{ position:absolute; left:0; top:0; bottom:0; }
  .gauge .tk{ position:absolute; top:-4px; bottom:-4px; width:1px; background:var(--ink); opacity:.4; }
  .gauge .tk.hi{ background:var(--risk); opacity:.7; }
  .gauge .ax{ display:flex; justify-content:space-between; font-family:var(--mono);
              font-size:9.5px; color:var(--faint); }

  /* diverging driver bars */
  .drv{ display:flex; align-items:center; gap:10px; padding:5px 0; }
  .drv .t{ flex:1.5; font-family:var(--mono); font-size:11.5px; color:var(--ink); }
  .drv .bar{ flex:1; position:relative; height:14px; }
  .drv .bar:before{ content:''; position:absolute; left:50%; top:0; bottom:0; width:1px;
                    background:var(--line); }
  .drv .f{ position:absolute; top:2px; bottom:2px; }
  .drv .f.up{ left:50%; background:var(--risk); }
  .drv .f.dn{ right:50%; background:var(--low); }

  /* analyst note */
  .note{ margin-top:14px; }
  .note .tab{ font-family:var(--mono); font-size:9.5px; letter-spacing:.16em;
              text-transform:uppercase; color:var(--muted); }
  .note .body{ border:1px solid var(--line); background:var(--paper); padding:14px 16px;
               font-size:15px; line-height:1.55; margin-top:5px; }

  /* tabs as desk nav */
  .stTabs [data-baseweb="tab-list"]{ gap:2px; border-bottom:1px solid var(--line); }
  .stTabs [data-baseweb="tab"]{ font-family:var(--mono); font-size:12px; letter-spacing:.1em;
        text-transform:uppercase; color:var(--muted); }
  .stTabs [aria-selected="true"]{ color:var(--ink); }

  /* widget legibility — force ink on warm paper */
  [data-testid="stWidgetLabel"] p, .stSlider label, label p{
        color:var(--ink) !important; font-family:var(--mono); font-size:11px !important;
        letter-spacing:.04em; }
  [data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
  .stSlider [data-testid="stThumbValue"]{ color:var(--muted) !important; font-family:var(--mono); }
  [data-baseweb="input"] input, [data-baseweb="select"] *{ color:var(--ink) !important; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Calibrating model…")
def get_model():
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    train_df = cp.load_train()
    X = cp.build_features(train_df)
    y = train_df[cp.TARGET].values
    cv = StratifiedKFold(5, shuffle=True, random_state=cp.SEED)
    oof = cross_val_predict(cp.make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, oof)
    auc = roc_auc_score(y, oof)
    metrics = dict(AUC=auc, Gini=2*auc-1, KS=float(np.max(tpr-fpr)),
                   Brier=brier_score_loss(y, oof))
    model, explainer, cols = cp.train()
    return model, explainer, cols, metrics, train_df


@st.cache_data
def get_scored(_model, cols):
    s = cp.load_scoring()
    s = s.copy()
    s["pd"] = _model.predict_proba(cp.build_features(s, columns=cols))[:, 1]
    s["tier"] = [cp.tier_of(p) for p in s["pd"]]
    return s


model, explainer, COLS, METRICS, TRAIN = get_model()
scored = get_scored(model, COLS)


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def meter(p, tier):
    return (f'<div class="meter"><div class="fill" style="width:{min(p,1)*100:.1f}%;'
            f'background:var(--{TIER_KEY[tier]})"></div>'
            f'<span class="tk" style="left:{BASE_RATE*100:.1f}%"></span>'
            f'<span class="tk hi" style="left:{HIGH_CUT*100:.0f}%"></span></div>')


def chip(tier):
    return f'<span class="chip {TIER_KEY[tier]}">{tier}</span>'


def console(df):
    rows = ['<div class="head"><div class="c-co">Company</div>'
            '<div class="c-sec">Sector / Country</div>'
            '<div class="c-pd">Probability of default</div>'
            '<div class="c-rt">Rating</div></div>']
    for _, r in df.iterrows():
        rows.append(
            f'<div class="crow"><div class="c-co"><span class="id">#{r.company_id}</span> '
            f'<span class="nm">{r.company_name}</span></div>'
            f'<div class="c-sec">{r.sector} · {r.country}</div>'
            f'<div class="c-pd">{meter(r.pd, r.tier)}<span class="mval">{r.pd:.1%}</span></div>'
            f'<div class="c-rt">{chip(r.tier)}</div></div>')
    return "".join(rows)


def gauge(p, tier):
    return (f'<div class="gauge"><div class="top">'
            f'<span class="pd" style="color:var(--{TIER_KEY[tier]})">{p:.1%}</span>{chip(tier)}</div>'
            f'<div class="track"><div class="fill" style="width:{min(p,1)*100:.1f}%;'
            f'background:var(--{TIER_KEY[tier]})"></div>'
            f'<span class="tk" style="left:{BASE_RATE*100:.1f}%"></span>'
            f'<span class="tk hi" style="left:{HIGH_CUT*100:.0f}%"></span></div>'
            f'<div class="ax"><span>0%</span><span>base {BASE_RATE:.0%}</span>'
            f'<span>high {HIGH_CUT:.0%}</span><span>100%</span></div></div>')


def driver_bars(drivers):
    mx = max((abs(v) for _, v in drivers), default=1e-9)
    out = ['<div class="lbl" style="margin-top:14px">Risk drivers · signed TreeSHAP</div>']
    for text, val in drivers:
        w = abs(val) / mx * 48
        cls = "up" if val > 0 else "dn"
        out.append(f'<div class="drv"><span class="t">{text}</span>'
                   f'<div class="bar"><div class="f {cls}" style="width:{w:.0f}%"></div></div></div>')
    return "".join(out)


def note(text):
    return (f'<div class="note"><div class="tab">Analyst note</div>'
            f'<div class="body">{text}</div></div>')


def explain_for(name, sector, country, p, drivers, cue):
    return (cp.llm_explanation(name, sector, country, p, drivers, cue)
            or cp.template_explanation(name, p, drivers, cue))


# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #
key_on = bool(__import__("os").getenv("GROQ_API_KEY"))
st.markdown(
    '<div class="desk"><div class="mark">Fasanara <b>Credit Desk</b></div>'
    f'<div class="meta">AI risk analyst · {"groq llm" if key_on else "template"} explanations</div>'
    '</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="ribbon">'
    f'<div class="s"><div class="v">{len(scored)}</div><div class="k">borrowers scored</div></div>'
    f'<div class="s"><div class="v">{scored["pd"].mean():.1%}</div><div class="k">scored-set mean PD</div></div>'
    f'<div class="s"><div class="v" style="color:var(--risk)">{(scored.tier=="High").sum()}</div>'
    f'<div class="k">high risk</div></div>'
    f'<div class="s"><div class="v">{METRICS["AUC"]:.3f}</div><div class="k">cv auc · gini {METRICS["Gini"]:.2f}</div></div>'
    f'</div>', unsafe_allow_html=True)

tab_pf, tab_live = st.tabs(["Portfolio", "Live scorer"])

# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
with tab_pf:
    left, right = st.columns([3, 2], gap="large")
    with left:
        sel_tiers = st.pills("tier", ["Low", "Medium", "High"],
                             selection_mode="multi", default=["Low", "Medium", "High"],
                             label_visibility="collapsed")
        view = scored[scored.tier.isin(sel_tiers or ["Low", "Medium", "High"])] \
            .sort_values("pd", ascending=False)
        st.markdown(console(view), unsafe_allow_html=True)

    with right:
        st.markdown('<div class="lbl">Dossier</div>', unsafe_allow_html=True)
        names = view.company_name.tolist() or scored.company_name.tolist()
        pick = st.selectbox("company", names, label_visibility="collapsed")
        row = scored[scored.company_name == pick].iloc[0]
        X_row = cp.build_features(row.to_frame().T, columns=COLS)
        drivers = cp.risk_drivers(explainer, X_row, row, k=4)
        cue = cp.narrative_cue(row["business_description"])
        st.markdown(
            f'<div class="dossier"><div class="co">{row.company_name}</div>'
            f'<div class="sub">#{row.company_id} · {row.sector} · {row.country}</div>'
            f'{gauge(row.pd, row.tier)}{driver_bars(drivers)}'
            f'{note(explain_for(row.company_name, row.sector, row.country, row.pd, drivers, cue))}'
            f'</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Live scorer
# --------------------------------------------------------------------------- #
with tab_live:
    st.markdown('<div class="lbl">Hypothetical borrower</div>', unsafe_allow_html=True)
    med = TRAIN[cp.NUMERIC].median()
    form, panel = st.columns([3, 2], gap="large")

    with form:
        a, b, c = st.columns(3)
        sector = a.selectbox("Sector", sorted(TRAIN.sector.unique()))
        country = b.selectbox("Country", sorted(TRAIN.country.unique()))
        years = c.number_input("Years in operation", 1, 60, int(med.years_in_operation))
        d, e, f, g = st.columns(4)
        revenue = d.number_input("Revenue (m)", 0.0, 500.0, float(round(med.revenue_m, 1)))
        margin = e.slider("EBITDA margin", -0.2, 0.6, float(round(med.ebitda_margin, 2)), 0.01)
        debt = f.slider("Debt ratio", 0.0, 1.5, float(round(med.debt_ratio, 2)), 0.01)
        cover = g.slider("Interest coverage", 0.0, 12.0, float(round(med.interest_coverage, 1)), 0.1)
        h, i, j = st.columns(3)
        cash = h.slider("Cash ratio", 0.0, 0.6, float(round(med.cash_ratio, 2)), 0.01)
        growth = i.slider("Revenue growth", -0.4, 0.6, float(round(med.revenue_growth, 2)), 0.01)
        emp = j.number_input("Employee count", 1, 5000, int(med.employee_count))

    profile = pd.DataFrame([{
        "revenue_m": revenue, "ebitda_margin": margin, "debt_ratio": debt,
        "interest_coverage": cover, "cash_ratio": cash, "years_in_operation": years,
        "employee_count": emp, "revenue_growth": growth, "sector": sector, "country": country}])
    X_row = cp.build_features(profile, columns=COLS)
    p = float(model.predict_proba(X_row)[:, 1][0])
    tier = cp.tier_of(p)
    drivers = cp.risk_drivers(explainer, X_row, profile.iloc[0], k=4)

    with panel:
        st.markdown('<div class="lbl">Assessment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dossier">{gauge(p, tier)}{driver_bars(drivers)}</div>',
                    unsafe_allow_html=True)
        if st.button("Generate analyst note"):
            expl = explain_for("This borrower", sector, country, p, drivers,
                               "no narrative (live profile)")
            st.markdown(note(expl), unsafe_allow_html=True)
