# GateEscape Dubai — Layover Decision-Confidence Analytics Dashboard

An individual Data Analytics assignment project. This dashboard validates the business
idea behind **GateEscape Dubai**, a smart layover decision-confidence and micro-experience
platform for Dubai transit passengers, using synthetic data and an end-to-end
validation-to-sales analytics pipeline.

---

## Project purpose

GateEscape Dubai helps transit passengers decide whether it is worth leaving the airport
during a layover, and routes them to a suitable experience tier (from staying airside to a
premium private tour). The core transformed variable is the **Layover Confidence Score** — a
transparent, rule-based 0–100 composite built from time buffer, visa eligibility, fatigue,
risk comfort, budget, interest, and luggage status.

This dashboard demonstrates the full analytics workflow required by the assignment brief:

- synthetic data validation of a new business idea (validation-to-sales pipeline)
- data cleaning, transformation, and feature engineering
- descriptive analytics
- diagnostic analytics
- correlation-based graphs and insights
- a clear business explanation under every major chart

No machine learning, classification, clustering, or forecasting is used. The Confidence
Score is a rule-based descriptive/diagnostic variable, not a predictive model.

---

## Files included

| File | Description |
|---|---|
| `app.py` | The Streamlit dashboard (loads `sample_data.csv` directly). |
| `data_preparation.py` | Generates the synthetic data, injects realistic raw issues, cleans/transforms it, engineers features, and exports `sample_data.csv`. |
| `sample_data.csv` | The cleaned, feature-engineered dataset the dashboard reads (~1,800 synthetic passenger sessions). |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

---

## How to run the dashboard

1. (Optional) Regenerate the dataset from scratch:
   ```bash
   python data_preparation.py
   ```
   This writes `sample_data.csv` into the project folder.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

The app expects `sample_data.csv` in the same folder as `app.py`. If the file is missing,
the dashboard displays a clear error message explaining how to generate it.

---

## What the dashboard shows

The dashboard is organised into seven tabs, each labelled with its analytics type so the
assignment brief is clearly addressed:

1. **Executive Overview** *(Descriptive)* — KPI cards (passengers, conversion, revenue,
   average spend, confidence score, satisfaction, referral intent, simulated late-return
   risk rate) plus an executive summary.
2. **Funnel Analytics** *(Descriptive)* — the validation-to-sales funnel (App Viewed →
   Eligibility Checked → Package Viewed → Booked → Completed) with retention and drop-off rates.
3. **Segment & Tier Performance** *(Descriptive)* — conversion by confidence band, revenue by
   tier, satisfaction by selected vs recommended tier, and conversion by party type and budget.
4. **Diagnostic Analytics** *(Diagnostic)* — why conversion and satisfaction differ, including
   confidence vs booking, packages viewed vs booking, fatigue vs recommended tier, safe-to-leave
   vs conversion, recommendation match vs satisfaction, and missingness vs conversion.
5. **Correlation Insights** *(Correlation-based diagnostic)* — a correlation heatmap and
   scatterplots, with an explicit reminder that correlation does not prove causation.
6. **Data Quality & Features** *(Data preparation)* — how raw issues were handled, why
   clean/derived columns were created, and how missingness flags preserve data quality.
7. **Business Recommendations** *(Strategy)* — evidence-linked recommendations on which
   segments, confidence bands, channels, and tiers to prioritise, plus limitations.

Sidebar filters (confidence band, recommended tier, budget tier, party type, marketing
channel, visa bucket, engagement level, and safe-to-leave) apply across all tabs. Analysis
uses the cleaned and engineered columns; raw columns are retained only as audit/background fields.

---

## Important notes on the data

- **The data is synthetic.** All records in `sample_data.csv` are computer-generated to
  simulate a pilot survey / prototype test for business validation. They do not describe
  real passengers, bookings, or revenue. Relationships were deliberately built into the
  generator so the dataset can demonstrate the analytical methods.

- **Late-return risk is a simulated synthetic indicator only.** The `late_return_risk_score`
  and `late_return_flag_clean` fields are analytical risk proxies created for this exercise.
  They do **not** represent any real airline, airport, immigration, or legal outcome and must
  not be interpreted as operational or legal claims.

- **The Layover Confidence Score is rule-based**, not a machine-learning model. It is used
  here purely as a transparent descriptive and diagnostic variable.
