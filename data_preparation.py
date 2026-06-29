"""
GateEscape Dubai — Data Preparation Pipeline
============================================
Individual Data Analytics Assignment

Single, self-contained script that runs the FULL data-preparation workflow:

    1. Generate synthetic raw passenger data (~1,800 sessions)
    2. Generate the small tier_reference lookup internally
    3. Inject intentional raw data-quality issues
    4. Clean and transform the data
    5. Apply feature engineering
    6. Export ONE dashboard-ready file: sample_data.csv

Design principles:
    - NO machine learning / classification. The Layover Confidence Score is a
      transparent rule-based composite only.
    - Raw-style columns are PRESERVED where useful; cleaning/derived work is
      written to NEW *_clean / *_flag columns so original values aren't destroyed.
    - Logically-missing values (e.g. satisfaction for a non-booking) are kept as
      NaN on purpose — they are not "errors" and must not be imputed.
    - late_return_* fields are SIMULATED synthetic risk indicators only; they do
      not represent any real operational, legal, or immigration outcome.

Output: sample_data.csv  (the only file this script writes)
"""

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
N = 1800


# =====================================================================
# PART 1 — TIER REFERENCE LOOKUP (generated internally, not exported)
# =====================================================================
def build_tier_reference():
    return pd.DataFrame({
        "tier_id": [1, 2, 3, 4, 5],
        "tier_name": [
            "Stay Airside", "Airport Lounge/Relaxation", "Indoor Micro-Experience",
            "City/Culture/Shopping", "Premium Fast-Track/Chauffeur",
        ],
        "typical_price_range_aed": ["0", "150-300", "250-450", "300-600", "800-2500"],
        "min_recommended_time_buffer_mins": [0, 60, 90, 180, 240],
        "base_risk_level": ["None", "Low", "Low", "Medium", "High"],
    })


# =====================================================================
# PART 2 — GENERATE CLEAN SYNTHETIC DATA (business logic)
# =====================================================================
def generate_clean_data():
    TIER_NAMES = build_tier_reference()["tier_name"].tolist()

    df = pd.DataFrame({
        "passenger_id": [f"PS{str(i).zfill(5)}" for i in range(1, N + 1)],
        "age_band": np.random.choice(
            ["18-24", "25-34", "35-44", "45-54", "55+"], N, p=[0.18, 0.32, 0.25, 0.15, 0.10]),
        "travel_party_type": np.random.choice(
            ["Solo", "Couple", "Family with kids", "Friends/Group"], N, p=[0.40, 0.25, 0.20, 0.15]),
        "visa_bucket": np.random.choice(
            ["Visa-Free/On-Arrival", "Transit-Visa-Required", "Not-Eligible"], N, p=[0.55, 0.33, 0.12]),
        "layover_duration_hrs": np.round(np.clip(np.random.gamma(3.0, 2.8, N), 2, 26), 1),
        "arrival_time_block": np.random.choice(
            ["Morning", "Afternoon", "Evening", "Night"], N, p=[0.28, 0.27, 0.27, 0.18]),
        "season_month": np.random.choice(
            ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"], N),
        "luggage_status": np.random.choice(
            ["Checked-Through", "Carry-On Only", "Needs Storage"], N, p=[0.55, 0.30, 0.15]),
        "fatigue_level": np.random.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.20, 0.30, 0.25, 0.15]),
        "budget_tier": np.random.choice(["Budget", "Mid-Range", "Premium"], N, p=[0.40, 0.40, 0.20]),
        "interest_category": np.random.choice(
            ["Shopping", "Culture & Heritage", "Family/Indoor", "Relaxation & Wellness", "Active/City"],
            N, p=[0.25, 0.20, 0.20, 0.20, 0.15]),
        "risk_comfort": np.random.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.20, 0.30, 0.25, 0.15]),
        "marketing_channel": np.random.choice(
            ["Airline Partner", "Airport Signage", "Social Media", "Travel Agent",
             "Word of Mouth/Referral", "Organic App Search"], N, p=[0.25, 0.20, 0.20, 0.10, 0.15, 0.10]),
    })

    # ---- Layover Confidence Score (transparent rule-based composite) ----
    time_buffer = (df["layover_duration_hrs"] * 60 - 90).clip(lower=0)
    time_buffer_score = (time_buffer / 360 * 100).clip(upper=100)
    visa_score = df["visa_bucket"].map(
        {"Visa-Free/On-Arrival": 100, "Transit-Visa-Required": 60, "Not-Eligible": 0})
    fatigue_score = (6 - df["fatigue_level"]) * 20
    risk_score = df["risk_comfort"] * 20
    budget_score = df["budget_tier"].map({"Budget": 40, "Mid-Range": 70, "Premium": 100})
    interest_score = df["interest_category"].map({
        "Shopping": 80, "Culture & Heritage": 85, "Family/Indoor": 75,
        "Relaxation & Wellness": 90, "Active/City": 78})
    luggage_score = df["luggage_status"].map(
        {"Checked-Through": 100, "Carry-On Only": 70, "Needs Storage": 40})

    raw_score = (time_buffer_score * 0.25 + visa_score * 0.20 + fatigue_score * 0.15
                 + risk_score * 0.15 + budget_score * 0.10 + interest_score * 0.10
                 + luggage_score * 0.05)
    score = (raw_score + np.random.normal(0, 3, N)).clip(0, 100)
    not_elig = df["visa_bucket"] == "Not-Eligible"
    score[not_elig] = score[not_elig].clip(upper=30)  # hard eligibility floor
    df["layover_confidence_score"] = score.round(1)

    df["confidence_band"] = pd.cut(
        df["layover_confidence_score"], [-0.1, 39, 69, 100], labels=["Low", "Medium", "High"]).astype(str)

    # ---- Rule-based recommended tier ----
    def rec_tier(r):
        if r["confidence_band"] == "Low":
            return "Stay Airside"
        if r["confidence_band"] == "Medium":
            return "Airport Lounge/Relaxation" if r["budget_tier"] == "Budget" else "Indoor Micro-Experience"
        return "Premium Fast-Track/Chauffeur" if r["budget_tier"] == "Premium" else "City/Culture/Shopping"

    df["recommended_tier"] = df.apply(rec_tier, axis=1)

    # High-fatigue nudge toward calmer tiers
    nudge = (df["fatigue_level"] >= 4) & df["recommended_tier"].isin(
        ["City/Culture/Shopping", "Premium Fast-Track/Chauffeur"]) & (np.random.rand(N) < 0.40)
    df.loc[nudge, "recommended_tier"] = "Indoor Micro-Experience"

    # ---- Funnel simulation ----
    df["app_viewed"] = True
    df["eligibility_checked"] = np.random.rand(N) < np.where(not_elig, 0.75, 0.93)
    df["packages_viewed"] = np.random.poisson(0.5 + df["layover_confidence_score"] / 100 * 2.5).clip(0, 6)

    band_base = df["confidence_band"].map({"Low": 0.08, "Medium": 0.32, "High": 0.62}).astype(float)
    book_prob = (band_base + df["packages_viewed"] * 0.07).clip(upper=0.95)
    book_prob = np.where(df["eligibility_checked"], book_prob, book_prob * 0.30)
    df["booked"] = np.random.rand(N) < book_prob

    def pick_pkg(r):
        if not r["booked"]:
            return np.nan
        if np.random.rand() < 0.80:
            return r["recommended_tier"]
        return np.random.choice([t for t in TIER_NAMES if t != r["recommended_tier"]])

    df["selected_package"] = df.apply(pick_pkg, axis=1)

    price = {"Stay Airside": (0, 0), "Airport Lounge/Relaxation": (150, 300),
             "Indoor Micro-Experience": (250, 450), "City/Culture/Shopping": (300, 600),
             "Premium Fast-Track/Chauffeur": (800, 2500)}
    bmult = {"Budget": 0.85, "Mid-Range": 1.0, "Premium": 1.2}

    def paid(r):
        if not r["booked"] or pd.isna(r["selected_package"]):
            return np.nan
        lo, hi = price[r["selected_package"]]
        return 0.0 if hi == 0 else round(np.random.uniform(lo, hi) * bmult[r["budget_tier"]], 0)

    df["paid_amount_aed"] = df.apply(paid, axis=1)

    tier_risk = {"Stay Airside": 0, "Airport Lounge/Relaxation": 10, "Indoor Micro-Experience": 15,
                 "City/Culture/Shopping": 35, "Premium Fast-Track/Chauffeur": 25}

    def late_risk(r):
        if not r["booked"]:
            return np.nan
        tb = (r["layover_duration_hrs"] * 60 - 90)
        buf_factor = np.clip(100 - tb / 6, 0, 100)
        return round(float(np.clip(0.40 * tier_risk.get(r["selected_package"], 10)
                                   + 0.60 * buf_factor + np.random.normal(0, 5), 0, 100)), 1)

    df["late_return_risk_score"] = df.apply(late_risk, axis=1)

    def on_time(r):
        if not r["booked"]:
            return np.nan
        return np.random.rand() > (r["late_return_risk_score"] / 100) * 0.5

    df["completed_on_time"] = df.apply(on_time, axis=1)

    def satis(r):
        if not r["booked"] or pd.isna(r["completed_on_time"]):
            return np.nan
        s = 3.2 + (0.8 if r["selected_package"] == r["recommended_tier"] else -0.5)
        if r["completed_on_time"] is False:
            s -= 0.7
        return int(np.clip(round(s + np.random.normal(0, 0.6)), 1, 5))

    df["satisfaction_rating"] = df.apply(satis, axis=1)

    def refer(r):
        if pd.isna(r["satisfaction_rating"]):
            return np.nan
        return int(np.clip(round(r["satisfaction_rating"] + np.random.normal(0, 0.5)), 1, 5))

    df["referral_intent"] = df.apply(refer, axis=1)
    return df


# =====================================================================
# PART 3 — INJECT INTENTIONAL RAW DATA-QUALITY ISSUES
# =====================================================================
def inject_raw_issues(df):
    df = df.copy()

    # 3.1 Duplicate passenger_id rows
    dup_idx = np.random.choice(df.index, 15, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 3.2 Missing values (survey non-response) — beyond logical nulls
    for col, frac in [("fatigue_level", 0.04), ("budget_tier", 0.03),
                      ("paid_amount_aed", 0.02), ("satisfaction_rating", 0.03),
                      ("referral_intent", 0.02)]:
        df[col] = df[col].astype(object)
        idx = df.sample(frac=frac, random_state=np.random.randint(0, 99999)).index
        df.loc[idx, col] = np.nan

    # 3.3 Inconsistent Yes/No-style boolean encodings
    yes_v, no_v = ["Yes", "Y", "1", "yes ", "TRUE"], ["No", "N", "0", "no ", "FALSE"]
    for col in ["app_viewed", "eligibility_checked", "booked", "completed_on_time"]:
        df[col] = df[col].astype(object)
        idx = df.sample(frac=0.10, random_state=np.random.randint(0, 99999)).index
        for i in idx:
            v = df.loc[i, col]
            if pd.isna(v):
                continue
            df.loc[i, col] = np.random.choice(yes_v) if v else np.random.choice(no_v)

    # 3.4 Inconsistent category casing/spacing
    def mess(col, frac):
        df[col] = df[col].astype(object)
        idx = df.sample(frac=frac, random_state=np.random.randint(0, 99999)).index
        for i in idx:
            v = df.loc[i, col]
            if pd.isna(v):
                continue
            style = np.random.choice(["upper", "lower", "space"])
            df.loc[i, col] = v.upper() if style == "upper" else v.lower() if style == "lower" else f"  {v} "

    for col in ["age_band", "travel_party_type", "visa_bucket", "luggage_status",
                "interest_category", "marketing_channel", "recommended_tier", "selected_package"]:
        mess(col, 0.06)

    # 3.5 Spelling variants for marketing channel + tier names
    mkt_map = {
        "Social Media": ["Social media", "social-media", "SOCIAL MEDIA "],
        "Word of Mouth/Referral": ["Word of Mouth", "word of mouth / referral", "Referral"],
        "Travel Agent": ["travel agent", "Travel Agency"]}
    tier_map = {
        "Premium Fast-Track/Chauffeur": ["Premium Fast Track", "Fast Track Premium"],
        "City/Culture/Shopping": ["City / Culture / Shopping", "Culture & Shopping"]}

    def apply_typos(col, m, frac):
        idx = df.sample(frac=frac, random_state=np.random.randint(0, 99999)).index
        for i in idx:
            v = df.loc[i, col]
            if v in m:
                df.loc[i, col] = np.random.choice(m[v])

    apply_typos("marketing_channel", mkt_map, 0.08)
    apply_typos("recommended_tier", tier_map, 0.05)
    apply_typos("selected_package", tier_map, 0.05)

    # 3.6 Impossible layover durations
    df["layover_duration_hrs"] = df["layover_duration_hrs"].astype(object)
    for n, i in enumerate(df.sample(10, random_state=7).index):
        df.loc[i, "layover_duration_hrs"] = -3.0 if n % 2 == 0 else 48.0

    # 3.7 Negative paid amounts on booked rows
    booked_mask = df["booked"].isin([True, "Yes", "Y", "1", "yes ", "TRUE"]) & df["paid_amount_aed"].notna()
    for i in df[booked_mask].sample(min(8, booked_mask.sum()), random_state=11).index:
        v = df.loc[i, "paid_amount_aed"]
        if pd.notna(v) and isinstance(v, (int, float)):
            df.loc[i, "paid_amount_aed"] = -abs(float(v))

    # 3.8 Numeric values stored as text
    for i in df.sample(12, random_state=13).index:
        v = df.loc[i, "layover_duration_hrs"]
        if isinstance(v, (int, float)):
            df.loc[i, "layover_duration_hrs"] = f"{v} hrs"
    for i in df[df["paid_amount_aed"].notna()].sample(10, random_state=17).index:
        v = df.loc[i, "paid_amount_aed"]
        if isinstance(v, (int, float)):
            df.loc[i, "paid_amount_aed"] = f"AED {v}"

    return df


# =====================================================================
# PART 4 — CLEANING & TRANSFORMATION HELPERS
# =====================================================================
def standardize_bool(x):
    """Map varied Yes/No encodings to True/False; preserve NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return np.nan


def clean_text(series):
    """Strip whitespace and standardize casing for category columns."""
    return series.astype(object).where(series.isna(), series.astype(str).str.strip())


def coerce_numeric(series):
    """Strip non-numeric text (e.g. 'AED 542.0', '7.6 hrs') and convert to float."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


# Canonical label maps for fuzzy/variant category values
MARKETING_CANON = {
    "social media": "Social Media", "social-media": "Social Media",
    "word of mouth": "Word of Mouth/Referral", "word of mouth / referral": "Word of Mouth/Referral",
    "word of mouth/referral": "Word of Mouth/Referral", "referral": "Word of Mouth/Referral",
    "travel agent": "Travel Agent", "travel agency": "Travel Agent",
    "airline partner": "Airline Partner", "airport signage": "Airport Signage",
    "organic app search": "Organic App Search",
}
TIER_CANON = {
    "stay airside": "Stay Airside",
    "airport lounge/relaxation": "Airport Lounge/Relaxation",
    "indoor micro-experience": "Indoor Micro-Experience",
    "city/culture/shopping": "City/Culture/Shopping",
    "city / culture / shopping": "City/Culture/Shopping",
    "culture & shopping": "City/Culture/Shopping",
    "premium fast-track/chauffeur": "Premium Fast-Track/Chauffeur",
    "premium fast track": "Premium Fast-Track/Chauffeur",
    "fast track premium": "Premium Fast-Track/Chauffeur",
}


def canonize(series, canon):
    return series.astype(object).apply(
        lambda v: canon.get(str(v).strip().lower(), str(v).strip()) if pd.notna(v) else v)


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    # ----- Generate raw -----
    raw = inject_raw_issues(generate_clean_data())
    raw_shape = raw.shape
    raw_missing = raw.isnull().sum()

    df = raw.copy()

    # --- CLEAN 1: remove duplicate passenger IDs ---
    dups_before = df["passenger_id"].duplicated().sum()
    df = df.drop_duplicates(subset="passenger_id", keep="first").reset_index(drop=True)

    # --- CLEAN 2: standardize boolean columns (new *_clean cols, keep raw) ---
    bool_cols = ["app_viewed", "eligibility_checked", "booked", "completed_on_time"]
    for col in bool_cols:
        df[col + "_clean"] = df[col].apply(standardize_bool)

    # --- CLEAN 3: text casing/spacing on category columns ---
    cat_cols = ["age_band", "travel_party_type", "visa_bucket", "luggage_status",
                "interest_category", "season_month"]
    for col in cat_cols:
        df[col] = clean_text(df[col])

    # --- CLEAN 4 + 5: canonical marketing channel & tier names ---
    df["marketing_channel"] = canonize(df["marketing_channel"], MARKETING_CANON)
    df["recommended_tier_clean"] = canonize(df["recommended_tier"], TIER_CANON)
    df["selected_package_clean"] = canonize(df["selected_package"], TIER_CANON)

    # --- CLEAN 6: numeric-as-text -> numeric ---
    df["layover_duration_hrs_clean"] = coerce_numeric(df["layover_duration_hrs"])
    df["paid_amount_aed_clean"] = coerce_numeric(df["paid_amount_aed"])

    # --- CLEAN 7: impossible layover durations -> NaN (out of 2-26 range) ---
    invalid_dur_mask = (df["layover_duration_hrs_clean"] < 2) | (df["layover_duration_hrs_clean"] > 26)
    invalid_dur_count = int(invalid_dur_mask.sum())
    df.loc[invalid_dur_mask, "layover_duration_hrs_clean"] = np.nan

    # --- CLEAN 8: negative paid amounts -> NaN ---
    neg_paid_mask = df["paid_amount_aed_clean"] < 0
    neg_paid_count = int(neg_paid_mask.sum())
    df.loc[neg_paid_mask, "paid_amount_aed_clean"] = np.nan

    # --- CLEAN 9: coerce outcome/count columns to numeric (in place) ---
    # NOTE: raw fatigue_level and budget_tier are intentionally LEFT UNTOUCHED here;
    # their analysis-ready versions are built as separate *_clean columns below so the
    # raw survey values (including non-response) are preserved.
    for col in ["satisfaction_rating", "referral_intent", "risk_comfort",
                "packages_viewed", "layover_confidence_score", "late_return_risk_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- CLEAN 10: missingness flags for survey non-response columns ---
    # These make missingness an explicit, analyzable variable (e.g. does a missing
    # response correlate with lower conversion?) rather than a silent gap.
    df["fatigue_missing_flag"] = df["fatigue_level"].isna()
    df["budget_missing_flag"] = df["budget_tier"].isna()

    # --- CLEAN 11: analysis-ready clean versions (raw columns preserved) ---
    # fatigue_level_clean: ordinal 1-5, numeric-coerced. Missing values are NOT
    # imputed (inventing an ordinal survey response would fabricate data); NaN is
    # preserved and tracked by fatigue_missing_flag.
    df["fatigue_level_clean"] = pd.to_numeric(df["fatigue_level"], errors="coerce")

    # budget_tier_clean: standardized casing/spacing, with missing labelled "Unknown".
    # An explicit "Unknown" category is used (rather than NaN) because pandas groupby
    # silently DROPS NaN rows, which would make those sessions disappear from dashboard
    # charts. Labelling keeps them visible as a transparent slice for filtering/grouping.
    budget_canon = {"budget": "Budget", "mid-range": "Mid-Range", "premium": "Premium"}
    df["budget_tier_clean"] = df["budget_tier"].apply(
        lambda v: budget_canon.get(str(v).strip().lower(), "Unknown") if pd.notna(v) else "Unknown")

    # NOTE on missing values:
    #   - fatigue_level / budget_tier missing = genuine survey non-response. Raw kept
    #     as-is; clean versions preserve the gap (NaN for ordinal, "Unknown" for category).
    #   - satisfaction_rating / referral_intent / paid_amount missing for non-bookings
    #     = LOGICALLY missing. Preserved as NaN on purpose, NOT imputed.

    # =================================================================
    # PART 5 — FEATURE ENGINEERING
    # =================================================================
    # time_buffer_mins_clean: usable minutes after fixed 90-min transit overhead
    df["time_buffer_mins_clean"] = (df["layover_duration_hrs_clean"] * 60 - 90).clip(lower=0)

    # confidence_band_clean: re-band from the numeric score (authoritative)
    df["confidence_band_clean"] = pd.cut(
        df["layover_confidence_score"], [-0.1, 39, 69, 100],
        labels=["Low", "Medium", "High"]).astype(object)

    # conversion_flag: did the passenger book (standardized)
    df["conversion_flag"] = df["booked_clean"]

    # package_viewed_flag: viewed at least one package
    df["package_viewed_flag"] = df["packages_viewed"] > 0

    # completed_flag: booked AND an on-time outcome was recorded
    df["completed_flag"] = df["booked_clean"].fillna(False) & df["completed_on_time_clean"].notna()

    # late_return_flag_clean: SIMULATED risk indicator; NaN preserved for non-bookings
    df["late_return_flag_clean"] = np.where(
        df["late_return_risk_score"].notna(), df["late_return_risk_score"] > 55, np.nan)

    # revenue_category: banded spend (logical "No Spend" for non-bookings)
    def rev_cat(a):
        if pd.isna(a):
            return "No Spend"
        if a < 300:
            return "Low"
        if a <= 800:
            return "Mid"
        return "High"
    df["revenue_category"] = df["paid_amount_aed_clean"].apply(rev_cat)

    # funnel_stage_clean: furthest funnel stage reached
    def funnel(r):
        if pd.notna(r["satisfaction_rating"]):
            return "Completed"
        if r["conversion_flag"] is True:
            return "Booked"
        if r["package_viewed_flag"]:
            return "Package Viewed"
        if r["eligibility_checked_clean"] is True:
            return "Eligibility Checked"
        return "App Viewed"
    df["funnel_stage_clean"] = df.apply(funnel, axis=1)

    # tier_category: risk grouping of the recommended tier
    tier_cat_map = {
        "Stay Airside": "Airside/Low-Risk", "Airport Lounge/Relaxation": "Airside/Low-Risk",
        "Indoor Micro-Experience": "Indoor/Moderate-Risk",
        "City/Culture/Shopping": "Outdoor/Higher-Risk",
        "Premium Fast-Track/Chauffeur": "Outdoor/Higher-Risk"}
    df["tier_category"] = df["recommended_tier_clean"].map(tier_cat_map).fillna("Unclassified")

    # high_value_lead_flag: high confidence + premium budget + engaged (business priority)
    df["high_value_lead_flag"] = (
        (df["confidence_band_clean"] == "High")
        & (df["budget_tier_clean"] == "Premium")
        & (df["packages_viewed"] >= 2)
    )

    # engagement_level: derived from packages viewed
    def eng(p):
        if pd.isna(p) or p == 0:
            return "None"
        if p <= 2:
            return "Low"
        if p <= 4:
            return "Medium"
        return "High"
    df["engagement_level"] = df["packages_viewed"].apply(eng)

    # safe_to_leave_flag: rule-based feasibility — enough buffer, eligible, not severe risk
    df["safe_to_leave_flag"] = (
        (df["time_buffer_mins_clean"] >= 180)
        & (df["visa_bucket"].isin(["Visa-Free/On-Arrival", "Transit-Visa-Required"]))
        & (df["layover_confidence_score"] >= 40)
    )

    # recommendation_match_flag: did the booked package match the system recommendation
    df["recommendation_match_flag"] = np.where(
        df["selected_package_clean"].notna(),
        df["selected_package_clean"] == df["recommended_tier_clean"], np.nan)

    engineered = [
        "time_buffer_mins_clean", "confidence_band_clean", "conversion_flag",
        "package_viewed_flag", "completed_flag", "late_return_flag_clean",
        "revenue_category", "funnel_stage_clean", "tier_category",
        "high_value_lead_flag", "engagement_level", "safe_to_leave_flag",
        "recommendation_match_flag",
        # clean versions + missingness flags for survey non-response columns
        "fatigue_level_clean", "budget_tier_clean",
        "fatigue_missing_flag", "budget_missing_flag",
    ]

    final_shape = df.shape
    final_missing = df.isnull().sum()

    # =================================================================
    # EXPORT — the ONLY file written
    # =================================================================
    df.to_csv("sample_data.csv", index=False)

    # =================================================================
    # REPORTING
    # =================================================================
    print("=" * 72)
    print("GATEESCAPE DUBAI — DATA PREPARATION SUMMARY")
    print("=" * 72)
    print(f"Raw data shape (before cleaning):   {raw_shape[0]} rows x {raw_shape[1]} cols")
    print(f"Final cleaned shape (after):        {final_shape[0]} rows x {final_shape[1]} cols")
    print(f"Duplicate passenger IDs removed:    {dups_before}")
    print(f"Invalid layover durations fixed:    {invalid_dur_count}")
    print(f"Negative paid amounts fixed:        {neg_paid_count}")

    print("\n--- Missing values BEFORE cleaning (selected key columns) ---")
    watch = ["fatigue_level", "budget_tier", "paid_amount_aed",
             "satisfaction_rating", "referral_intent"]
    print(raw_missing[watch].to_string())

    print("\n--- Missing values AFTER cleaning (clean/derived columns) ---")
    watch_after = ["fatigue_level_clean", "budget_tier_clean", "paid_amount_aed_clean",
                   "satisfaction_rating", "referral_intent"]
    print(final_missing[watch_after].to_string())
    print(f"  fatigue_missing_flag = True:  {int(df['fatigue_missing_flag'].sum())} rows")
    print(f"  budget_missing_flag  = True:  {int(df['budget_missing_flag'].sum())} rows")
    print("(budget_tier_clean shows 0 missing because non-response is labelled 'Unknown'\n"
          " and tracked separately by budget_missing_flag. fatigue_level_clean keeps NaN\n"
          " since an ordinal response cannot be invented. satisfaction/referral/paid missing\n"
          " for non-bookings are LOGICAL nulls, preserved intentionally.)")

    print("\n--- Engineered features created ---")
    for f in engineered:
        print(f"  - {f}")

    print("\n--- Sample rows from final dataset ---")
    sample_cols = ["passenger_id", "visa_bucket", "layover_duration_hrs_clean",
                   "time_buffer_mins_clean", "layover_confidence_score",
                   "confidence_band_clean", "recommended_tier_clean", "conversion_flag",
                   "paid_amount_aed_clean", "revenue_category", "engagement_level",
                   "safe_to_leave_flag", "recommendation_match_flag", "funnel_stage_clean"]
    print(df[sample_cols].sample(6, random_state=3).to_string(index=False))

    # =================================================================
    # VALIDATION CHECKLIST — raw vs cleaned
    # =================================================================
    print("\n" + "=" * 72)
    print("VALIDATION CHECKLIST (raw vs cleaned)")
    print("=" * 72)

    checks = []

    # 1. Duplicates removed
    checks.append(("Duplicate passenger IDs removed",
                   df["passenger_id"].duplicated().sum() == 0))

    # 2. Invalid durations fixed (no values outside 2-26 remain in clean col)
    bad_dur_remaining = df["layover_duration_hrs_clean"].dropna()
    checks.append(("Invalid layover durations fixed (2-26 range)",
                   ((bad_dur_remaining >= 2) & (bad_dur_remaining <= 26)).all()))

    # 3. Negative paid amounts fixed
    checks.append(("Negative paid amounts removed",
                   not (df["paid_amount_aed_clean"].dropna() < 0).any()))

    # 4. Boolean values standardised (only True/False/NaN remain)
    bool_ok = all(
        set(df[c + "_clean"].dropna().unique()).issubset({True, False})
        for c in bool_cols)
    checks.append(("Boolean values standardised to True/False", bool_ok))

    # 5. Category labels standardised (no stray casing/whitespace in tiers/channel)
    tier_vals = set(df["recommended_tier_clean"].dropna().unique())
    valid_tiers = set(TIER_CANON.values())
    mkt_vals = set(df["marketing_channel"].dropna().unique())
    valid_mkt = set(MARKETING_CANON.values())
    checks.append(("Tier labels standardised",
                   tier_vals.issubset(valid_tiers)))
    checks.append(("Marketing channel labels standardised",
                   mkt_vals.issubset(valid_mkt)))

    # 6. Numeric columns converted correctly
    numeric_ok = (pd.api.types.is_numeric_dtype(df["layover_duration_hrs_clean"])
                  and pd.api.types.is_numeric_dtype(df["paid_amount_aed_clean"]))
    checks.append(("Numeric columns converted from text", numeric_ok))

    # 7. Engineered features created
    checks.append(("All engineered features present",
                   all(f in df.columns for f in engineered)))

    # 8. Logical nulls preserved (non-bookings must have NaN satisfaction)
    non_booking_satis = df.loc[df["conversion_flag"] != True, "satisfaction_rating"]
    checks.append(("Logical nulls preserved (no satisfaction for non-bookings)",
                   non_booking_satis.isna().all()))

    # 9. Raw fatigue/budget columns preserved (not overwritten by cleaning)
    checks.append(("Raw fatigue_level & budget_tier columns preserved",
                   "fatigue_level" in df.columns and "budget_tier" in df.columns))

    # 10. fatigue_level_clean numeric with NaN preserved (not imputed)
    fatigue_clean_ok = (
        pd.api.types.is_numeric_dtype(df["fatigue_level_clean"])
        and df["fatigue_level_clean"].isna().sum() == df["fatigue_missing_flag"].sum()
    )
    checks.append(("fatigue_level_clean numeric & missing NOT imputed", fatigue_clean_ok))

    # 11. budget_tier_clean has no NaN (missing labelled 'Unknown'), valid categories only
    valid_budget = {"Budget", "Mid-Range", "Premium", "Unknown"}
    budget_clean_ok = (
        df["budget_tier_clean"].isna().sum() == 0
        and set(df["budget_tier_clean"].unique()).issubset(valid_budget)
    )
    checks.append(("budget_tier_clean labelled (no NaN, valid categories)", budget_clean_ok))

    # 12. Missing flags correctly track raw missingness
    flags_ok = (
        df["fatigue_missing_flag"].sum() == df["fatigue_level"].isna().sum()
        and df["budget_missing_flag"].sum() == df["budget_tier"].isna().sum()
    )
    checks.append(("Missing flags match raw missingness", flags_ok))

    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    all_passed = all(p for _, p in checks)
    print("\nOVERALL:", "ALL CHECKS PASSED ✅" if all_passed else "SOME CHECKS FAILED ❌")
    print("\nOutput written: sample_data.csv")


if __name__ == "__main__":
    main()
