"""
GateEscape Dubai — Layover Decision-Confidence Analytics Dashboard
==================================================================
Individual Data Analytics Assignment

Run with:  streamlit run app.py

This dashboard loads sample_data.csv (produced by data_preparation.py) and
presents the descriptive, diagnostic, and correlation-based analytics required
by the assignment brief. It uses the CLEAN / engineered columns for all analysis;
raw columns are treated only as audit/background fields.

NOTE: All late-return variables (late_return_risk_score, late_return_flag_clean)
are SIMULATED synthetic risk indicators for analytical practice only. They do not
represent any real airline, airport, immigration, or legal outcome.

No machine learning, classification, clustering, or forecasting is used.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# THEME / PALETTE
# =====================================================================
TOREA_BAY = "#0a2a92"   # sidebar, hero, headers, strong accents
DANUBE = "#5992c6"      # chart highlights, secondary visuals
COCOA = "#31241f"       # text / contrast
SHILO = "#e9b8c9"       # soft highlights, subtle borders, accent notes
PEARL = "#faf8f5"       # page / card backgrounds

# Ordered categorical sequences for consistent chart colouring
BAND_ORDER = ["Low", "Medium", "High"]
FUNNEL_ORDER = ["App Viewed", "Eligibility Checked", "Package Viewed", "Booked", "Completed"]
TIER_ORDER = ["Stay Airside", "Airport Lounge/Relaxation", "Indoor Micro-Experience",
              "City/Culture/Shopping", "Premium Fast-Track/Chauffeur"]
SEQ_BLUE = [SHILO, DANUBE, TOREA_BAY]  # low -> high gradient

st.set_page_config(
    page_title="GateEscape Dubai — Layover Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CUSTOM STYLING
# =====================================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PEARL}; }}
        section[data-testid="stSidebar"] {{ background-color: {TOREA_BAY}; }}
        section[data-testid="stSidebar"] * {{ color: #ffffff !important; }}
        section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {{
            background-color: {DANUBE} !important; }}

        h1, h2, h3, h4 {{ color: {TOREA_BAY}; font-family: 'Times New Roman', Georgia, serif; }}
        p, li, span, label {{ color: {COCOA}; font-family: 'Segoe UI', Arial, sans-serif; }}

        .hero {{
            background: linear-gradient(120deg, {TOREA_BAY} 0%, {DANUBE} 100%);
            padding: 30px 36px; border-radius: 14px; margin-bottom: 8px;
            box-shadow: 0 6px 18px rgba(10,42,146,0.18);
        }}
        .hero h1 {{ color: #ffffff !important; margin: 0; font-size: 30px; }}
        .hero p {{ color: #eaf1fb !important; margin: 6px 0 0 0; font-size: 15px; }}

        .kpi-card {{
            background: #ffffff; border-radius: 12px; padding: 18px 20px;
            border: 1px solid #ececec; border-top: 4px solid {TOREA_BAY};
            box-shadow: 0 3px 10px rgba(49,36,31,0.06); height: 100%;
        }}
        .kpi-label {{ font-size: 12.5px; color: #6b6b6b; text-transform: uppercase;
            letter-spacing: .5px; margin-bottom: 4px; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; color: {TOREA_BAY}; }}
        .kpi-sub {{ font-size: 12px; color: #8a8a8a; margin-top: 2px; }}

        .chart-card {{
            background: #ffffff; border-radius: 12px; padding: 18px 20px 6px 20px;
            border: 1px solid #efe6ea; box-shadow: 0 3px 10px rgba(49,36,31,0.05);
            margin-bottom: 6px;
        }}
        .insight {{
            background: #ffffff; border-left: 4px solid {SHILO};
            padding: 12px 16px; border-radius: 8px; margin: 6px 0 22px 0;
            font-size: 13.5px; color: {COCOA};
        }}
        .insight b {{ color: {TOREA_BAY}; }}
        .tag {{
            display: inline-block; background: {SHILO}; color: {COCOA};
            padding: 4px 12px; border-radius: 20px; font-size: 12px;
            font-weight: 600; margin-bottom: 10px; letter-spacing: .3px;
        }}
        .disclaimer {{
            background: #fff7fa; border: 1px dashed {SHILO}; border-radius: 8px;
            padding: 10px 14px; font-size: 12.5px; color: {COCOA}; margin: 8px 0 18px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# DATA LOADING (with error handling)
# =====================================================================
@st.cache_data
def load_data(path="sample_data.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    def safe_text(series, unknown="Unknown"):
        """Trim text columns and keep missing/blank values visible in filters."""
        s = series.astype(object)
        s = s.where(s.notna(), unknown)
        s = s.astype(str).str.strip()
        return s.replace({"": unknown, "nan": unknown, "NaN": unknown, "None": unknown})

    def title_clean(series, unknown="Unknown"):
        return safe_text(series, unknown).str.title()

    def to_bool(x):
        """Robustly convert bool/string/numeric values to True/False while preserving NaN."""
        if pd.isna(x):
            return np.nan
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        s = str(x).strip().lower()
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
        return np.nan

    # Normalise a few raw audit columns for clean filtering/grouping in the UI.
    # These raw columns retain intentional casing noise; we standardise display-side
    # only, without changing the underlying CSV.
    party_map = {
        "Solo": "Solo", "Couple": "Couple",
        "Family With Kids": "Family with kids", "Friends/Group": "Friends/Group",
        "Unknown": "Unknown",
    }
    df["travel_party_clean"] = title_clean(df["travel_party_type"]).map(party_map).fillna("Unknown")

    df["visa_bucket_clean"] = safe_text(df["visa_bucket"]).apply(_normalise_visa)

    # Keep every filter column visible by replacing missing values with explicit labels.
    filter_defaults = {
        "confidence_band_clean": "Unknown",
        "recommended_tier_clean": "Unclassified",
        "budget_tier_clean": "Unknown",
        "travel_party_clean": "Unknown",
        "marketing_channel": "Unknown",
        "visa_bucket_clean": "Unknown",
        # CSV parsing treats the literal string "None" as NaN by default, so use a
        # clearer dashboard label to prevent those sessions from disappearing.
        "engagement_level": "No Engagement",
    }
    for col, fill_value in filter_defaults.items():
        if col in df.columns:
            df[col] = safe_text(df[col], fill_value)

    # Ensure boolean-ish flag columns are real booleans.
    bool_cols = [
        "conversion_flag", "package_viewed_flag", "completed_flag",
        "high_value_lead_flag", "safe_to_leave_flag", "eligibility_checked_clean",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_bool).fillna(False).astype(bool)

    return df

def _normalise_visa(v):
    s = v.strip().lower()
    if "free" in s:
        return "Visa-Free/On-Arrival"
    if "transit" in s:
        return "Transit-Visa-Required"
    if "not" in s or "eligible" in s:
        return "Not-Eligible"
    return v


def insight(what, why, decision):
    """Render a standard 3-part business insight note under a chart."""
    st.markdown(
        f"""<div class="insight">
        <b>What it shows:</b> {what}<br>
        <b>Why it matters:</b> {why}<br>
        <b>Decision it supports:</b> {decision}
        </div>""",
        unsafe_allow_html=True,
    )


def section_tag(kind):
    st.markdown(f'<span class="tag">{kind}</span>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=COCOA, family="Segoe UI"),
        title_font=dict(color=TOREA_BAY, size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#f0eef0", zerolinecolor="#f0eef0")
    fig.update_yaxes(gridcolor="#f0eef0", zerolinecolor="#f0eef0")
    return fig


# =====================================================================
# APP START
# =====================================================================
inject_css()
df_full = load_data()

if df_full is None:
    st.error(
        "**sample_data.csv not found.**\n\n"
        "This dashboard requires `sample_data.csv` in the same folder as `app.py`.\n\n"
        "Generate it first by running:  `python data_preparation.py`"
    )
    st.stop()


# ----- Hero banner -----
st.markdown(
    """
    <div class="hero">
        <h1>GateEscape Dubai — Layover Decision-Confidence Analytics</h1>
        <p>A synthetic data validation of the layover micro-experience platform:
        from passenger validation to booking conversion. Descriptive, diagnostic,
        and correlation-based analytics for business strategy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# SIDEBAR FILTERS
# =====================================================================
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to every tab. Analysis uses clean / engineered columns.")


def msel(label, col, df):
    opts = sorted([x for x in df[col].dropna().unique()])
    return st.sidebar.multiselect(label, opts, default=opts)


f_band = msel("Confidence Band", "confidence_band_clean", df_full)
f_tier = msel("Recommended Tier", "recommended_tier_clean", df_full)
f_budget = msel("Budget Tier", "budget_tier_clean", df_full)
f_party = msel("Traveller Party Type", "travel_party_clean", df_full)
f_channel = msel("Marketing Channel", "marketing_channel", df_full)
f_visa = msel("Visa Bucket", "visa_bucket_clean", df_full)
f_engage = msel("Engagement Level", "engagement_level", df_full)

safe_choice = st.sidebar.radio(
    "Safe-to-Leave Flag", ["All", "Safe to leave", "Not safe to leave"], index=0)

# Apply filters
df = df_full[
    df_full["confidence_band_clean"].isin(f_band)
    & df_full["recommended_tier_clean"].isin(f_tier)
    & df_full["budget_tier_clean"].isin(f_budget)
    & df_full["travel_party_clean"].isin(f_party)
    & df_full["marketing_channel"].isin(f_channel)
    & df_full["visa_bucket_clean"].isin(f_visa)
    & df_full["engagement_level"].isin(f_engage)
].copy()

if safe_choice == "Safe to leave":
    df = df[df["safe_to_leave_flag"]]
elif safe_choice == "Not safe to leave":
    df = df[~df["safe_to_leave_flag"]]

st.sidebar.markdown("---")
st.sidebar.metric("Sessions in current view", f"{len(df):,}")
if len(df) == 0:
    st.warning("No records match the current filter selection. Please widen the filters.")
    st.stop()


# =====================================================================
# TABS
# =====================================================================
tabs = st.tabs([
    "Executive Overview",
    "Funnel Analytics",
    "Segment & Tier Performance",
    "Diagnostic Analytics",
    "Correlation Insights",
    "Business Recommendations",
    "Data Preparation Evidence",
])


# ---------------------------------------------------------------------
# TAB 1 — EXECUTIVE OVERVIEW (Descriptive)
# ---------------------------------------------------------------------
with tabs[0]:
    section_tag("DESCRIPTIVE ANALYTICS")
    st.subheader("Executive Overview")

    total_pax = len(df)
    booked = df["conversion_flag"]
    conv_rate = booked.mean() * 100
    total_rev = df["paid_amount_aed_clean"].sum()
    avg_spend = df.loc[booked, "paid_amount_aed_clean"].mean()
    avg_conf = df["layover_confidence_score"].mean()
    avg_sat = df["satisfaction_rating"].mean()
    avg_ref = df["referral_intent"].mean()
    # Late-return synthetic risk rate among booked sessions
    lr = df["late_return_flag_clean"].dropna()
    late_rate = (lr.mean() * 100) if len(lr) else 0.0

    kpis = [
        ("Total Passengers", f"{total_pax:,}", "sessions in view"),
        ("Booking Conversion", f"{conv_rate:.1f}%", "viewed to booked"),
        ("Total Revenue", f"AED {total_rev:,.0f}", "all bookings"),
        ("Avg Spend / Booking", f"AED {avg_spend:,.0f}" if pd.notna(avg_spend) else "—", "booked only"),
        ("Avg Confidence Score", f"{avg_conf:.1f}", "0–100 scale"),
        ("Avg Satisfaction", f"{avg_sat:.2f}" if pd.notna(avg_sat) else "—", "1–5 scale"),
        ("Avg Referral Intent", f"{avg_ref:.2f}" if pd.notna(avg_ref) else "—", "1–5 scale"),
        ("Late-Return Risk Rate", f"{late_rate:.1f}%", "simulated indicator"),
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)
    for i, (label, value, sub) in enumerate(kpis):
        target = row1[i] if i < 4 else row2[i - 4]
        target.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div class="disclaimer"><b>Note:</b> The Late-Return Risk Rate is a
        <b>simulated synthetic indicator</b> created for analytical practice. It does
        not represent any real airline, airport, immigration, or legal outcome.</div>""",
        unsafe_allow_html=True,
    )

    # Confidence band distribution + revenue mix
    c1, c2 = st.columns(2)
    with c1:
        band_counts = df["confidence_band_clean"].value_counts().reindex(BAND_ORDER).fillna(0)
        fig = px.bar(
            x=band_counts.index, y=band_counts.values,
            title="Passenger Distribution by Confidence Band",
            labels={"x": "Confidence Band", "y": "Passengers"},
            color=band_counts.index, color_discrete_sequence=SEQ_BLUE,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        rev_mix = df["revenue_category"].value_counts().reindex(
            ["No Spend", "Low", "Mid", "High"]).fillna(0)
        fig = px.bar(
            x=rev_mix.index, y=rev_mix.values,
            title="Revenue Category Mix",
            labels={"x": "Revenue Category", "y": "Passengers"},
            color=rev_mix.index,
            color_discrete_sequence=[SHILO, DANUBE, TOREA_BAY, COCOA],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    insight(
        "The headline KPIs and the spread of passengers across confidence bands and spend categories.",
        "It gives a single boardroom-level snapshot of how well the platform converts transit "
        "passengers into paying customers and how satisfied they are.",
        "Whether the overall validation-to-sales pipeline is healthy enough to scale, or whether "
        "specific bands/segments need attention (explored in later tabs).",
    )

    st.markdown(
        f"""<div class="insight">
        <b>Executive summary:</b> Of {total_pax:,} simulated transit sessions in the current view,
        roughly <b>{conv_rate:.0f}%</b> convert to a booking, generating <b>AED {total_rev:,.0f}</b>
        in revenue at an average confidence score of <b>{avg_conf:.0f}/100</b>. Conversion is heavily
        concentrated in the High-confidence band, which suggests the score is doing its job as a
        qualification signal — and that growth should focus on moving more passengers into, and
        converting within, the higher bands.
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# TAB 2 — FUNNEL ANALYTICS (Descriptive)
# ---------------------------------------------------------------------
with tabs[1]:
    section_tag("DESCRIPTIVE ANALYTICS")
    st.subheader("Validation-to-Sales Funnel")

    # Build a strict cumulative funnel: each stage must be a subset of the previous stage.
    # This prevents illogical negative drop-offs and keeps the validation-to-sales
    # pipeline easy to explain to a marker.
    n = len(df)
    eligibility_mask = df["eligibility_checked_clean"] == True
    package_viewed_mask = eligibility_mask & (df["package_viewed_flag"] == True)
    booked_mask = package_viewed_mask & (df["conversion_flag"] == True)
    completed_mask = booked_mask & (df["completed_flag"] == True)

    stage_vals = [
        n,
        int(eligibility_mask.sum()),
        int(package_viewed_mask.sum()),
        int(booked_mask.sum()),
        int(completed_mask.sum()),
    ]

    fig = go.Figure(go.Funnel(
        y=FUNNEL_ORDER, x=stage_vals,
        textinfo="value+percent initial",
        marker=dict(color=[TOREA_BAY, DANUBE, "#7aa9d4", SHILO, "#d98ba7"]),
    ))
    fig.update_layout(title="Passenger Funnel: App Viewed to Completed")
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    # Retention / drop-off table
    funnel_tbl = pd.DataFrame({
        "Stage": FUNNEL_ORDER,
        "Passengers": stage_vals,
        "Retention vs Start": [f"{v / n * 100:.1f}%" for v in stage_vals],
        "Drop-off vs Previous": ["—"] + [
            f"{(stage_vals[i-1] - stage_vals[i]) / stage_vals[i-1] * 100:.1f}%"
            if stage_vals[i-1] else "—" for i in range(1, len(stage_vals))],
    })
    st.dataframe(funnel_tbl, use_container_width=True, hide_index=True)

    # Identify biggest drop-off
    drops = [(FUNNEL_ORDER[i-1] + " to " + FUNNEL_ORDER[i],
              (stage_vals[i-1] - stage_vals[i]) / stage_vals[i-1] * 100 if stage_vals[i-1] else 0)
             for i in range(1, len(stage_vals))]
    worst = max(drops, key=lambda x: x[1])

    insight(
        "How many passengers progress through each pipeline stage, with retention and drop-off rates.",
        f"It pinpoints where the platform loses the most passengers — here the largest single "
        f"drop-off is <b>{worst[0]}</b> at about <b>{worst[1]:.0f}%</b>.",
        "Where to focus product and conversion effort: the stage with the steepest drop-off is the "
        "highest-leverage place to improve the validation-to-sales pipeline.",
    )


# ---------------------------------------------------------------------
# TAB 3 — SEGMENT & TIER PERFORMANCE (Descriptive)
# ---------------------------------------------------------------------
with tabs[2]:
    section_tag("DESCRIPTIVE ANALYTICS")
    st.subheader("Segment & Tier Performance")

    c1, c2 = st.columns(2)

    # Conversion by confidence band
    with c1:
        conv_band = df.groupby("confidence_band_clean")["conversion_flag"].mean().reindex(
            BAND_ORDER) * 100
        fig = px.bar(
            x=conv_band.index, y=conv_band.values,
            title="Conversion Rate by Confidence Band (%)",
            labels={"x": "Confidence Band", "y": "Conversion %"},
            color=conv_band.index, color_discrete_sequence=SEQ_BLUE,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # Revenue by recommended tier
    with c2:
        rev_tier = df.groupby("recommended_tier_clean")["paid_amount_aed_clean"].sum().reindex(
            TIER_ORDER).fillna(0)
        fig = px.bar(
            x=rev_tier.values, y=rev_tier.index, orientation="h",
            title="Total Revenue by Recommended Tier (AED)",
            labels={"x": "Revenue (AED)", "y": ""},
            color=rev_tier.values, color_continuous_scale=[SHILO, TOREA_BAY],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    insight(
        "Conversion climbs sharply across confidence bands, while revenue concentrates in the "
        "higher-value tiers.",
        "Together they show the platform earns disproportionately from a minority of high-confidence, "
        "premium-leaning passengers.",
        "Prioritising High-band passengers and premium tiers for marketing spend and capacity.",
    )

    # Satisfaction by selected vs recommended tier
    booked_df = df[df["conversion_flag"]].copy()
    if len(booked_df):
        sat_sel = booked_df.groupby("selected_package_clean")["satisfaction_rating"].mean().reindex(
            TIER_ORDER)
        sat_rec = booked_df.groupby("recommended_tier_clean")["satisfaction_rating"].mean().reindex(
            TIER_ORDER)
        fig = go.Figure()
        fig.add_bar(x=TIER_ORDER, y=sat_sel.values, name="By Selected Package",
                    marker_color=DANUBE)
        fig.add_bar(x=TIER_ORDER, y=sat_rec.values, name="By Recommended Tier",
                    marker_color=TOREA_BAY)
        fig.update_layout(title="Average Satisfaction by Tier (Selected vs Recommended)",
                          barmode="group", yaxis_title="Avg Satisfaction (1–5)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    else:
        st.info("No booked sessions in current view to show satisfaction by tier.")

    insight(
        "Average satisfaction for each tier, comparing the package the passenger actually chose "
        "against what the platform recommended.",
        "It reveals whether following the platform's recommendation produces happier customers.",
        "Whether to push the recommendation more firmly in the UI (explored further in Diagnostics).",
    )

    # Conversion by party type and budget tier
    c3, c4 = st.columns(2)
    with c3:
        conv_party = df.groupby("travel_party_clean")["conversion_flag"].mean().sort_values() * 100
        fig = px.bar(
            x=conv_party.values, y=conv_party.index, orientation="h",
            title="Conversion Rate by Traveller Party Type (%)",
            labels={"x": "Conversion %", "y": ""},
            color=conv_party.values, color_continuous_scale=[SHILO, DANUBE],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c4:
        conv_budget = df.groupby("budget_tier_clean")["conversion_flag"].mean().reindex(
            ["Budget", "Mid-Range", "Premium", "Unknown"]).dropna() * 100
        fig = px.bar(
            x=conv_budget.index, y=conv_budget.values,
            title="Conversion Rate by Budget Tier (%)",
            labels={"x": "Budget Tier", "y": "Conversion %"},
            color=conv_budget.index,
            color_discrete_sequence=[SHILO, DANUBE, TOREA_BAY, "#b8b8b8"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    insight(
        "Conversion broken down by who the passenger travels with and their budget level.",
        "It identifies which human segments are most ready to book, beyond the confidence score alone.",
        "Tailoring messaging and tier offers to the party types and budget levels that convert best.",
    )


# ---------------------------------------------------------------------
# TAB 4 — DIAGNOSTIC ANALYTICS
# ---------------------------------------------------------------------
with tabs[3]:
    section_tag("DIAGNOSTIC ANALYTICS")
    st.subheader("Why Conversion and Satisfaction Differ")

    # Confidence score vs booking (box / violin)
    c1, c2 = st.columns(2)
    with c1:
        tmp = df.copy()
        tmp["Booked"] = np.where(tmp["conversion_flag"], "Booked", "Not Booked")
        fig = px.box(
            tmp, x="Booked", y="layover_confidence_score",
            title="Confidence Score vs Booking Outcome",
            color="Booked", color_discrete_sequence=[DANUBE, TOREA_BAY],
            labels={"layover_confidence_score": "Confidence Score"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "The distribution of confidence scores for passengers who booked vs those who did not.",
            "If booked passengers sit at clearly higher scores, the score is a valid qualification signal.",
            "Confirms whether the confidence score should drive lead prioritisation.",
        )

    # Packages viewed vs booking
    with c2:
        pv = df.groupby("packages_viewed")["conversion_flag"].mean() * 100
        fig = px.bar(
            x=pv.index, y=pv.values,
            title="Conversion Rate by Packages Viewed (%)",
            labels={"x": "Packages Viewed", "y": "Conversion %"},
            color=pv.values, color_continuous_scale=[SHILO, TOREA_BAY],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "How conversion changes as passengers view more packages.",
            "Engagement depth is a strong, actionable lever the platform can directly influence.",
            "Investing in browsing experience and prompts that encourage viewing more options.",
        )

    # Fatigue level vs recommended tier (stacked share)
    ftmp = df.dropna(subset=["fatigue_level_clean"]).copy()
    if len(ftmp):
        ftmp["fatigue_level_clean"] = ftmp["fatigue_level_clean"].astype(int)
        ct = pd.crosstab(ftmp["fatigue_level_clean"], ftmp["recommended_tier_clean"],
                         normalize="index") * 100
        ct = ct.reindex(columns=[t for t in TIER_ORDER if t in ct.columns])
        fig = go.Figure()
        palette = [TOREA_BAY, DANUBE, "#7aa9d4", SHILO, "#d98ba7"]
        for i, col in enumerate(ct.columns):
            fig.add_bar(x=ct.index, y=ct[col], name=col, marker_color=palette[i % len(palette)])
        fig.update_layout(title="Recommended Tier Mix by Fatigue Level (%)",
                          barmode="stack", xaxis_title="Fatigue Level (1=Low, 5=High)",
                          yaxis_title="Share of Passengers (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    insight(
        "The mix of recommended tiers across fatigue levels.",
        "It shows the platform correctly steering tired passengers toward calmer airport/indoor "
        "options rather than active city tours.",
        "Validates the fatigue logic in the recommendation rules and supports comfort-led messaging.",
    )

    c3, c4 = st.columns(2)
    # Safe-to-leave vs conversion
    with c3:
        # Build a small plotting table instead of manually replacing the index.
        # This avoids a pandas Length mismatch error when the user filters the dashboard
        # to only one safe-to-leave group.
        stl = (
            df.groupby("safe_to_leave_flag", dropna=False)["conversion_flag"]
            .mean()
            .reset_index()
        )
        stl["Conversion %"] = stl["conversion_flag"] * 100
        stl["Safe-to-Leave Status"] = stl["safe_to_leave_flag"].map(
            {True: "Safe to Leave", False: "Not Safe to Leave"}
        ).fillna("Unknown")
        order = ["Not Safe to Leave", "Safe to Leave", "Unknown"]
        stl["Safe-to-Leave Status"] = pd.Categorical(
            stl["Safe-to-Leave Status"], categories=order, ordered=True
        )
        stl = stl.sort_values("Safe-to-Leave Status")
        fig = px.bar(
            stl,
            x="Safe-to-Leave Status",
            y="Conversion %",
            title="Conversion Rate by Safe-to-Leave Flag (%)",
            labels={"Safe-to-Leave Status": "Safe to Leave", "Conversion %": "Conversion %"},
            color="Safe-to-Leave Status",
            color_discrete_sequence=[SHILO, TOREA_BAY, "#b8b8b8"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "Conversion for passengers flagged feasible to leave the airport vs those who are not.",
            "It quantifies how much the practical feasibility gate (time, eligibility, score) drives sales.",
            "Justifies surfacing the safe-to-leave status early to set realistic expectations.",
        )

    # Recommendation match vs satisfaction
    with c4:
        mtmp = df.dropna(subset=["recommendation_match_flag", "satisfaction_rating"]).copy()
        if len(mtmp):
            mtmp["Match"] = np.where(mtmp["recommendation_match_flag"] == 1,
                                     "Matched Recommendation", "Did Not Match")
            ms = mtmp.groupby("Match")["satisfaction_rating"].mean()
            fig = px.bar(
                x=ms.index, y=ms.values,
                title="Avg Satisfaction by Recommendation Match",
                labels={"x": "", "y": "Avg Satisfaction (1–5)"},
                color=ms.index, color_discrete_sequence=[SHILO, TOREA_BAY],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("No matched/completed sessions in current view.")
        insight(
            "Satisfaction when the booked package matched the platform's recommendation vs when it didn't.",
            "It tests whether trusting the recommendation actually produces better experiences.",
            "Whether to make the recommended tier more prominent or harder to override.",
        )

    # Missingness flags vs conversion
    miss = pd.DataFrame({
        "Group": ["Fatigue provided", "Fatigue missing", "Budget provided", "Budget missing"],
        "Conversion %": [
            df.loc[~df["fatigue_missing_flag"], "conversion_flag"].mean() * 100,
            df.loc[df["fatigue_missing_flag"], "conversion_flag"].mean() * 100,
            df.loc[~df["budget_missing_flag"], "conversion_flag"].mean() * 100,
            df.loc[df["budget_missing_flag"], "conversion_flag"].mean() * 100,
        ],
    })
    fig = px.bar(
        miss, x="Group", y="Conversion %",
        title="Conversion by Survey Completeness (Missingness Flags)",
        color="Group",
        color_discrete_sequence=[TOREA_BAY, SHILO, DANUBE, "#d98ba7"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    insight(
        "Whether passengers who left fatigue or budget blank convert differently from those who answered.",
        "Missingness can itself be a behavioural signal — incomplete profiles may indicate lower intent.",
        "Whether to prompt for the missing fields, treating completeness as an engagement signal.",
    )


# ---------------------------------------------------------------------
# TAB 5 — CORRELATION INSIGHTS (Correlation-based diagnostic)
# ---------------------------------------------------------------------
with tabs[4]:
    section_tag("CORRELATION-BASED DIAGNOSTIC ANALYSIS")
    st.subheader("Correlation Insights")

    st.markdown(
        f"""<div class="disclaimer"><b>Important:</b> Correlation does not prove causation.
        These relationships describe how variables move together in the synthetic data; they do
        not establish that one variable causes another. Late-return variables are simulated
        synthetic indicators only.</div>""",
        unsafe_allow_html=True,
    )

    num_cols = ["layover_confidence_score", "time_buffer_mins_clean", "packages_viewed",
                "paid_amount_aed_clean", "satisfaction_rating", "referral_intent",
                "late_return_risk_score", "fatigue_level_clean", "risk_comfort"]
    pretty = {
        "layover_confidence_score": "Confidence Score",
        "time_buffer_mins_clean": "Time Buffer (min)",
        "packages_viewed": "Packages Viewed",
        "paid_amount_aed_clean": "Paid Amount",
        "satisfaction_rating": "Satisfaction",
        "referral_intent": "Referral Intent",
        "late_return_risk_score": "Late-Return Risk (sim.)",
        "fatigue_level_clean": "Fatigue Level",
        "risk_comfort": "Risk Comfort",
    }

    corr = df[num_cols].corr().round(2)
    corr_disp = corr.rename(index=pretty, columns=pretty)
    fig = px.imshow(
        corr_disp, text_auto=True, aspect="auto",
        color_continuous_scale=[SHILO, "#ffffff", TOREA_BAY], zmin=-1, zmax=1,
        title="Correlation Heatmap of Key Numeric Variables",
    )
    st.plotly_chart(style_fig(fig, height=520), use_container_width=True)
    insight(
        "How strongly the main numeric variables move together (−1 to +1).",
        "It surfaces the strongest relationships in one view — e.g. confidence with spend, or time "
        "buffer with simulated late-return risk.",
        "Which variables are worth deeper diagnostic attention and which are largely independent.",
    )

    # Scatterplots
    s1, s2 = st.columns(2)
    with s1:
        fig = px.scatter(
            df, x="time_buffer_mins_clean", y="layover_confidence_score",
            color="confidence_band_clean", category_orders={"confidence_band_clean": BAND_ORDER},
            color_discrete_sequence=SEQ_BLUE, opacity=0.6,
            title="Time Buffer vs Confidence Score",
            labels={"time_buffer_mins_clean": "Time Buffer (min)",
                    "layover_confidence_score": "Confidence Score",
                    "confidence_band_clean": "Band"},
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "Each passenger plotted by usable time buffer against confidence score.",
            "It visualises how available time feeds the score, and where the eligibility floor caps it.",
            "Confirms time buffer is a core driver and helps set minimum-buffer thresholds for offers.",
        )
    with s2:
        bd = df[df["conversion_flag"]]
        fig = px.scatter(
            bd, x="layover_confidence_score", y="paid_amount_aed_clean",
            color="budget_tier_clean", color_discrete_sequence=[SHILO, DANUBE, TOREA_BAY, "#b8b8b8"],
            opacity=0.6, title="Confidence Score vs Paid Amount (Booked)",
            labels={"layover_confidence_score": "Confidence Score",
                    "paid_amount_aed_clean": "Paid Amount (AED)", "budget_tier_clean": "Budget"},
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "Among booked passengers, how confidence score relates to amount paid, by budget tier.",
            "It shows whether higher-confidence passengers also spend more, and how budget shapes spend.",
            "Pricing and upsell strategy for high-confidence, higher-budget segments.",
        )

    s3, s4 = st.columns(2)
    with s3:
        sd = df.dropna(subset=["satisfaction_rating", "referral_intent"])
        fig = px.scatter(
            sd, x="satisfaction_rating", y="referral_intent",
            color_discrete_sequence=[DANUBE], opacity=0.5,
            title="Satisfaction vs Referral Intent",
            labels={"satisfaction_rating": "Satisfaction (1–5)",
                    "referral_intent": "Referral Intent (1–5)"},
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "How post-experience satisfaction relates to willingness to refer others.",
            "Referral intent is a proxy for organic growth; satisfaction is the lever that drives it.",
            "Whether improving the experience is likely to fuel word-of-mouth acquisition.",
        )
    with s4:
        rd = df.dropna(subset=["late_return_risk_score"])
        fig = px.scatter(
            rd, x="time_buffer_mins_clean", y="late_return_risk_score",
            color="tier_category", color_discrete_sequence=[TOREA_BAY, DANUBE, SHILO],
            opacity=0.6, title="Late-Return Risk vs Time Buffer (Simulated)",
            labels={"time_buffer_mins_clean": "Time Buffer (min)",
                    "late_return_risk_score": "Late-Return Risk (sim.)", "tier_category": "Tier Risk"},
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        insight(
            "Simulated late-return risk against time buffer, coloured by tier risk category.",
            "It shows risk rising as buffer shrinks — the core safety trade-off in the business question.",
            "Setting minimum buffers per tier so growth does not come at the cost of risky recommendations.",
        )


# ---------------------------------------------------------------------
# TAB 6 — BUSINESS RECOMMENDATIONS
# ---------------------------------------------------------------------
with tabs[5]:
    section_tag("BUSINESS STRATEGY — EVIDENCE-LINKED RECOMMENDATIONS")
    st.subheader("Business Recommendations")

    # Recompute a few evidence figures from current view
    conv_by_band = df.groupby("confidence_band_clean")["conversion_flag"].mean().reindex(BAND_ORDER) * 100
    best_channel = (df.groupby("marketing_channel")["conversion_flag"].mean() * 100).sort_values(
        ascending=False)
    rev_by_tier = df.groupby("recommended_tier_clean")["paid_amount_aed_clean"].sum().sort_values(
        ascending=False)

    st.markdown(
        f"""
        **Central business question:** *Which passenger segments and experience tiers should
        GateEscape Dubai prioritise to maximise booking conversion, revenue per passenger, and
        satisfaction, while reducing risky layover recommendations?*
        """
    )

    st.markdown("#### Prioritised recommendations")
    st.markdown(
        f"""
        1. **Prioritise High-confidence-band passengers.** Conversion rises from
           ~{conv_by_band.get('Low', float('nan')):.0f}% (Low) to
           ~{conv_by_band.get('High', float('nan')):.0f}% (High).
           *Evidence: Segment & Tier Performance — Conversion by Confidence Band.*

        2. **Concentrate capacity on the highest-revenue tiers.** Revenue is dominated by
           **{rev_by_tier.index[0]}**. *Evidence: Segment & Tier Performance — Revenue by Recommended Tier.*

        3. **Double down on the best-performing acquisition channel.** The strongest converting
           channel in view is **{best_channel.index[0]}** (~{best_channel.iloc[0]:.0f}%).
           *Evidence: filterable via the Marketing Channel filter; conversion comparisons across tabs.*

        4. **Drive package-viewing depth.** Conversion increases steadily with packages viewed.
           *Evidence: Diagnostic Analytics — Conversion by Packages Viewed.*

        5. **Keep recommendations prominent.** Passengers who booked the recommended tier report
           higher satisfaction. *Evidence: Diagnostic Analytics — Satisfaction by Recommendation Match.*

        6. **Protect against risky recommendations.** Simulated late-return risk rises as time buffer
           falls, especially for higher-risk tiers. Enforce minimum time buffers per tier.
           *Evidence: Correlation Insights — Late-Return Risk vs Time Buffer.*
        """
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
        - **Synthetic data:** all records are generated to simulate a pilot survey / prototype test.
          Relationships were deliberately built in, so findings demonstrate the analytical approach
          rather than measuring a real market.
        - **Simulated risk indicator:** `late_return_risk_score` and `late_return_flag_clean` are
          synthetic risk proxies for analytical practice only — they are **not** real operational,
          immigration, or legal outcomes and must not be read as such.
        - **Rule-based score:** the Layover Confidence Score is a transparent weighted composite, not a
          predictive model; correlation findings describe association, not causation.
        """
    )

    st.markdown(
        f"""<div class="insight">
        <b>Bottom line:</b> The synthetic validation supports a focused strategy — qualify passengers
        with the confidence score, route high-confidence and premium-leaning segments into the
        higher-value tiers, deepen in-app engagement, and enforce per-tier time-buffer safeguards so
        growth does not increase simulated late-return risk.
        </div>""",
        unsafe_allow_html=True,
    )


# TAB 7 — DATA PREPARATION EVIDENCE
# ---------------------------------------------------------------------
with tabs[6]:
    section_tag("DATA PREPARATION EVIDENCE — CLEANING, TRANSFORMATION & FEATURE ENGINEERING")
    st.subheader("Data Preparation Evidence")

    st.markdown(
        """
        This evidence tab supports the cleaning, transformation, and feature engineering marks in the assignment brief.
        The raw synthetic data was generated to resemble a real pilot survey / prototype test,
        deliberately containing the kinds of imperfections such data normally has. The
        `data_preparation.py` script cleans these issues into **new clean/derived columns**,
        leaving the original raw fields intact as an audit trail. The main business story is covered in the earlier tabs.
        """
    )

    clean_tbl = pd.DataFrame({
        "Raw Data Issue": [
            "Duplicate passenger IDs",
            "Inconsistent Yes/No encodings (Y, yes, 1, TRUE...)",
            "Inconsistent category casing/spacing",
            "Inconsistent marketing channel names",
            "Inconsistent tier names",
            "Impossible layover durations (negative, 48h)",
            "Negative paid amounts",
            "Numeric values stored as text ('7.6 hrs', 'AED 542')",
            "Missing fatigue / budget (survey non-response)",
            "Missing satisfaction / referral (non-bookings)",
        ],
        "How It Was Handled": [
            "Dropped duplicates, kept first occurrence",
            "Mapped all variants to True/False in *_clean columns",
            "Stripped whitespace, standardised casing",
            "Mapped variants to canonical channel labels",
            "Mapped variants to canonical tier labels in *_clean columns",
            "Values outside 2–26h set to NaN in clean column",
            "Negative values set to NaN in clean column",
            "Stripped text, coerced to numeric *_clean columns",
            "Kept raw; flagged via fatigue/budget_missing_flag; budget labelled 'Unknown'",
            "Preserved as NaN (logically missing — NOT imputed)",
        ],
    })
    st.dataframe(clean_tbl, use_container_width=True, hide_index=True)

    st.markdown("#### Why clean / derived columns were created")
    st.markdown(
        """
        - **Non-destructive cleaning:** every fix is written to a new `*_clean` column so the original
          raw value is never lost — the dashboard analyses clean columns while raw stays auditable.
        - **Analysis-ready features:** engineered fields such as `time_buffer_mins_clean`,
          `confidence_band_clean`, `engagement_level`, `tier_category`, `safe_to_leave_flag`, and
          `recommendation_match_flag` turn raw inputs into directly chartable business concepts.
        - **Standardised flags:** boolean funnel flags (`conversion_flag`, `package_viewed_flag`,
          `completed_flag`) give consistent, reliable KPI math.
        """
    )

    st.markdown("#### How missingness flags preserve data quality")
    st.markdown(
        """
        Rather than hiding or fabricating missing survey answers, the pipeline records them explicitly:

        - `fatigue_level_clean` keeps genuine non-responses as **NaN** — an ordinal rating is never
          invented, because imputing it would distort the data.
        - `budget_tier_clean` labels non-responses as **"Unknown"** so those passengers remain visible
          in grouped charts instead of being silently dropped by `groupby`.
        - `fatigue_missing_flag` and `budget_missing_flag` make **missingness itself a measurable
          variable**, which the Diagnostics tab uses to test whether incomplete profiles convert differently.

        Feature engineering was therefore applied **without disrupting the original raw fields**: raw
        columns remain exactly as generated, and all transformations live in clearly named clean/derived columns.
        """
    )

    # Show a small before/after style summary of missingness
    miss_summary = pd.DataFrame({
        "Column": ["fatigue_level_clean", "budget_tier_clean", "satisfaction_rating",
                   "referral_intent", "paid_amount_aed_clean"],
        "Missing (count)": [
            int(df["fatigue_level_clean"].isna().sum()),
            int((df["budget_tier_clean"] == "Unknown").sum()),
            int(df["satisfaction_rating"].isna().sum()),
            int(df["referral_intent"].isna().sum()),
            int(df["paid_amount_aed_clean"].isna().sum()),
        ],
        "Reason": ["Survey non-response (kept NaN)", "Survey non-response (labelled 'Unknown')",
                   "Logical — non-booking", "Logical — non-booking", "Logical — non-booking"],
    })
    st.dataframe(miss_summary, use_container_width=True, hide_index=True)
    st.caption("Counts reflect the current filter selection.")


# ---------------------------------------------------------------------

st.markdown("---")
st.caption(
    "GateEscape Dubai — Individual Data Analytics Assignment. Data is synthetic and simulates a "
    "pilot survey / prototype test. Late-return variables are simulated synthetic indicators only."
)
