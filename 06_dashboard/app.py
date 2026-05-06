"""
Phase 5: Party-aware interactive dashboard.

Usage:
    streamlit run 06_dashboard/app.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, FIGURES_DIR, CONSTITUENCY_NAME, PARTY_NAMES

ADVANCE_FORMS = {"5_16", "5_16_party", "5_17", "5_17_party"}

st.set_page_config(
    page_title=f"Election 2026 — {CONSTITUENCY_NAME}",
    page_icon="🗳️", layout="wide",
)


def build_party_long(df: pd.DataFrame) -> pd.DataFrame:
    cand_re = re.compile(r"^candidate_(\d+)_votes$")
    cand_nums = sorted({int(m.group(1)) for c in df.columns
                        if (m := cand_re.match(c))})
    rows = []
    for _, row in df.iterrows():
        ft = str(row.get("form_type", ""))
        is_pl = ft.endswith("_party")
        for n in cand_nums:
            v_col = f"candidate_{n}_votes"
            p_col = f"candidate_{n}_party"
            if v_col not in row.index:
                continue
            v = row[v_col]
            if pd.isna(v) or v == 0:
                continue
            if is_pl:
                party = PARTY_NAMES.get(n, f"พรรคเบอร์ {n}")
            else:
                party = (row[p_col] if p_col in row.index and pd.notna(row[p_col]) else "")
                party = str(party).strip() or f"ผู้สมัครเบอร์ {n}"
            rows.append({
                "form_type": ft, "is_party_list": is_pl,
                "is_advance": ft in ADVANCE_FORMS,
                "station_id": row.get("station_id"),
                "candidate_number": n, "party": party,
                "votes": float(v),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load():
    path = CLEANED_DIR / "election_results_cleaned.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df, build_party_long(df)


df, long = load()
if df is None:
    st.error("No cleaned data found. Run the OCR pipeline first.")
    st.stop()

# ── Sidebar ──
st.sidebar.title("Filters")
form_options = sorted(df["form_type"].dropna().unique().tolist())
form_sel = st.sidebar.multiselect("Form type", form_options, default=form_options)
ballot_kind = st.sidebar.radio("Ballot kind",
                               ["All", "Constituency", "Party-list"], index=0)

mask = df["form_type"].isin(form_sel)
df_f = df[mask]
long_f = long[long["form_type"].isin(form_sel)] if not long.empty else long
if ballot_kind == "Constituency":
    long_f = long_f[~long_f["is_party_list"]]
elif ballot_kind == "Party-list":
    long_f = long_f[long_f["is_party_list"]]

st.title(f"🇹🇭 Thailand Election 2026 — {CONSTITUENCY_NAME}")

tab_overview, tab_party, tab_station, tab_compare, tab_anom, tab_data = st.tabs(
    ["Overview", "Party Performance", "Per-Station",
     "Advance vs Day", "Anomalies", "Data"])

# ── Overview ──
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    total_votes = long_f["votes"].sum() if not long_f.empty else 0
    n_stations = df_f["station_id"].nunique() if "station_id" in df_f.columns else 0
    n_parties = long_f["party"].nunique() if not long_f.empty else 0
    if "invalid_ballot_ratio" in df_f.columns:
        avg_inv = df_f["invalid_ballot_ratio"].mean() * 100
    else:
        avg_inv = np.nan
    c1.metric("Total votes", f"{total_votes:,.0f}")
    c2.metric("Polling stations", f"{n_stations}")
    c3.metric("Distinct parties", f"{n_parties}")
    c4.metric("Avg invalid %", f"{avg_inv:.2f}%" if pd.notna(avg_inv) else "N/A")

    st.markdown("---")
    if "total_ballots" in df_f.columns:
        b1, b2, b3 = st.columns(3)
        b1.metric("Total ballots", f"{df_f['total_ballots'].sum():,.0f}")
        if "good_ballots" in df_f.columns:
            b2.metric("Valid ballots", f"{df_f['good_ballots'].sum():,.0f}")
        if "bad_ballots" in df_f.columns:
            b3.metric("Invalid ballots", f"{df_f['bad_ballots'].sum():,.0f}")

# ── Party Performance ──
with tab_party:
    st.subheader("Party totals & vote share")
    if long_f.empty:
        st.info("No party-level data after filters.")
    else:
        agg = (long_f.groupby("party", as_index=False)
                     .agg(votes=("votes", "sum"),
                          stations=("station_id", "nunique")))
        total = agg["votes"].sum()
        agg["share_pct"] = agg["votes"] / total * 100 if total else 0
        agg = agg.sort_values("votes", ascending=False)
        top_n = st.slider("Top N parties", 5, max(5, min(50, len(agg))),
                          min(15, len(agg)))
        top = agg.head(top_n)
        fig = px.bar(top.iloc[::-1], x="votes", y="party", orientation="h",
                     color="share_pct", color_continuous_scale="Blues",
                     labels={"votes": "Votes", "share_pct": "Share %"},
                     title=f"Top {top_n} parties")
        fig.update_layout(height=max(400, 25 * top_n))
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            shares = agg["votes"] / total if total else agg["votes"]
            hhi = float((shares ** 2).sum())
            enp = 1 / hhi if hhi else float("nan")
            st.metric("Effective Number of Parties", f"{enp:.2f}")
            st.metric("HHI", f"{hhi:.4f}")
        with c2:
            fig_pie = px.pie(top, names="party", values="votes",
                             title="Share among top parties")
            st.plotly_chart(fig_pie, width='stretch')

        st.dataframe(agg.assign(share_pct=agg["share_pct"].round(2)),
                     width='stretch')

# ── Per-Station ──
with tab_station:
    st.subheader("Per-station drill-down")
    if "station_id" not in df_f.columns or df_f["station_id"].isna().all():
        st.info("No station_id available.")
    else:
        stations = sorted(df_f["station_id"].dropna().unique())
        sel = st.selectbox("Select station", stations)
        st_long = long_f[long_f["station_id"] == sel] if not long_f.empty else long_f
        st_df = df_f[df_f["station_id"] == sel]

        c1, c2, c3 = st.columns(3)
        c1.metric("Votes (filter)", f"{st_long['votes'].sum():,.0f}"
                  if not st_long.empty else "0")
        if "total_ballots" in st_df.columns:
            c2.metric("Total ballots", f"{st_df['total_ballots'].sum():,.0f}")
        if "invalid_ballot_ratio" in st_df.columns:
            c3.metric("Avg invalid ratio",
                      f"{st_df['invalid_ballot_ratio'].mean()*100:.2f}%")

        if not st_long.empty:
            agg = (st_long.groupby("party", as_index=False)["votes"].sum()
                          .sort_values("votes", ascending=False))
            fig = px.bar(agg.head(20).iloc[::-1], x="votes", y="party",
                         orientation="h", color_discrete_sequence=["#10b981"],
                         title=f"Station {sel} — votes by party")
            st.plotly_chart(fig, width='stretch')
        st.dataframe(st_df, width='stretch')

# ── Advance vs Day ──
with tab_compare:
    st.subheader("Advance voting vs election day")
    if long.empty:
        st.info("No data.")
    else:
        kind_long = (long[~long["is_party_list"]] if ballot_kind == "Constituency"
                     else long[long["is_party_list"]] if ballot_kind == "Party-list"
                     else long)
        agg = (kind_long.groupby(["party", "is_advance"], as_index=False)["votes"]
                        .sum())
        agg["bucket"] = np.where(agg["is_advance"], "Advance", "Election Day")
        wide = agg.pivot_table(index="party", columns="bucket",
                               values="votes", fill_value=0).reset_index()
        if "Advance" in wide.columns and "Election Day" in wide.columns:
            wide["total"] = wide["Advance"] + wide["Election Day"]
            wide = wide.sort_values("total", ascending=False).head(15)
            fig = px.bar(wide, x="party", y=["Election Day", "Advance"],
                         barmode="group",
                         color_discrete_sequence=["#6366f1", "#f59e0b"],
                         title="Top 15 parties — advance vs day")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, width='stretch')
            st.dataframe(wide, width='stretch')
        else:
            st.info("Need both advance and election-day rows for comparison.")

# ── Anomalies ──
with tab_anom:
    st.subheader("Anomaly detection (robust MAD-z)")
    metric = st.selectbox("Metric",
                          [c for c in ["total_ballots", "invalid_ballot_ratio",
                                       "turnout_valid_ratio"]
                           if c in df_f.columns])
    if metric:
        x = df_f[metric].fillna(0)
        med = x.median()
        mad = (x - med).abs().median()
        z = 0.6745 * (x - med) / mad if mad else pd.Series(np.zeros(len(x)),
                                                            index=x.index)
        flagged = df_f.assign(mad_z=z).loc[z.abs() > 3.5]
        st.write(f"Flagged: **{len(flagged)}** stations (|MAD-z| > 3.5)")
        st.dataframe(flagged[["station_id", "form_type", metric]]
                     .assign(mad_z=z.loc[flagged.index].round(2))
                     if not flagged.empty else flagged, width='stretch')
        fig = px.histogram(df_f, x=metric, nbins=30,
                           title=f"{metric} distribution")
        st.plotly_chart(fig, width='stretch')

# ── Data ──
with tab_data:
    st.subheader("Cleaned dataset")
    st.dataframe(df_f, width='stretch')
    st.download_button("Download CSV",
                       df_f.to_csv(index=False).encode("utf-8-sig"),
                       file_name="election_results_filtered.csv",
                       mime="text/csv")
    if not long_f.empty:
        st.subheader("Long-format party data")
        st.dataframe(long_f, width='stretch')
        st.download_button("Download long-format CSV",
                           long_f.to_csv(index=False).encode("utf-8-sig"),
                           file_name="party_long.csv", mime="text/csv")

    saved_figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if saved_figs:
        st.subheader("Saved analysis figures")
        for fp in saved_figs:
            st.image(str(fp), caption=fp.stem.replace("_", " ").title())

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Constituency:** {CONSTITUENCY_NAME}\n\n"
    f"**Source:** ECT (Election Commission of Thailand)"
)
