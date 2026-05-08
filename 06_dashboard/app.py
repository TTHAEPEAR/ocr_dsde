"""
Quality-aware election dashboard.

Run:
    streamlit run 06_dashboard/app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, FIGURES_DIR, OCR_RAW_DIR, CONSTITUENCY_NAME, PARTY_NAMES

ADVANCE_FORMS = {"5_16", "5_16_party", "5_17", "5_17_party"}

st.set_page_config(page_title=f"Election Dashboard - {CONSTITUENCY_NAME}", layout="wide")


def _to_bool(value, default=False):
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _series(df: pd.DataFrame, column: str, default="") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


@st.cache_data
def load_data():
    candidates = [
        CLEANED_DIR / "election_results_cleaned.csv",
        OCR_RAW_DIR / "raw_all_forms_split.csv",
        OCR_RAW_DIR / "raw_election_split.csv",
        OCR_RAW_DIR / "raw_election_split_checkpoint.csv",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        return None, None, None
    df = pd.read_csv(source)
    if "polling_unit_id" not in df.columns:
        df["polling_unit_id"] = df.get("source_file", pd.Series(range(len(df)))).astype(str).str.replace(
            r"\.pdf$", "", regex=True
        )
    if "ballot_record_id" not in df.columns:
        kind_key = _series(df, "ballot_kind", None).fillna(_series(df, "form_type", "unknown"))
        df["ballot_record_id"] = df["polling_unit_id"].astype(str) + "__" + kind_key.astype(str)
    if "ballot_kind" not in df.columns:
        form_type = _series(df, "form_type", "")
        df["ballot_kind"] = np.where(form_type.astype(str).str.endswith("_party"), "party_list", "constituency")
    for col in [
        "good_ballots",
        "bad_ballots",
        "no_vote_ballots",
        "total_ballots",
        "votes_sum",
        "total_votes_sum",
        "station_id",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["ocr_success", "needs_review", "vote_sum_match", "ballot_sum_match", "summary_votes_match"]:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _to_bool(v) if pd.notna(v) else np.nan)
    df["ocr_success"] = df.get("ocr_success", True)
    df["needs_review"] = df.get("needs_review", False)
    df["vote_sum_match"] = df.get("vote_sum_match", df.get("votes_sum", 0).eq(df.get("good_ballots", -1)))
    df["ballot_sum_match"] = df.get(
        "ballot_sum_match",
        (df.get("good_ballots", 0) + df.get("bad_ballots", 0) + df.get("no_vote_ballots", 0)).eq(df.get("total_ballots", -1)),
    )
    df["is_confirmed"] = (
        df["ocr_success"].fillna(False).astype(bool)
        & ~df["needs_review"].fillna(True).astype(bool)
        & df["vote_sum_match"].fillna(False).astype(bool)
        & df["ballot_sum_match"].fillna(False).astype(bool)
    )
    long = build_party_long(df)
    return df, long, source


def build_party_long(df):
    cand_re = re.compile(r"^candidate_(\d+)_votes$")
    cand_nums = sorted({int(m.group(1)) for c in df.columns if (m := cand_re.match(c))})
    rows = []
    for _, row in df.iterrows():
        kind = str(row.get("ballot_kind", ""))
        is_pl = kind == "party_list" or str(row.get("form_type", "")).endswith("_party")
        good = row.get("good_ballots", np.nan)
        for n in cand_nums:
            v = pd.to_numeric(pd.Series([row.get(f"candidate_{n}_votes")]), errors="coerce").iloc[0]
            if pd.isna(v) or v == 0:
                continue
            if is_pl:
                party = PARTY_NAMES.get(n, f"party_no_{n}")
            else:
                party = str(row.get(f"candidate_{n}_party", "") or "").strip() or f"candidate_no_{n}"
            rows.append(
                {
                    "polling_unit_id": row.get("polling_unit_id"),
                    "ballot_record_id": row.get("ballot_record_id"),
                    "station_id": row.get("station_id"),
                    "ballot_kind": kind,
                    "is_party_list": is_pl,
                    "form_type": row.get("form_type", ""),
                    "is_advance": row.get("form_type", "") in ADVANCE_FORMS,
                    "candidate_number": n,
                    "party": party,
                    "votes": float(v),
                    "vote_share": float(v / good) if pd.notna(good) and good > 0 else np.nan,
                    "is_confirmed": bool(row.get("is_confirmed", False)),
                    "needs_review": bool(row.get("needs_review", False)),
                }
            )
    return pd.DataFrame(rows)


df, long, source = load_data()
if df is None:
    st.error("No election data found. Run OCR or cleaning first.")
    st.stop()

st.title(f"Thailand Election 2026 - {CONSTITUENCY_NAME}")
st.caption(f"Source: {source}")

with st.sidebar:
    st.header("Filters")
    kind_options = sorted(df["ballot_kind"].dropna().unique().tolist())
    kind_sel = st.multiselect("Ballot kind", kind_options, default=kind_options)
    quality_mode = st.radio("Quality slice", ["Confirmed only", "All rows", "Needs review only"], index=0)
    station_options = sorted(df["polling_unit_id"].dropna().astype(str).unique().tolist())
    station_query = st.text_input("Search polling unit")
    top_n = st.slider("Top parties", 5, 40, 15)

mask = df["ballot_kind"].isin(kind_sel)
if quality_mode == "Confirmed only":
    mask &= df["is_confirmed"]
elif quality_mode == "Needs review only":
    mask &= ~df["is_confirmed"]
if station_query:
    mask &= df["polling_unit_id"].astype(str).str.contains(station_query, case=False, na=False)

df_f = df[mask].copy()
long_f = long[long["ballot_record_id"].isin(df_f["ballot_record_id"])] if not long.empty else long

tab_quality, tab_overview, tab_party, tab_station, tab_anomaly, tab_data = st.tabs(
    ["Quality", "Overview", "Party Performance", "Station Drilldown", "Anomalies", "Data"]
)

with tab_quality:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Confirmed", f"{int(df['is_confirmed'].sum()):,}")
    c3.metric("Needs review", f"{int((~df['is_confirmed']).sum()):,}")
    c4.metric("Polling units", f"{df['polling_unit_id'].nunique():,}")

    q = pd.DataFrame(
        {
            "status": ["confirmed", "needs_review"],
            "rows": [int(df["is_confirmed"].sum()), int((~df["is_confirmed"]).sum())],
        }
    )
    st.plotly_chart(px.bar(q, x="status", y="rows", title="Quality gate"), width="stretch")

    flag_cols = [
        c
        for c in [
            "source_file",
            "ballot_kind",
            "page_range",
            "station_id",
            "good_ballots",
            "votes_sum",
            "total_votes_sum",
            "vote_sum_match",
            "vote_sum_match_good_ballots",
            "vote_sum_match_total_votes",
            "summary_votes_match",
            "ballot_sum_match",
            "ocr_success",
            "needs_review",
        ]
        if c in df.columns
    ]
    st.subheader("Rows requiring review")
    st.dataframe(df.loc[~df["is_confirmed"], flag_cols], width="stretch")

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered rows", f"{len(df_f):,}")
    c2.metric("Votes", f"{long_f['votes'].sum():,.0f}" if not long_f.empty else "0")
    c3.metric("Total ballots", f"{df_f.get('total_ballots', pd.Series(dtype=float)).sum():,.0f}")
    invalid = df_f.get("bad_ballots", pd.Series(dtype=float)).sum()
    total = df_f.get("total_ballots", pd.Series(dtype=float)).sum()
    c4.metric("Invalid share", f"{invalid / total * 100:.2f}%" if total else "N/A")

    if "total_ballots" in df_f.columns and not df_f.empty:
        st.plotly_chart(px.histogram(df_f, x="total_ballots", color="ballot_kind", nbins=30), width="stretch")

with tab_party:
    if long_f.empty:
        st.info("No party rows after filters.")
    else:
        agg = (
            long_f.groupby(["ballot_kind", "party"], as_index=False)
            .agg(votes=("votes", "sum"), stations=("polling_unit_id", "nunique"), mean_share=("vote_share", "mean"))
            .sort_values(["ballot_kind", "votes"], ascending=[True, False])
        )
        agg["share_pct"] = agg["votes"] / agg.groupby("ballot_kind")["votes"].transform("sum") * 100
        st.dataframe(agg, width="stretch")
        for kind, sub in agg.groupby("ballot_kind"):
            top = sub.head(top_n)
            fig = px.bar(
                top.iloc[::-1],
                x="share_pct",
                y="party",
                orientation="h",
                title=f"Top {top_n} - {kind}",
                hover_data=["votes", "stations", "mean_share"],
            )
            fig.update_layout(height=max(420, 28 * len(top)))
            st.plotly_chart(fig, width="stretch")

with tab_station:
    station_sel = st.selectbox("Polling unit", station_options, index=0)
    st_df = df[df["polling_unit_id"].astype(str) == station_sel]
    st_long = long[long["polling_unit_id"].astype(str) == station_sel] if not long.empty else long
    st.dataframe(st_df, width="stretch")
    if not st_long.empty:
        for kind, sub in st_long.groupby("ballot_kind"):
            agg = sub.groupby("party", as_index=False)["votes"].sum().sort_values("votes", ascending=False)
            st.plotly_chart(px.bar(agg.head(25).iloc[::-1], x="votes", y="party", orientation="h", title=kind), width="stretch")

with tab_anomaly:
    anomaly_path = FIGURES_DIR / "anomaly_records.csv"
    if anomaly_path.exists():
        anomalies = pd.read_csv(anomaly_path)
        st.dataframe(anomalies, width="stretch")
    else:
        st.info("Run `python 05_analysis/analysis.py` to generate anomaly records.")

    metric = st.selectbox("Quick MAD metric", [c for c in ["total_ballots", "good_ballots", "bad_ballots", "no_vote_ballots"] if c in df_f.columns])
    if metric and not df_f.empty:
        x = pd.to_numeric(df_f[metric], errors="coerce")
        med = x.median()
        mad = (x - med).abs().median()
        z = 0.6745 * (x - med) / mad if mad else pd.Series(np.zeros(len(x)), index=x.index)
        quick = df_f.assign(mad_z=z).loc[z.abs() > 3.5]
        st.write(f"Flagged rows: {len(quick)}")
        st.dataframe(quick, width="stretch")

with tab_data:
    st.subheader("Filtered records")
    st.dataframe(df_f, width="stretch")
    st.download_button(
        "Download filtered records",
        df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_election_records.csv",
        mime="text/csv",
    )
    if not long_f.empty:
        st.subheader("Long party rows")
        st.dataframe(long_f, width="stretch")
        st.download_button(
            "Download long party rows",
            long_f.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_party_long.csv",
            mime="text/csv",
        )
