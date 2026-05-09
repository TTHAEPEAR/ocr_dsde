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
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLEANED_DIR,
    FIGURES_DIR,
    OCR_RAW_DIR,
    REFERENCE_DIR,
    CONSTITUENCY_NAME,
    PARTY_NAMES,
    CONSTITUENCY_CANDIDATE_PARTIES,
)

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


def locality_from_unit(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"_?หน_วย.*$", "", text)
    text = re.sub(r"_?หน่วย.*$", "", text)
    text = re.sub(r"_?202\d{5}T\d+Z.*$", "", text)
    text = text.strip("_ ")
    return text or "unknown_locality"


def unit_number(value: object, fallback: object = np.nan) -> float:
    text = str(value or "")
    patterns = [
        r"หน_วย(?:เล_อกต_ง)?ท_?(\d+)",
        r"หน่วย(?:เลือกตั้ง)?ที่\s*(\d+)",
        r"unit[_ ]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    parsed = pd.to_numeric(pd.Series([fallback]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def _format_number(value: object) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return ""
    return str(int(parsed)) if float(parsed).is_integer() else f"{float(parsed):g}"


def _clean_ocr_safe_name(value: object) -> str:
    text = str(value or "").replace(".pdf", "")
    text = re.sub(r"_?202\d{5}T\d+Z_\d+_\d+_?", " ", text)
    replacements = {
        "ทม_กำแพงเพชร": "ทม.กำแพงเพชร",
        "ทม_หนองปล_ง": "ทม.หนองปลิง",
        "ทต_น_คมท_งโพธ_ทะเล": "ทต.นิคมทุ่งโพธิ์ทะเล",
        "ทต_เทพนคร": "ทต.เทพนคร",
        "ทต_นครช_ม": "ทต.นครชุม",
        "ทต_คลองแม_ลาย": "ทต.คลองแม่ลาย",
        "ตำบลนครช_ม": "ตำบลนครชุม",
        "ตำบลอ_างทอง": "ตำบลอ่างทอง",
        "ตำบลคณฑ_": "ตำบลคณฑี",
        "ตำบลท_าข_นราม": "ตำบลท่าขุนราม",
        "ตำบลคลองแม_ลาย": "ตำบลคลองแม่ลาย",
        "ตำบลธำมรงค_": "ตำบลธำมรงค์",
        "อบต_ไตรตร_งษ_": "อบต.ไตรตรึงษ์",
        "อบต_สระแก_ว": "อบต.สระแก้ว",
        "หน_วยเล_อกต_งท_": "หน่วยเลือกตั้งที่ ",
        "หน_วยท_": "หน่วยที่ ",
        "ช_ด_": "ชุด ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_location_enrichment() -> pd.DataFrame:
    candidates = [REFERENCE_DIR / "polling_unit_locations.csv", REFERENCE_DIR / "polling_unit_locations_draft.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame()
    loc = pd.read_csv(path)
    keep = [
        c
        for c in [
            "polling_unit_id",
            "path_local_government",
            "path_local_government_name",
            "path_unit_number",
            "official_subdistrict",
            "draft_subdistrict_or_municipality",
            "draft_municipality",
            "draft_moo",
            "draft_unit_number",
            "pdf_path",
        ]
        if c in loc.columns
    ]
    return loc[keep].drop_duplicates("polling_unit_id") if "polling_unit_id" in keep else pd.DataFrame()


def add_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "polling_unit_id" in out.columns and "area_label" not in out.columns:
        loc = load_location_enrichment()
        if not loc.empty:
            out = out.merge(loc, on="polling_unit_id", how="left", suffixes=("", "_loc"))
    source = out["source_file"] if "source_file" in out.columns else _series(out, "polling_unit_id", "")
    out["file_label"] = source.map(_clean_ocr_safe_name)

    area_candidates = [
        "official_subdistrict",
        "path_local_government",
        "draft_municipality",
        "draft_subdistrict_or_municipality",
        "locality",
    ]
    area = pd.Series("", index=out.index, dtype=object)
    for col in area_candidates:
        if col in out.columns:
            values = out[col].fillna("").astype(str).str.strip()
            area = area.mask(area.astype(str).str.strip().eq(""), values)
    area = area.mask(area.astype(str).str.strip().eq(""), out["file_label"].map(locality_from_unit))
    out["area_label"] = area.fillna("").replace("", "ไม่ทราบพื้นที่")

    unit = pd.Series("", index=out.index, dtype=object)
    for col in ["station_id", "path_unit_number", "draft_unit_number", "unit_number"]:
        if col in out.columns:
            values = out[col].map(_format_number)
            unit = unit.mask(unit.astype(str).str.strip().eq(""), values)
    if unit.astype(str).str.strip().eq("").any() and "polling_unit_id" in out.columns:
        parsed = out["polling_unit_id"].map(lambda v: _format_number(unit_number(v)))
        unit = unit.mask(unit.astype(str).str.strip().eq(""), parsed)
    out["unit_no"] = unit

    outside = source.astype(str).str.contains("นอกเขต", na=False)
    set_no = source.astype(str).str.extract(r"ช_ด_(\d+)", expand=False).fillna(unit)
    out["unit_label"] = np.where(
        outside,
        "นอกเขต ชุด " + set_no.astype(str).replace("", "ไม่ทราบ"),
        np.where(
            out["unit_no"].astype(str).str.strip().ne(""),
            out["area_label"].astype(str) + " หน่วย " + out["unit_no"].astype(str),
            out["area_label"].astype(str),
        ),
    )
    out["trace_id"] = _series(out, "polling_unit_id", "")
    return out


def display_table(frame: pd.DataFrame, preferred: list[str] | None = None) -> pd.DataFrame:
    view = add_display_columns(frame)
    front = preferred or ["unit_label", "area_label", "ballot_kind"]
    front = [c for c in front if c in view.columns]
    rest = [c for c in view.columns if c not in front]
    return view[front + rest]


def constituency_party_lookup(df: pd.DataFrame) -> dict[int, str]:
    if CONSTITUENCY_CANDIDATE_PARTIES:
        return dict(CONSTITUENCY_CANDIDATE_PARTIES)
    lookup: dict[int, str] = {}
    if "ballot_kind" not in df.columns:
        return lookup
    const_df = df[df["ballot_kind"].astype(str).eq("constituency")]
    for col in [c for c in const_df.columns if re.match(r"^candidate_\d+_party$", c)]:
        number = int(re.search(r"candidate_(\d+)_party", col).group(1))
        values = const_df[col].dropna().astype(str).str.strip()
        values = values[values.ne("") & values.ne("-")]
        if values.empty:
            continue
        lookup[number] = values.value_counts().idxmax()
    return lookup


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
    df = add_display_columns(df)
    long = build_party_long(df, constituency_party_lookup(df))
    return df, long, source


def build_party_long(df, constituency_parties=None):
    constituency_parties = constituency_parties or {}
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
                party = constituency_parties.get(n) or str(row.get(f"candidate_{n}_party", "") or "").strip() or f"candidate_no_{n}"
            rows.append(
                {
                    "polling_unit_id": row.get("polling_unit_id"),
                    "ballot_record_id": row.get("ballot_record_id"),
                    "station_id": row.get("station_id"),
                    "unit_label": row.get("unit_label", ""),
                    "area_label": row.get("area_label", ""),
                    "file_label": row.get("file_label", ""),
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


def build_pseudo_spatial(df_in, long_in):
    confirmed_long = long_in[long_in["is_confirmed"]].copy() if not long_in.empty else long_in
    if confirmed_long.empty:
        confirmed_long = long_in.copy()
    if confirmed_long.empty:
        return pd.DataFrame()
    top = confirmed_long.sort_values("votes", ascending=False).groupby(
        ["polling_unit_id", "ballot_kind"], as_index=False
    ).first()
    totals = confirmed_long.groupby(["polling_unit_id", "ballot_kind"], as_index=False).agg(
        total_party_votes=("votes", "sum"), party_count=("party", "nunique")
    )
    out = top.merge(totals, on=["polling_unit_id", "ballot_kind"], how="left")
    out["locality"] = out["polling_unit_id"].map(locality_from_unit)
    out["unit_number"] = out.apply(lambda r: unit_number(r["polling_unit_id"], r.get("station_id")), axis=1)
    out["layout_x"] = out["unit_number"]
    missing_x = out["layout_x"].isna()
    if missing_x.any():
        out.loc[missing_x, "layout_x"] = out[missing_x].groupby("locality").cumcount() + 1
    locality_order = {name: i for i, name in enumerate(sorted(out["locality"].dropna().unique()))}
    out["layout_y"] = out["locality"].map(locality_order).astype(float)
    out["winner_party"] = out["party"]
    out["winner_share"] = out["vote_share"]
    return out


@st.cache_data
def load_location_reference():
    reference_path = REFERENCE_DIR / "polling_unit_locations.csv"
    draft_path = REFERENCE_DIR / "polling_unit_locations_draft.csv"
    path = reference_path if reference_path.exists() else draft_path
    if not path.exists():
        return pd.DataFrame(), None
    loc = pd.read_csv(path)
    for col in ["location_verified", "needs_location_review", "has_markdown"]:
        if col in loc.columns:
            loc[col] = loc[col].map(lambda v: _to_bool(v) if pd.notna(v) else False)
    for col in ["lat", "lon", "draft_moo", "draft_unit_number", "official_moo"]:
        if col in loc.columns:
            loc[col] = pd.to_numeric(loc[col], errors="coerce")
    return loc, path


@st.cache_data
def load_geo_winners():
    geo_path = FIGURES_DIR / "geographic_winner_summary.csv"
    if geo_path.exists():
        geo = pd.read_csv(geo_path)
        for col in ["location_verified", "needs_location_review", "is_confirmed"]:
            if col in geo.columns:
                geo[col] = geo[col].map(lambda v: _to_bool(v) if pd.notna(v) else False)
        for col in ["lat", "lon", "winner_share", "winner_votes", "draft_moo", "draft_unit_number"]:
            if col in geo.columns:
                geo[col] = pd.to_numeric(geo[col], errors="coerce")
        return geo, geo_path

    loc, loc_path = load_location_reference()
    evidence_path = FIGURES_DIR / "evidence_fingerprint_map.csv"
    if loc.empty or not evidence_path.exists():
        return pd.DataFrame(), None
    evidence = pd.read_csv(evidence_path)
    geo = evidence.merge(loc, on="polling_unit_id", how="left")
    return geo, loc_path


@st.cache_data
def load_tambon_summary():
    path = FIGURES_DIR / "tambon_party_summary.csv"
    if not path.exists():
        return pd.DataFrame(), None
    tambon = pd.read_csv(path)
    for col in ["votes", "polling_units", "tambon_vote_share", "tambon_rank", "tambon_lat", "tambon_lon"]:
        if col in tambon.columns:
            tambon[col] = pd.to_numeric(tambon[col], errors="coerce")
    if "tambon_verified" in tambon.columns:
        tambon["tambon_verified"] = tambon["tambon_verified"].map(lambda v: _to_bool(v) if pd.notna(v) else False)
    return tambon, path


def circular_network(edges, max_edges=60):
    if edges.empty:
        return go.Figure()
    top = edges.sort_values("abs_weight", ascending=False).head(max_edges)
    labels = pd.unique(pd.concat([top["source_label"], top["target_label"]], ignore_index=True))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    pos = {label: (np.cos(angle), np.sin(angle)) for label, angle in zip(labels, angles)}

    edge_x, edge_y = [], []
    edge_text = []
    for _, row in top.iterrows():
        x0, y0 = pos[row["source_label"]]
        x1, y1 = pos[row["target_label"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{row['source_label']} - {row['target_label']}: {row['weight']:.3f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="rgba(75,85,99,0.45)"),
            hoverinfo="skip",
        )
    )
    node_x = [pos[label][0] for label in labels]
    node_y = [pos[label][1] for label in labels]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=18, color="#2563eb", line=dict(width=1, color="#0f172a")),
            hovertext=labels,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="Party similarity network",
        showlegend=False,
    )
    return fig


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
    station_label_lookup = (
        df.drop_duplicates("polling_unit_id").set_index("polling_unit_id")["unit_label"].astype(str).to_dict()
        if "unit_label" in df.columns
        else {}
    )
    station_query = st.text_input("Search polling unit / tambon")
    top_n = st.slider("Top parties", 5, 40, 15)

mask = df["ballot_kind"].isin(kind_sel)
if quality_mode == "Confirmed only":
    mask &= df["is_confirmed"]
elif quality_mode == "Needs review only":
    mask &= ~df["is_confirmed"]
if station_query:
    searchable = (
        df["polling_unit_id"].astype(str)
        + " "
        + _series(df, "unit_label", "").astype(str)
        + " "
        + _series(df, "area_label", "").astype(str)
        + " "
        + _series(df, "file_label", "").astype(str)
    )
    mask &= searchable.str.contains(station_query, case=False, na=False)

df_f = df[mask].copy()
long_f = long[long["ballot_record_id"].isin(df_f["ballot_record_id"])] if not long.empty else long

tab_evidence, tab_quality, tab_overview, tab_party, tab_geo, tab_spatial, tab_network, tab_station, tab_anomaly, tab_data = st.tabs(
    [
        "Evidence Map",
        "Quality",
        "Overview",
        "Party Performance",
        "Geo QA / Map",
        "Spatial",
        "Network",
        "Station Drilldown",
        "Anomalies",
        "Data",
    ]
)

with tab_evidence:
    st.subheader("Election evidence fingerprint map")
    st.caption(
        "Each point is one ballot record. Position comes from PCA over party vote-share fingerprints, "
        "color is the winner, size is total ballots, and symbol separates confirmed rows from review rows."
    )
    evidence_path = FIGURES_DIR / "evidence_fingerprint_map.csv"
    if not evidence_path.exists():
        st.info("Run `python 05_analysis\\analysis.py` to generate the evidence map.")
    else:
        evidence = pd.read_csv(evidence_path)
        evidence = add_display_columns(evidence)
        evidence = evidence[evidence["ballot_kind"].isin(kind_sel)]
        if station_query:
            searchable = (
                evidence["polling_unit_id"].astype(str)
                + " "
                + _series(evidence, "unit_label", "").astype(str)
                + " "
                + _series(evidence, "area_label", "").astype(str)
                + " "
                + _series(evidence, "file_label", "").astype(str)
            )
            evidence = evidence[searchable.str.contains(station_query, case=False, na=False)]
        if evidence.empty:
            st.info("No evidence-map rows after filters.")
        else:
            fig = px.scatter(
                evidence,
                x="fingerprint_x",
                y="fingerprint_y",
                color="winner_party",
                symbol="quality_label",
                size="total_ballots",
                facet_col="ballot_kind" if evidence["ballot_kind"].nunique() > 1 else None,
                hover_data=[
                    "unit_label",
                    "area_label",
                    "unit_number",
                    "winner_party",
                    "winner_votes",
                    "winner_share",
                    "good_ballots",
                    "total_ballots",
                    "ocr_confidence",
                    "review_reason",
                    "file_label",
                    "trace_id",
                ],
                title="Vote-share fingerprint space with OCR quality overlay",
            )
            fig.update_traces(marker=dict(line=dict(width=0.8, color="rgba(15,23,42,0.55)")))
            fig.update_layout(height=760, legend_title_text="Winner / quality")
            st.plotly_chart(fig, width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Records on map", f"{len(evidence):,}")
            c2.metric("Needs review on map", f"{int((evidence['quality_label'] == 'needs_review').sum()):,}")
            c3.metric("Winner parties", f"{evidence['winner_party'].nunique():,}")

            st.subheader("Most unusual fingerprints")
            center_x = evidence["fingerprint_x"].median()
            center_y = evidence["fingerprint_y"].median()
            ranked = evidence.assign(
                distance=((evidence["fingerprint_x"] - center_x) ** 2 + (evidence["fingerprint_y"] - center_y) ** 2) ** 0.5
            ).sort_values("distance", ascending=False)
            st.dataframe(
                ranked[
                    [
                        "polling_unit_id",
                        "unit_label",
                        "area_label",
                        "ballot_kind",
                        "winner_party",
                        "winner_share",
                        "quality_label",
                        "review_reason",
                        "distance",
                        "file_label",
                    ]
                ].head(20),
                width="stretch",
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
            "unit_label",
            "area_label",
            "file_label",
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
    st.dataframe(display_table(df.loc[~df["is_confirmed"], flag_cols]), width="stretch")

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

with tab_geo:
    st.subheader("Verified geographic winner map")
    st.caption(
        "Use this for real geography only after `polling_unit_locations.csv` has verified tambon/moo/district fields "
        "and coordinates. OCR-derived draft fields are shown for QA, not treated as official truth."
    )
    loc, loc_source = load_location_reference()
    geo, geo_source = load_geo_winners()
    if loc.empty:
        st.info("Run `python 05_analysis\\build_location_reference.py` to create the location QA table.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Location rows", f"{len(loc):,}")
        c2.metric("Verified", f"{int(loc.get('location_verified', pd.Series(False, index=loc.index)).sum()):,}")
        c3.metric("Needs location review", f"{int(loc.get('needs_location_review', pd.Series(False, index=loc.index)).sum()):,}")
        coord_ready = loc.get("lat", pd.Series(dtype=float)).notna() & loc.get("lon", pd.Series(dtype=float)).notna()
        c4.metric("With lat/lon", f"{int(coord_ready.sum()):,}")
        st.caption(f"Location source: {loc_source}")

        st.subheader("Location QA table")
        qa_cols = [
            c
            for c in [
                "unit_label",
                "area_label",
                "polling_unit_id",
                "draft_province",
                "draft_district",
                "draft_subdistrict_or_municipality",
                "draft_municipality",
                "draft_moo",
                "draft_unit_number",
                "official_province",
                "official_district",
                "official_subdistrict",
                "official_municipality",
                "official_moo",
                "lat",
                "lon",
                "location_verified",
                "needs_location_review",
                "location_review_reason",
                "location_confidence",
                "verification_source",
                "verification_note",
                "pdf_path",
            ]
            if c in loc.columns
        ]
        review_only = st.toggle("Show only unverified / needs review locations", value=True)
        loc_view = loc.copy()
        if review_only:
            loc_view = loc_view[
                ~loc_view.get("location_verified", pd.Series(False, index=loc_view.index)).astype(bool)
                | loc_view.get("needs_location_review", pd.Series(False, index=loc_view.index)).astype(bool)
            ]
        st.dataframe(display_table(loc_view[qa_cols]), width="stretch")

        if geo.empty:
            st.info("Run `python 05_analysis\\analysis.py` and `python 05_analysis\\build_location_reference.py` to join winners with locations.")
        else:
            tambon_summary, tambon_source = load_tambon_summary()
            if not tambon_summary.empty:
                st.subheader("Tambon-level winner clusters")
                tambon_view = tambon_summary[tambon_summary["ballot_kind"].isin(kind_sel)].copy()
                winners = tambon_view[tambon_view["is_tambon_winner"] == True].copy() if "is_tambon_winner" in tambon_view.columns else pd.DataFrame()
                if not winners.empty:
                    winners["share_pct"] = winners["tambon_vote_share"] * 100
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tambons in summary", f"{winners['tambon_label'].nunique():,}")
                    c2.metric("Winner parties", f"{winners['party'].nunique():,}")
                    c3.metric("Source", str(tambon_source.name if tambon_source else "N/A"))
                    fig = px.scatter(
                        winners,
                        x="tambon_label",
                        y="ballot_kind",
                        color="party",
                        size="share_pct",
                        hover_data=[
                            "official_district",
                            "tambon_label",
                            "ballot_kind",
                            "party",
                            "votes",
                            "tambon_vote_share",
                            "polling_units",
                            "tambon_verified",
                        ],
                        title="Tambon winners by ballot kind",
                    )
                    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(15,23,42,0.6)")))
                    fig.update_layout(height=430, xaxis_title="Tambon", yaxis_title="Ballot kind")
                    st.plotly_chart(fig, width="stretch")

                    map_ready_tambon = winners[
                        winners.get("tambon_verified", False).astype(bool)
                        & winners["tambon_lat"].notna()
                        & winners["tambon_lon"].notna()
                    ].copy() if {"tambon_lat", "tambon_lon"}.issubset(winners.columns) else pd.DataFrame()
                    if not map_ready_tambon.empty:
                        kind_offset = map_ready_tambon["ballot_kind"].map(
                            {"constituency": -0.006, "party_list": 0.006}
                        ).fillna(0)
                        map_ready_tambon["map_lat"] = map_ready_tambon["tambon_lat"] + kind_offset
                        map_ready_tambon["map_lon"] = map_ready_tambon["tambon_lon"] + kind_offset
                        fig = px.scatter_mapbox(
                            map_ready_tambon,
                            lat="map_lat",
                            lon="map_lon",
                            color="party",
                            size="share_pct",
                            hover_data=[
                                "tambon_label",
                                "ballot_kind",
                                "party",
                                "votes",
                                "share_pct",
                                "polling_units",
                                "tambon_lat",
                                "tambon_lon",
                            ],
                            center={
                                "lat": float(map_ready_tambon["tambon_lat"].mean()),
                                "lon": float(map_ready_tambon["tambon_lon"].mean()),
                            },
                            zoom=10,
                            height=650,
                            title="Verified tambon winner map",
                        )
                        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=45, b=0))
                        st.plotly_chart(fig, width="stretch")
                        st.caption(
                            "Map points are slightly offset by ballot kind so constituency and party-list winners "
                            "at the same tambon centroid do not hide each other. Hover shows the original centroid."
                        )
                    else:
                        st.info("Tambon summary is ready. Add `tambon_lat/tambon_lon` in `data/reference/tambon_reference.csv` to draw a real tambon map.")

                    with st.expander("Tambon party summary"):
                        st.dataframe(tambon_view, width="stretch")

            geo = geo[geo["ballot_kind"].isin(kind_sel)] if "ballot_kind" in geo.columns else geo
            geo = add_display_columns(geo)
            if station_query and "polling_unit_id" in geo.columns:
                searchable = (
                    geo["polling_unit_id"].astype(str)
                    + " "
                    + _series(geo, "unit_label", "").astype(str)
                    + " "
                    + _series(geo, "area_label", "").astype(str)
                    + " "
                    + _series(geo, "file_label", "").astype(str)
                )
                geo = geo[searchable.str.contains(station_query, case=False, na=False)]

            st.subheader("Draft geographic cluster by tambon/moo")
            group_cols = [
                c
                for c in ["ballot_kind", "draft_district", "draft_subdistrict_or_municipality", "draft_moo", "winner_party"]
                if c in geo.columns
            ]
            if group_cols and "winner_votes" in geo.columns:
                summary = (
                    geo.groupby(group_cols, dropna=False)
                    .agg(records=("polling_unit_id", "nunique"), winner_votes=("winner_votes", "sum"), mean_winner_share=("winner_share", "mean"))
                    .reset_index()
                    .sort_values(["ballot_kind", "draft_subdistrict_or_municipality", "draft_moo", "winner_votes"], ascending=[True, True, True, False])
                )
                st.dataframe(summary, width="stretch")

            map_ready = geo.copy()
            for col in ["lat", "lon"]:
                if col not in map_ready.columns:
                    map_ready[col] = np.nan
            map_ready["location_verified"] = map_ready.get("location_verified", False)
            map_ready = map_ready[
                map_ready["location_verified"].astype(bool)
                & map_ready["lat"].notna()
                & map_ready["lon"].notna()
            ]
            st.subheader("Real map from verified coordinates")
            if map_ready.empty:
                st.warning(
                    "ยังไม่มีแถวที่ `location_verified=True` และมี `lat/lon` ครบ จึงยังไม่วาดแผนที่จริง "
                    "เพื่อกันการสรุปผิดพื้นที่ ให้เติม/ตรวจ `data/reference/polling_unit_locations.csv` ก่อน."
                )
            else:
                fig = px.scatter_mapbox(
                    map_ready,
                    lat="lat",
                    lon="lon",
                    color="winner_party",
                    size="winner_share" if "winner_share" in map_ready.columns else None,
                    hover_data=[
                        c
                        for c in [
                            "polling_unit_id",
                            "unit_label",
                            "area_label",
                            "ballot_kind",
                            "winner_party",
                            "winner_votes",
                            "winner_share",
                            "official_district",
                            "official_subdistrict",
                            "official_moo",
                            "draft_municipality",
                            "review_reason",
                        ]
                        if c in map_ready.columns
                    ],
                    zoom=10,
                    height=760,
                    title="Verified polling-unit winners by geographic coordinate",
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=45, b=0))
                st.plotly_chart(fig, width="stretch")

with tab_spatial:
    st.subheader("Pseudo-spatial polling-unit layout")
    st.caption("This is not a geographic map because the OCR data has no lat/lon. It lays out units by source/locality and unit order.")
    spatial_path = FIGURES_DIR / "pseudo_spatial_units.csv"
    if spatial_path.exists():
        spatial = pd.read_csv(spatial_path)
    else:
        spatial = build_pseudo_spatial(df, long)
    if spatial.empty:
        st.info("Run analysis first or add confirmed party rows.")
    else:
        spatial = add_display_columns(spatial)
        spatial = spatial[spatial["ballot_kind"].isin(kind_sel)]
        if quality_mode == "Confirmed only" and "polling_unit_id" in df_f.columns:
            spatial = spatial[spatial["polling_unit_id"].isin(df_f["polling_unit_id"])]
        color_by = st.selectbox("Color by", ["winner_party", "ballot_kind", "locality"], index=0)
        size_by = st.selectbox("Size by", ["winner_share", "total_party_votes", "party_count"], index=0)
        fig = px.scatter(
            spatial,
            x="layout_x",
            y="layout_y",
            color=color_by,
            size=size_by,
            facet_row="ballot_kind" if len(spatial["ballot_kind"].dropna().unique()) > 1 else None,
            hover_data=[
                "unit_label",
                "area_label",
                "unit_number",
                "winner_party",
                "winner_votes" if "winner_votes" in spatial.columns else "votes",
                "winner_share",
            ],
            title="Winner pattern by source/locality and polling-unit order",
        )
        fig.update_layout(height=760)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(display_table(spatial), width="stretch")

with tab_network:
    st.subheader("Party and polling-unit networks")
    nodes_path = FIGURES_DIR / "network_nodes.csv"
    edges_path = FIGURES_DIR / "network_edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        st.info("Run `python 05_analysis\\analysis.py` to generate network artifacts.")
    else:
        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        edge_kind = st.selectbox("Network type", ["party_similarity", "unit_top_party"], index=0)
        kind_for_network = st.selectbox("Ballot kind for network", sorted(edges["ballot_kind"].dropna().unique()))
        edge_view = edges[(edges["edge_type"] == edge_kind) & (edges["ballot_kind"] == kind_for_network)].copy()
        if edge_view.empty:
            st.info("No edges for this selection.")
        elif edge_kind == "party_similarity":
            threshold = st.slider("Minimum absolute correlation", 0.0, 1.0, 0.35, 0.05)
            edge_view = edge_view[edge_view["abs_weight"] >= threshold]
            st.plotly_chart(circular_network(edge_view), width="stretch")
            top = edge_view.sort_values("abs_weight", ascending=False).head(40)
            fig = px.bar(
                top.iloc[::-1],
                x="weight",
                y=top.iloc[::-1]["source_label"] + " <-> " + top.iloc[::-1]["target_label"],
                orientation="h",
                color="weight",
                color_continuous_scale="RdBu",
                title="Strongest party-share similarity edges",
            )
            fig.update_layout(height=max(480, 24 * len(top)))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(top, width="stretch")
        else:
            min_share = st.slider("Minimum unit-party vote share", 0.0, 1.0, 0.2, 0.05)
            top = edge_view[edge_view["weight"] >= min_share].sort_values("weight", ascending=False).head(100)
            fig = px.scatter(
                top,
                x="target_label",
                y="source_label",
                size="weight",
                color="weight",
                hover_data=["source_label", "target_label", "weight"],
                title="Polling units connected to their top parties",
            )
            fig.update_layout(height=850)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(top, width="stretch")
        with st.expander("Network nodes"):
            st.dataframe(nodes[nodes["ballot_kind"] == kind_for_network], width="stretch")

with tab_station:
    station_sel = st.selectbox(
        "Polling unit",
        station_options,
        index=0,
        format_func=lambda value: station_label_lookup.get(value, value),
    )
    st_df = df[df["polling_unit_id"].astype(str) == station_sel]
    st_long = long[long["polling_unit_id"].astype(str) == station_sel] if not long.empty else long
    st.dataframe(display_table(st_df), width="stretch")
    if not st_long.empty:
        for kind, sub in st_long.groupby("ballot_kind"):
            agg = sub.groupby("party", as_index=False)["votes"].sum().sort_values("votes", ascending=False)
            st.plotly_chart(px.bar(agg.head(25).iloc[::-1], x="votes", y="party", orientation="h", title=kind), width="stretch")

with tab_anomaly:
    anomaly_path = FIGURES_DIR / "anomaly_records.csv"
    if anomaly_path.exists():
        anomalies = pd.read_csv(anomaly_path)
        st.dataframe(display_table(anomalies), width="stretch")
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
        st.dataframe(display_table(quick), width="stretch")

with tab_data:
    st.subheader("Filtered records")
    st.dataframe(display_table(df_f), width="stretch")
    st.download_button(
        "Download filtered records",
        df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_election_records.csv",
        mime="text/csv",
    )
    if not long_f.empty:
        st.subheader("Long party rows")
        st.dataframe(display_table(long_f), width="stretch")
        st.download_button(
            "Download long party rows",
            long_f.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_party_long.csv",
            mime="text/csv",
        )
