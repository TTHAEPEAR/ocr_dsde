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
PARTY_COLORS = {
    "กล้าธรรม": "#16a34a",
    "ประชาชน": "#f97316",
    "ภูมิใจไทย": "#2563eb",
    "เพื่อไทย": "#dc2626",
    "ประชาธิปัตย์": "#38bdf8",
    "ประชาธิปัต": "#38bdf8",
    "เพื่อชาติไทย": "#c026d3",
}
DEFAULT_COLORS = [
    "#8b5cf6",
    "#14b8a6",
    "#eab308",
    "#ec4899",
    "#64748b",
    "#a855f7",
    "#06b6d4",
    "#84cc16",
    "#f59e0b",
    "#ef4444",
]

st.set_page_config(page_title=f"Election Dashboard - {CONSTITUENCY_NAME}", layout="wide")

st.markdown(
    """
<style>
    :root {
        --bg-card: rgba(15, 23, 42, 0.74);
        --border-soft: rgba(148, 163, 184, 0.18);
        --text-soft: #94a3b8;
        --accent: #38bdf8;
        --accent-2: #f43f5e;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1480px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(30,41,59,.82), rgba(15,23,42,.82));
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 18px 50px rgba(2, 6, 23, 0.18);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-soft);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 760;
    }
    .hero {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
        background:
            linear-gradient(135deg, rgba(14,165,233,.20), rgba(244,63,94,.08) 42%, rgba(15,23,42,.88)),
            rgba(15,23,42,.86);
    }
    .hero-eyebrow {
        color: var(--accent);
        text-transform: uppercase;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .08rem;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.15rem;
        line-height: 1.14;
        font-weight: 850;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        max-width: 920px;
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .story-card {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        padding: 15px 16px;
        background: var(--bg-card);
        margin: 8px 0 14px;
    }
    .chapter {
        border-left: 4px solid var(--accent);
        padding: 12px 0 12px 18px;
        margin: 18px 0 10px;
    }
    .chapter-kicker {
        color: var(--accent);
        font-size: .78rem;
        font-weight: 850;
        letter-spacing: .06rem;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .chapter-title {
        color: #f8fafc;
        font-size: 1.42rem;
        font-weight: 820;
        line-height: 1.24;
    }
    .chapter-subtitle {
        color: #cbd5e1;
        max-width: 980px;
        line-height: 1.55;
        margin-top: 4px;
    }
    .bridge {
        border: 1px solid rgba(56, 189, 248, .26);
        border-radius: 8px;
        padding: 12px 14px;
        background: rgba(8, 47, 73, .34);
        color: #dbeafe;
        margin: 8px 0 18px;
    }
    .bridge strong {
        color: #7dd3fc;
    }
    .story-card strong {
        color: #f8fafc;
    }
    .small-note {
        color: var(--text-soft);
        font-size: .88rem;
        line-height: 1.45;
    }
    div[data-testid="stTabs"] button p {
        font-weight: 720;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


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


def _format_pct(value: object, digits: int = 1) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "N/A"
    return f"{parsed * 100:.{digits}f}%"


def normalize_party_name(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^พรรค", "", text).strip()
    if text == "ประชาธิปัต":
        return "ประชาธิปัตย์"
    return text


def party_color(value: object, fallback_index: int = 0) -> str:
    key = normalize_party_name(value)
    if key in PARTY_COLORS:
        return PARTY_COLORS[key]
    return DEFAULT_COLORS[fallback_index % len(DEFAULT_COLORS)]


def party_color_map(values) -> dict[str, str]:
    mapping = {}
    fallback_index = 0
    for value in pd.Series(values).dropna().astype(str).unique().tolist():
        key = normalize_party_name(value)
        if key in PARTY_COLORS:
            mapping[value] = PARTY_COLORS[key]
        else:
            mapping[value] = DEFAULT_COLORS[fallback_index % len(DEFAULT_COLORS)]
            fallback_index += 1
    return mapping


def hex_to_rgba(color: str, alpha: float = 0.55) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return f"rgba(100,116,139,{alpha})"
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def party_from_flow_label(label: object) -> str:
    text = str(label or "")
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def story_card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="story-card">
  <strong>{title}</strong>
  <div class="small-note">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def story_chapter(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="chapter">
  <div class="chapter-kicker">{kicker}</div>
  <div class="chapter-title">{title}</div>
  <div class="chapter-subtitle">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def story_bridge(text: str) -> None:
    st.markdown(f"""<div class="bridge"><strong>Next:</strong> {text}</div>""", unsafe_allow_html=True)


def presentation_table(label: str, frame: pd.DataFrame, presentation_mode: bool, expanded: bool = False) -> None:
    if presentation_mode:
        with st.expander(label, expanded=expanded):
            st.dataframe(frame, width="stretch")
    else:
        st.dataframe(frame, width="stretch")


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


def scatter_map(*args, **kwargs):
    """Use Plotly's current map API, falling back for older installations."""
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(*args, **kwargs)
        fig.update_layout(map_style="open-street-map")
    else:
        fig = px.scatter_mapbox(*args, **kwargs)
        fig.update_layout(mapbox_style="open-street-map")
    return fig


def _is_focus_party(party_name: str) -> bool:
    """Return True if the party is one of the three focus parties for flow insight."""
    normalized = normalize_party_name(party_name)
    return normalized in {"กล้าธรรม", "ประชาชน", "ประชาธิปัตย์", "ประชาธิปัต"}


def split_flow_figure(matrix: pd.DataFrame, limit: int = 18, title: str = "Winner flow across the two ballots") -> go.Figure:
    GRAY_NODE = "rgba(160,170,180,0.45)"
    GRAY_LINK = "rgba(160,170,180,0.18)"

    top_matrix = matrix.sort_values("polling_units", ascending=False).head(limit).copy()
    left_labels = top_matrix["winner_party_constituency"].astype(str).map(lambda v: f"Constituency: {v}")

    # For non-focus party-list parties, use a generic label to de-emphasize
    _other_counter: dict[str, int] = {}

    def _right_label(party: str) -> str:
        if _is_focus_party(party):
            return f"Party-list: {party}"
        # Keep a unique label per non-focus party so links stay separate,
        # but display generic text to pull attention away.
        if party not in _other_counter:
            _other_counter[party] = len(_other_counter) + 1
        return f"Party-list: อื่นๆ ({_other_counter[party]})"

    right_labels = top_matrix["winner_party_party_list"].astype(str).map(_right_label)

    labels = pd.unique(pd.concat([left_labels, right_labels], ignore_index=True)).tolist()
    label_index = {label: i for i, label in enumerate(labels)}

    # Node colors: gray for non-focus party-list nodes
    node_colors = []
    for i, label in enumerate(labels):
        party = party_from_flow_label(label)
        is_party_list_node = label.startswith("Party-list:")
        if is_party_list_node and not _is_focus_party(party):
            node_colors.append(GRAY_NODE)
        else:
            node_colors.append(party_color(party, i))

    # Link colors: gray if the target party-list party is non-focus
    link_colors = []
    for i, (_, row) in enumerate(top_matrix.iterrows()):
        pl_party = str(row["winner_party_party_list"])
        if not _is_focus_party(pl_party):
            link_colors.append(GRAY_LINK)
        else:
            link_colors.append(hex_to_rgba(party_color(row["winner_party_constituency"], i), 0.5))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=18,
                    thickness=18,
                    color=node_colors,
                ),
                link=dict(
                    source=[label_index[v] for v in left_labels],
                    target=[label_index[v] for v in right_labels],
                    value=top_matrix["polling_units"].astype(float).tolist(),
                    color=link_colors,
                    customdata=top_matrix[
                        [
                            "winner_party_constituency",
                            "winner_party_party_list",
                            "polling_units",
                            "mean_constituency_winner_share",
                            "mean_party_list_winner_share",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "Constituency winner: %{customdata[0]}<br>"
                        "Party-list winner: %{customdata[1]}<br>"
                        "Polling units: %{customdata[2]}<br>"
                        "Mean constituency winner share: %{customdata[3]:.1%}<br>"
                        "Mean party-list winner share: %{customdata[4]:.1%}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=42, b=10), title=title)
    return fig


def dominance_frontier_figure(splits: pd.DataFrame, title: str = "Dominance frontier") -> go.Figure:
    plot_df = splits.copy()
    plot_df["split_label"] = np.where(plot_df["split_ticket"], "split winner", "same winner")
    fig = px.scatter(
        plot_df,
        x="winner_share_constituency",
        y="winner_share_party_list",
        color="split_label",
        size="total_ballots" if "total_ballots" in plot_df.columns else None,
        hover_data=[
            c
            for c in [
                "unit_label",
                "area_label",
                "winner_party_constituency",
                "winner_party_party_list",
                "winner_share_constituency",
                "winner_share_party_list",
                "winner_margin_share_constituency",
                "winner_margin_share_party_list",
                "file_label",
            ]
            if c in plot_df.columns
        ],
        title=title,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(148,163,184,0.7)", dash="dash"))
    fig.update_layout(
        height=560,
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        xaxis_title="Constituency winner share",
        yaxis_title="Party-list winner share",
        legend_title_text="Winner relationship",
    )
    return fig


def number_collision_insight(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare constituency no.2 Klatham with party-list no.2 Puea Chat Thai and no.42 Klatham."""
    if df.empty or "polling_unit_id" not in df.columns:
        return {}, pd.DataFrame()
    required = {"candidate_2_votes", "candidate_42_votes", "ballot_kind"}
    if not required.intersection(df.columns):
        return {}, pd.DataFrame()

    const = df[df["ballot_kind"].astype(str).eq("constituency")].copy()
    party = df[df["ballot_kind"].astype(str).eq("party_list")].copy()
    if const.empty or party.empty or "candidate_2_votes" not in const.columns:
        return {}, pd.DataFrame()

    const2 = pd.to_numeric(const.set_index("polling_unit_id")["candidate_2_votes"], errors="coerce")
    party2 = pd.to_numeric(party.set_index("polling_unit_id")["candidate_2_votes"], errors="coerce")
    party42 = pd.to_numeric(
        party.set_index("polling_unit_id").get("candidate_42_votes", pd.Series(dtype=float)),
        errors="coerce",
    )
    unit = pd.concat(
        {
            "klatham_constituency_no2": const2,
            "puea_chat_thai_party_list_no2": party2,
            "klatham_party_list_no42": party42,
        },
        axis=1,
    ).dropna()
    if unit.empty:
        return {}, pd.DataFrame()

    const_votes = pd.DataFrame(
        {
            i: pd.to_numeric(const.get(f"candidate_{i}_votes", pd.Series(0, index=const.index)), errors="coerce").fillna(0)
            for i in range(1, 8)
        },
        index=const.index,
    )
    winner_no = const_votes.idxmax(axis=1)
    klatham_win_units = set(const.loc[winner_no.eq(2), "polling_unit_id"])
    unit["klatham_won_constituency"] = unit.index.isin(klatham_win_units)
    unit["party_list_no2_minus_klatham_no42"] = unit["puea_chat_thai_party_list_no2"] - unit["klatham_party_list_no42"]
    
    # Format labels to be human-readable instead of raw file names
    loc = load_location_enrichment()
    if not loc.empty and "polling_unit_id" in loc.columns:
        loc_map = loc.drop_duplicates("polling_unit_id").set_index("polling_unit_id")
    else:
        loc_map = pd.DataFrame()

    def format_unit_label(idx):
        import re
        if not loc_map.empty and idx in loc_map.index:
            row = loc_map.loc[idx]
            sub = row.get("official_subdistrict") or row.get("draft_subdistrict_or_municipality") or ""
            unit_no = row.get("path_unit_number") or row.get("draft_unit_number") or "?"
            sub_clean = str(sub).replace("ตำบล", "").replace("ตําบล", "").replace("ทม", "").replace("_", "")
        else:
            label = str(idx)
            label = re.sub(r'_000\d+$', '', label)
            m = re.search(r'หน_วย.*?_(\d+)', label)
            unit_no = m.group(1) if m else '?'
            sub_clean = label.split('หน_วย')[0]
            sub_clean = re.sub(r'_[0-9]{8}T[0-9]{6}Z.*?(ทม_|ตําบล|ตำบล)', r'\1', sub_clean)
            sub_clean = sub_clean.replace('_', '').replace('ตำบล', '').replace('ตําบล', '').replace('ทม', '')
            
        # Fix common OCR/raw string anomalies
        sub_clean = sub_clean.replace("ทาขุนราม", "ท่าขุนราม").replace("ทาขนราม", "ท่าขุนราม").replace("ทาขุน", "ท่าขุน")
        sub_clean = sub_clean.replace("แมลาย", "แม่ลาย").replace("นครชม", "นครชุม")
        sub_clean = sub_clean.replace("หนองปลง", "หนองปลิง").replace("ทนองปลิง", "หนองปลิง")
        sub_clean = sub_clean.strip("1234567890 ")
        if sub_clean:
            return f"ต.{sub_clean} หน่วยฯ {unit_no}"
        return str(idx)

    unit["unit_label"] = unit.index.map(format_unit_label)

    focus = unit[unit["klatham_won_constituency"]]
    summary = {
        "paired_units": int(len(unit)),
        "klatham_won_units": int(len(focus)),
        "units_no2_beats_no42_all": int((unit["party_list_no2_minus_klatham_no42"] > 0).sum()),
        "units_no2_beats_no42_klatham_won": int((focus["party_list_no2_minus_klatham_no42"] > 0).sum()) if not focus.empty else 0,
        "constituency_no2_total_all": float(unit["klatham_constituency_no2"].sum()),
        "party_list_no2_total_all": float(unit["puea_chat_thai_party_list_no2"].sum()),
        "party_list_no42_total_all": float(unit["klatham_party_list_no42"].sum()),
        "party_list_no2_total_klatham_won": float(focus["puea_chat_thai_party_list_no2"].sum()) if not focus.empty else 0.0,
        "party_list_no42_total_klatham_won": float(focus["klatham_party_list_no42"].sum()) if not focus.empty else 0.0,
    }
    return summary, unit.reset_index(names="polling_unit_id")


@st.cache_data
def load_insight_tables():
    paths = {
        "splits": FIGURES_DIR / "cross_ballot_winner_splits.csv",
        "matrix": FIGURES_DIR / "cross_ballot_winner_matrix.csv",
        "dominance": FIGURES_DIR / "unit_party_dominance.csv",
        "strongholds": FIGURES_DIR / "stronghold_units.csv",
        "competitiveness": FIGURES_DIR / "unit_competitiveness.csv",
    }
    tables = {}
    for name, path in paths.items():
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for name in ["splits", "matrix", "dominance", "strongholds", "competitiveness"]:
        for col in tables[name].columns:
            if col.endswith("_share") or col in {"winner_share", "runner_up_share", "winner_margin_share", "hhi", "effective_number_of_parties"}:
                tables[name][col] = pd.to_numeric(tables[name][col], errors="coerce")
        if "split_ticket" in tables[name].columns:
            tables[name]["split_ticket"] = tables[name]["split_ticket"].map(lambda v: _to_bool(v) if pd.notna(v) else False)
    return tables


@st.cache_data
def load_multisource_features():
    path = FIGURES_DIR / "tambon_multisource_features.csv"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    for col in df.columns:
        if col.endswith("_share") or col in {"urbanization_index", "split_rate", "distance_to_center_km",
                                               "pca_x", "pca_y", "distance_score", "density_score", "mean_turnout"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    return df, path


@st.cache_data
def load_national_2023():
    path = REFERENCE_DIR / "national_2023_kp1.csv"
    if not path.exists():
        return pd.DataFrame(), None
    return pd.read_csv(path), path


@st.cache_data
def load_google_trends():
    path = FIGURES_DIR / "google_trends_over_time.csv"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    return df, path


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

insight_tables = load_insight_tables()
split_table = insight_tables.get("splits", pd.DataFrame())
matrix_table = insight_tables.get("matrix", pd.DataFrame())
confirmed_rows = int(df["is_confirmed"].sum())
total_rows = len(df)
confirmed_rate = confirmed_rows / total_rows if total_rows else np.nan
needs_review_rows = int((~df["is_confirmed"]).sum())
split_units = int(split_table.get("split_ticket", pd.Series(dtype=bool)).sum()) if not split_table.empty else 0
split_rate = split_table.get("split_ticket", pd.Series(dtype=bool)).mean() if not split_table.empty else np.nan
top_flow = matrix_table.sort_values("polling_units", ascending=False).head(1) if not matrix_table.empty else pd.DataFrame()
top_flow_value = "N/A"
top_flow_delta = "Run analysis"
if not top_flow.empty:
    flow = top_flow.iloc[0]
    top_flow_value = f"{int(flow['polling_units']):,} units"
    top_flow_delta = f"{flow['winner_party_constituency']} -> {flow['winner_party_party_list']}"

st.markdown(
    f"""
<div class="hero">
  <div class="hero-eyebrow">Research dashboard</div>
  <div class="hero-title">Election 2026: {CONSTITUENCY_NAME}</div>
  <div class="hero-subtitle">
    Quality-aware OCR analysis for polling-unit evidence, split-ticket behavior, spatial patterns,
    and review-ready anomalies. The default view uses confirmed rows first and keeps raw audit detail tucked away.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Confirmed records", f"{confirmed_rows:,}", _format_pct(confirmed_rate))
k2.metric("Needs review", f"{needs_review_rows:,}", "kept separate")
k3.metric("Split-ticket units", f"{split_units:,}", _format_pct(split_rate))
k4.metric("Dominant flow", top_flow_value, top_flow_delta)
st.caption(f"Data source: {source}")

with st.sidebar:
    st.header("Presentation Controls")
    presentation_mode = st.toggle("Presentation mode", value=True)
    st.caption("Presentation mode hides raw/debug tables behind expanders.")

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

primary_tabs = ["Presentation Story", "Evidence", "Drilldown"]
appendix_tabs = [
    "Quality Gate",
    "Summary",
    "Split Ticket",
    "Parties",
    "Geo Map",
    "Spatial",
    "Networks",
    "Outliers",
    "Data Lab",
]
tab_names = primary_tabs + appendix_tabs
tab_lookup = dict(zip(tab_names, st.tabs(tab_names)))

with tab_lookup["Presentation Story"]:
    story_chapter(
        "Act 1",
        "Start with trust: this is not just OCR output, it is quality-filtered evidence.",
        "Before talking about politics, the dashboard separates confirmed rows from rows that still need human review. That makes every later claim easier to defend.",
    )
    q = pd.DataFrame(
        {
            "status": ["confirmed", "needs_review"],
            "rows": [int(df["is_confirmed"].sum()), int((~df["is_confirmed"]).sum())],
        }
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Confirmed records used first", f"{confirmed_rows:,}", _format_pct(confirmed_rate))
    c2.metric("Review queue kept visible", f"{needs_review_rows:,}", "excluded by default")
    c3.metric("Polling units paired", f"{df['polling_unit_id'].nunique():,}")
    qfig = px.bar(q, x="status", y="rows", color="status", text="rows", title="Quality gate before interpretation")
    qfig.update_layout(height=360, showlegend=False, yaxis_title="Records")
    st.plotly_chart(qfig, width="stretch")

    story_bridge("Once the evidence is bounded, the main question becomes whether voters behaved the same way across the two ballots.")

    story_chapter(
        "Act 2",
        "The election story is not one race. It is two linked choices in the same polling unit.",
        "The flow chart connects the winner of the constituency ballot to the winner of the party-list ballot. Thick bands reveal the dominant political trade-off.",
    )
    if split_table.empty or matrix_table.empty:
        st.info("Run `python 05_analysis\\analysis.py` to generate split-ticket story artifacts.")
    else:
        story_splits = add_display_columns(split_table.copy())
        story_matrix = matrix_table.copy()
        split_rate_story = story_splits["split_ticket"].mean() if len(story_splits) else np.nan
        s1, s2, s3 = st.columns(3)
        s1.metric("Units with both ballots", f"{len(story_splits):,}")
        s2.metric("Split-ticket units", f"{int(story_splits['split_ticket'].sum()):,}", _format_pct(split_rate_story))
        s3.metric("Winner-pair patterns", f"{len(story_matrix):,}")
        lead = story_matrix.sort_values("polling_units", ascending=False).iloc[0]
        story_card(
            "The headline pattern",
            f"The largest flow is {lead['winner_party_constituency']} on constituency ballots to {lead['winner_party_party_list']} on party-list ballots, covering {int(lead['polling_units']):,} polling units.",
        )
        st.plotly_chart(split_flow_figure(story_matrix, limit=14, title="How constituency winners flow into party-list winners"), width="stretch")

        story_bridge("The flow tells us who switches. The next question is why one small party-list choice is unusually visible.")

        story_chapter(
            "Act 3",
            "A number collision: local candidate no.2 and party-list no.2 point to different parties.",
            "กล้าธรรม wins many constituency ballots with local candidate number 2. But on the party-list ballot, number 2 is เพื่อชาติไทย, while กล้าธรรม is number 42. That creates a testable pattern: did the party-list no.2 receive attention where กล้าธรรม was locally strong?",
        )
        collision_summary, collision_units = number_collision_insight(df)
        if collision_summary:
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Klatham constituency wins", f"{collision_summary['klatham_won_units']:,}", "candidate no.2")
            n2.metric(
                "No.2 beats no.42",
                f"{collision_summary['units_no2_beats_no42_klatham_won']:,}",
                "within Klatham-won units",
            )
            n3.metric("Puea Chat Thai PL no.2", f"{collision_summary['party_list_no2_total_klatham_won']:,.0f}")
            n4.metric("Klatham PL no.42", f"{collision_summary['party_list_no42_total_klatham_won']:,.0f}")

            totals = pd.DataFrame(
                [
                    {
                        "Ballot / number": "Constituency no.2",
                        "Party": "กล้าธรรม",
                        "Votes": collision_summary["constituency_no2_total_all"],
                    },
                    {
                        "Ballot / number": "Party-list no.2",
                        "Party": "เพื่อชาติไทย",
                        "Votes": collision_summary["party_list_no2_total_all"],
                    },
                    {
                        "Ballot / number": "Party-list no.42",
                        "Party": "กล้าธรรม",
                        "Votes": collision_summary["party_list_no42_total_all"],
                    },
                ]
            )
            fig = px.bar(
                totals,
                x="Ballot / number",
                y="Votes",
                color="Party",
                text="Votes",
                color_discrete_map=party_color_map(totals["Party"]),
                title="Same visible number, different party identity across ballots",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=430, yaxis_title="Votes")
            st.plotly_chart(fig, width="stretch")

            focus = collision_units[collision_units["klatham_won_constituency"]].copy()
            if not focus.empty:
                top_collision = focus.sort_values("party_list_no2_minus_klatham_no42", ascending=False).head(15)
                fig = px.bar(
                    top_collision.iloc[::-1],
                    x="party_list_no2_minus_klatham_no42",
                    y="unit_label",
                    orientation="h",
                    color="party_list_no2_minus_klatham_no42",
                    color_continuous_scale="Oranges",
                    hover_data=[
                        "klatham_constituency_no2",
                        "puea_chat_thai_party_list_no2",
                        "klatham_party_list_no42",
                    ],
                    title="Where party-list no.2 most exceeds Klatham party-list no.42",
                )
                fig.update_layout(height=520, xaxis_title="Party-list no.2 minus Klatham no.42", yaxis_title="")
                st.plotly_chart(fig, width="stretch")
            story_card(
                "Interpret carefully",
                "This is not proof of voter confusion by itself. It is a strong, presentation-worthy clue: a locally dominant candidate number overlaps with another party's party-list number, and that party-list number outperforms Klatham's own party-list number in many Klatham-won units.",
            )
        else:
            st.info("Number-collision insight needs candidate_2 and candidate_42 columns in the cleaned dataset.")

        story_bridge("The number-collision clue raises a broader question: were these wins landslides, narrow wins, or ballot-specific effects?")

        story_chapter(
            "Act 4",
            "Dominance shows whether the split is strategic, local, or simply noisy.",
            "Points far from the diagonal mean one ballot was much more decisive than the other. Those are the units worth discussing, because they show local candidate strength versus party preference.",
        )
        st.plotly_chart(dominance_frontier_figure(story_splits, "Dominance frontier: candidate strength vs party-list strength"), width="stretch")

    story_bridge("Now that the voting behavior is visible, we ask where the pattern lives geographically.")

    story_chapter(
        "Act 5",
        "Geography turns vote shares into territory.",
        "Tambon-level winners show whether the pattern is scattered or spatially concentrated. This is where the story becomes a field map instead of only a spreadsheet.",
    )
    tambon_summary, tambon_source = load_tambon_summary()
    if tambon_summary.empty:
        st.info("Run `python 05_analysis\\analysis.py` and verify tambon coordinates to generate the map story.")
    else:
        if "is_tambon_winner" in tambon_summary.columns:
            winners = tambon_summary[tambon_summary["is_tambon_winner"].map(lambda v: _to_bool(v))].copy()
        else:
            winners = pd.DataFrame()
        winners = winners[winners["ballot_kind"].isin(kind_sel)] if "ballot_kind" in winners.columns else winners
        if winners.empty:
            st.info("No tambon winners after current filters.")
        else:
            winners["share_pct"] = winners["tambon_vote_share"] * 100
            # Standardize party names to group 'พรรคประชาชน' and 'ประชาชน' together
            winners["party"] = winners["party"].str.replace(r"^พรรค", "", regex=True).str.strip()
            map_ready = winners[
                winners.get("tambon_verified", pd.Series(False, index=winners.index)).map(lambda v: _to_bool(v))
                & winners["tambon_lat"].notna()
                & winners["tambon_lon"].notna()
            ].copy() if {"tambon_lat", "tambon_lon"}.issubset(winners.columns) else pd.DataFrame()
            g1, g2, g3 = st.columns(3)
            g1.metric("Tambons represented", f"{winners['tambon_label'].nunique():,}")
            g2.metric("Winner parties on map", f"{winners['party'].nunique():,}")
            g3.metric("Map-ready tambon rows", f"{len(map_ready):,}")
            if not map_ready.empty:
                offset = map_ready["ballot_kind"].map({"constituency": -0.006, "party_list": 0.006}).fillna(0)
                map_ready["map_lat"] = map_ready["tambon_lat"] + offset
                map_ready["map_lon"] = map_ready["tambon_lon"] + offset
                fig = scatter_map(
                map_ready,
                lat="map_lat",
                lon="map_lon",
                color="party",
                color_discrete_map=party_color_map(map_ready["party"]),
                size="share_pct",
                    hover_data=["tambon_label", "ballot_kind", "party", "votes", "share_pct", "polling_units"],
                    center={"lat": float(map_ready["tambon_lat"].mean()), "lon": float(map_ready["tambon_lon"].mean())},
                    zoom=10,
                    height=560,
                    title="Tambon-level winner map",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=45, b=0))
                st.plotly_chart(fig, width="stretch")
            else:
                fig = px.scatter(
                winners,
                x="tambon_label",
                y="ballot_kind",
                color="party",
                color_discrete_map=party_color_map(winners["party"]),
                size="share_pct",
                    hover_data=["tambon_label", "ballot_kind", "party", "votes", "tambon_vote_share", "polling_units"],
                    title="Tambon winners by ballot kind",
                )
                fig.update_layout(height=430)
                st.plotly_chart(fig, width="stretch")

    story_bridge("The geographic pattern invites a deeper question: what predicts how a tambon votes? We bring in data from outside the ballot box.")

    # ---- Act 6: 2023 vs 2026 Comparison ----
    story_chapter(
        "Act 6",
        "Same constituency, different era: how 2026 compares to 2023.",
        "Both elections are ส.ส. races for **Kamphaeng Phet Constituency 1 (เขต 1)** — the exact same geographic area and the same type of election. In 2023 it was won by ไผ่ ลิกค์ from พลังประชารัฐ, a party not even on the 2026 ballot. By placing both elections side by side we reveal how much the political landscape has shifted.",
    )
    nat_2023, nat_source = load_national_2023()
    if not nat_2023.empty and not split_table.empty:
        # Build 2026 aggregate for comparison
        perf_path = FIGURES_DIR / "party_performance.csv"
        if perf_path.exists():
            perf = pd.read_csv(perf_path)
            perf_conf = perf[perf["quality_slice"] == "confirmed"].copy()
            # Constituency comparison
            local_const = perf_conf[perf_conf["ballot_kind"] == "constituency"][["party", "vote_share"]].copy()
            local_const["election"] = "2026 (constituency)"
            local_const["vote_share"] = pd.to_numeric(local_const["vote_share"], errors="coerce")
            # Standardize party names for clean x-axis
            local_const["party"] = local_const["party"].str.replace(r"^พรรค", "", regex=True).str.strip()
            
            # Party-list comparison
            local_pl = perf_conf[perf_conf["ballot_kind"] == "party_list"][["party", "vote_share"]].copy()
            local_pl["election"] = "2026 (party-list)"
            local_pl["vote_share"] = pd.to_numeric(local_pl["vote_share"], errors="coerce")
            local_pl["party"] = local_pl["party"].str.replace(r"^พรรค", "", regex=True).str.strip()
            
            # 2023
            nat_compare = nat_2023[["party", "vote_share"]].copy()
            nat_compare["election"] = "2023 (เขต 1)"
            nat_compare["party"] = nat_compare["party"].str.replace(r"^พรรค", "", regex=True).str.strip()
            
            # Combine top parties
            combined = pd.concat([nat_compare, local_const.head(7), local_pl.head(7)], ignore_index=True)
            combined = combined[combined["vote_share"].notna() & (combined["vote_share"] > 0.01)]
            if not combined.empty:
                n1, n2 = st.columns(2)
                n1.metric("2023 winner (เขต 1)", "พลังประชารัฐ", "not on 2026 ballot")
                n2.metric("2026 winner (เขต 1)", "กล้าธรรม", "51% constituency")
                fig = px.bar(
                    combined.sort_values("vote_share", ascending=False),
                    x="party",
                    y="vote_share",
                    color="election",
                    barmode="group",
                    text=combined["vote_share"].map(lambda v: f"{v:.1%}" if pd.notna(v) else ""),
                    title="Party landscape shift: 2023 vs 2026 — Kamphaeng Phet เขต 1",
                    color_discrete_sequence=["#64748b", "#16a34a", "#f97316"],
                )
                fig.update_layout(height=480, xaxis_title="", yaxis_title="Vote share", yaxis_tickformat=".0%")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, width="stretch")
                story_card(
                    "The landscape reshuffled completely",
                    "พลังประชารัฐ won the 2023 race but did not contest in 2026. "
                    "กล้าธรรม, absent from the 2023 ballot, now dominates with 51% constituency share. "
                    "ก้าวไกล (17% in 2023) became ประชาชน (35% in 2026) — the party succession is visible in the numbers.",
                )
        else:
            st.info("Run `python 05_analysis/analysis.py` to generate party performance data.")
    else:
        st.info("Run `python 05_analysis/enrich_external.py` to generate comparison data.")

    # ---- Google Trends visualization ----
    trends_data, trends_source = load_google_trends()
    if not trends_data.empty:
        st.markdown("---")
        st.subheader("🔍 Google Trends: party search interest before election day")
        party_cols = [c for c in trends_data.columns if c != "date" and c != "isPartial"]
        if party_cols:
            trend_melt = trends_data.melt(
                id_vars=["date"], value_vars=party_cols,
                var_name="party", value_name="interest",
            )
            trend_melt["date"] = pd.to_datetime(trend_melt["date"], errors="coerce")
            trend_melt["interest"] = pd.to_numeric(trend_melt["interest"], errors="coerce")
            fig = px.line(
                trend_melt,
                x="date",
                y="interest",
                color="party",
                title="Google Trends: search interest before election (Thailand)",
                markers=True,
            )
            # Add election day marker
            try:
                fig.add_vline(
                    x=pd.Timestamp("2026-05-07"),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Election Day",
                    annotation_position="top left",
                )
            except Exception:
                pass  # skip vline if date type mismatch
            fig.update_layout(
                height=420,
                xaxis_title="",
                yaxis_title="Search interest (0-100)",
                legend_title_text="Party",
            )
            st.plotly_chart(fig, width="stretch")
            story_card(
                "Online awareness ≠ ballot strength",
                "กล้าธรรม had near-zero search interest until weeks before the election, yet won 51% of constituency votes. "
                "ประชาชน maintained steady online presence but translated it to only 35%. "
                "This gap between digital footprint and ballot-box result highlights the power of local ground operations over online visibility.",
            )

    story_bridge("With the political context set, we now ask: does the character of a tambon predict how it votes?")

    # ---- Act 7: Multi-source Clustering ----
    story_chapter(
        "Act 7",
        "Beyond the ballot box: what kind of community votes how?",
        "We combine election results with non-election data — OSM point-of-interest density (shops, schools, banks) as a real-world urbanization proxy and distance from city center — to cluster tambons. The question is not just who won, but whether the community's character predicts the voting pattern.",
    )
    ms_features, ms_path = load_multisource_features()
    if not ms_features.empty:
        ms_valid = ms_features[ms_features["cluster"].notna()].copy()
        ms_valid["cluster_label"] = ms_valid["cluster"].map(
            {0.0: "Rural Loyal", 1.0: "Peri-urban Swing", 2.0: "Urban Core"}
        ).fillna("Unknown")
        # Sort clusters by distance to center for consistent labeling
        cluster_order = ms_valid.groupby("cluster")["distance_to_center_km"].mean().sort_values(ascending=False)
        label_map = {}
        cluster_names = ["Rural Loyal", "Peri-urban Swing", "Urban Core"]
        for i, (cluster_id, _) in enumerate(cluster_order.items()):
            if i < len(cluster_names):
                label_map[cluster_id] = cluster_names[i]
        ms_valid["cluster_label"] = ms_valid["cluster"].map(label_map).fillna("Unknown")

        m1, m2, m3 = st.columns(3)
        for i, (cl, sub) in enumerate(ms_valid.groupby("cluster_label")):
            col = [m1, m2, m3][i % 3]
            col.metric(
                cl,
                f"{len(sub)} tambons",
                f"avg {sub['distance_to_center_km'].mean():.1f} km from center",
            )

        # Distance regression scatter
        const_share_col = [c for c in ms_features.columns if c.startswith("constituency_share_") and "กล้าธรรม" in c]
        if const_share_col:
            share_col = const_share_col[0]
            reg_df = ms_valid[ms_valid[share_col].notna()].copy()
            if not reg_df.empty:
                fig = px.scatter(
                    reg_df,
                    x="distance_to_center_km",
                    y=share_col,
                    color="cluster_label",
                    size="n_paired_units" if "n_paired_units" in reg_df.columns else None,
                    hover_data=["official_subdistrict", "split_rate", "urbanization_index"],
                    trendline="ols",
                    title="Distance from city center vs กล้าธรรม vote share",
                    color_discrete_map={"Urban Core": "#dc2626", "Peri-urban Swing": "#f97316", "Rural Loyal": "#16a34a"},
                )
                fig.update_layout(
                    height=480,
                    xaxis_title="Distance from ศาลากลาง (km)",
                    yaxis_title="กล้าธรรม constituency vote share",
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig, width="stretch")

                # Extract regression stats from trendline
                try:
                    from scipy import stats as sp_stats
                    slope, intercept, r_val, p_val, _ = sp_stats.linregress(
                        reg_df["distance_to_center_km"].values, reg_df[share_col].values
                    )
                    story_card(
                        f"Distance effect: r² = {r_val**2:.3f}, p = {p_val:.4f}",
                        f"Every 5 km further from city center, กล้าธรรม share {'increases' if slope > 0 else 'decreases'} "
                        f"by ~{abs(slope * 5):.1%}. {'The local party thrives at the periphery.' if slope > 0 else 'The local party is stronger in urban areas.'}",
                    )
                except ImportError:
                    pass

        # PCA cluster visualization
        if "pca_x" in ms_valid.columns and "pca_y" in ms_valid.columns:
            pca_df = ms_valid[ms_valid["pca_x"].notna()].copy()
            if not pca_df.empty:
                fig = px.scatter(
                    pca_df,
                    x="pca_x",
                    y="pca_y",
                    color="cluster_label",
                    text="official_subdistrict",
                    size="n_paired_units" if "n_paired_units" in pca_df.columns else None,
                    hover_data=["distance_to_center_km", "urbanization_index", "split_rate"],
                    title="Multi-source tambon clustering (PCA projection)",
                    color_discrete_map={"Urban Core": "#dc2626", "Peri-urban Swing": "#f97316", "Rural Loyal": "#16a34a"},
                )
                fig.update_traces(textposition="top center", textfont_size=10)
                var_expl = pca_df["pca_var_explained"].iloc[0] if "pca_var_explained" in pca_df.columns else np.nan
                fig.update_layout(
                    height=540,
                    xaxis_title=f"PC1",
                    yaxis_title=f"PC2",
                    legend_title_text="Cluster",
                )
                if pd.notna(var_expl):
                    fig.add_annotation(
                        text=f"PCA explains {var_expl:.0%} of variance",
                        xref="paper", yref="paper", x=0.02, y=0.98,
                        showarrow=False, font=dict(size=11, color="#94a3b8"),
                    )
                st.plotly_chart(fig, width="stretch")

        # Cluster profile radar chart
        profile_cols = {"distance_to_center_km": "Distance (km)", "urbanization_index": "Urbanization",
                        "split_rate": "Split-ticket rate"}
        # Add top constituency party shares
        for c in ms_valid.columns:
            if c.startswith("constituency_share_") and ms_valid[c].notna().sum() > 0:
                party_name = c.replace("constituency_share_", "").replace("พรรค", "")
                if ms_valid[c].mean() > 0.03:
                    profile_cols[c] = party_name
        if len(profile_cols) >= 3:
            radar_data = []
            for cl, sub in ms_valid.groupby("cluster_label"):
                for col, label in profile_cols.items():
                    if col in sub.columns:
                        val = sub[col].mean()
                        radar_data.append({"cluster": cl, "feature": label, "value": val})
            radar_df = pd.DataFrame(radar_data)
            if not radar_df.empty:
                # Normalize to 0-1 for radar
                for feat in radar_df["feature"].unique():
                    mask = radar_df["feature"] == feat
                    vals = radar_df.loc[mask, "value"]
                    vmin, vmax = vals.min(), vals.max()
                    if vmax > vmin:
                        radar_df.loc[mask, "value_norm"] = (vals - vmin) / (vmax - vmin)
                    else:
                        radar_df.loc[mask, "value_norm"] = 0.5
                # Use explicit color map to match PCA/scatter charts
                cluster_color_map = {
                    "Urban Core": "#dc2626",
                    "Peri-urban Swing": "#f97316",
                    "Rural Loyal": "#16a34a",
                }
                fig = px.line_polar(
                    radar_df,
                    r="value_norm",
                    theta="feature",
                    color="cluster",
                    line_close=True,
                    title="Cluster profiles: what defines each community type?",
                    color_discrete_map=cluster_color_map,
                )
                fig.update_traces(fill="toself", opacity=0.3)
                fig.update_layout(height=500, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                st.plotly_chart(fig, width="stretch")

        story_card(
            "The election is shaped by geography",
            "Urban tambons close to city center show high split-ticket rates and competitive multi-party races. "
            "Rural tambons at the periphery concentrate behind กล้าธรรม with minimal ticket-splitting. "
            "This is not just a party story — it is an urban-rural divide measured through non-election data.",
        )
    else:
        st.info("Run `python 05_analysis/enrich_external.py` to generate multi-source features.")

    story_bridge("Finally, the dashboard points to the exact units that can make or break the interpretation.")

    story_chapter(
        "Act 8",
        "The ending is an evidence trail: unusual units become review targets, not unsupported claims.",
        "The fingerprint map turns high-dimensional vote shares into a visual audit queue. Isolated points are the best places to check images, OCR, and local context.",
    )
    evidence_path = FIGURES_DIR / "evidence_fingerprint_map.csv"
    if evidence_path.exists():
        evidence_story = add_display_columns(pd.read_csv(evidence_path))
        evidence_story = evidence_story[evidence_story["ballot_kind"].isin(kind_sel)]
        if not evidence_story.empty:
            fig = px.scatter(
                evidence_story,
                x="fingerprint_x",
                y="fingerprint_y",
                color="winner_party",
                color_discrete_map=party_color_map(evidence_story["winner_party"]),
                symbol="quality_label",
                size="total_ballots",
                hover_data=["unit_label", "area_label", "ballot_kind", "winner_party", "winner_share", "review_reason", "file_label"],
                title="Evidence fingerprint: where the story should be checked against source images",
            )
            fig.update_layout(height=600, legend_title_text="Winner / quality")
            st.plotly_chart(fig, width="stretch")
            center_x = evidence_story["fingerprint_x"].median()
            center_y = evidence_story["fingerprint_y"].median()
            ranked = evidence_story.assign(
                distance=((evidence_story["fingerprint_x"] - center_x) ** 2 + (evidence_story["fingerprint_y"] - center_y) ** 2) ** 0.5
            ).sort_values("distance", ascending=False)
            presentation_table(
                "Open the exact evidence rows behind the most unusual points",
                ranked[["unit_label", "area_label", "ballot_kind", "winner_party", "winner_share", "quality_label", "review_reason", "file_label"]].head(12),
                presentation_mode,
            )
    else:
        st.info("Run `python 05_analysis\\analysis.py` to generate the evidence fingerprint map.")

with tab_lookup["Evidence"]:
    st.subheader("Story Map: vote-share fingerprints")
    story_card(
        "What this shows",
        "Each point is one ballot record projected from its party vote-share fingerprint. Clusters are similar vote patterns; isolated points are either politically unusual units or OCR records needing image review.",
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
                color_discrete_map=party_color_map(evidence["winner_party"]),
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
            presentation_table(
                "Open outlier evidence table",
                ranked[
                    [
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
                presentation_mode,
            )

if "Quality Gate" in tab_lookup:
  with tab_lookup["Quality Gate"]:
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
    presentation_table(
        "Open review queue",
        display_table(df.loc[~df["is_confirmed"], flag_cols]),
        presentation_mode,
    )

if "Summary" in tab_lookup:
  with tab_lookup["Summary"]:
    st.subheader("Executive summary")
    story_card(
        "How to read this dashboard",
        "Use Split Ticket for the headline pattern, Story Map for outliers and OCR evidence, Parties for vote-share standings, and Drilldown when you need to inspect a specific polling unit.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered rows", f"{len(df_f):,}")
    c2.metric("Votes", f"{long_f['votes'].sum():,.0f}" if not long_f.empty else "0")
    c3.metric("Total ballots", f"{df_f.get('total_ballots', pd.Series(dtype=float)).sum():,.0f}")
    invalid = df_f.get("bad_ballots", pd.Series(dtype=float)).sum()
    total = df_f.get("total_ballots", pd.Series(dtype=float)).sum()
    c4.metric("Invalid share", f"{invalid / total * 100:.2f}%" if total else "N/A")

    if "total_ballots" in df_f.columns and not df_f.empty:
        st.plotly_chart(px.histogram(df_f, x="total_ballots", color="ballot_kind", nbins=30), width="stretch")


if "Split Ticket" in tab_lookup:
  with tab_lookup["Split Ticket"]:
    st.subheader("Split-ticket and dominance insights")
    story_card(
        "Headline",
        "This page compares who wins the constituency ballot against who wins the party-list ballot in the same polling unit. It reveals where candidate strength and party-list preference diverge.",
    )
    insight = load_insight_tables()
    splits = insight["splits"].copy()
    matrix = insight["matrix"].copy()
    dominance = insight["dominance"].copy()
    strongholds = insight["strongholds"].copy()
    competitiveness = insight["competitiveness"].copy()

    if splits.empty or matrix.empty:
        st.info("Run `python 05_analysis\\analysis.py` to generate insight artifacts.")
    else:
        splits = add_display_columns(splits)
        if station_query:
            searchable = (
                splits["polling_unit_id"].astype(str)
                + " "
                + _series(splits, "unit_label", "").astype(str)
                + " "
                + _series(splits, "area_label", "").astype(str)
                + " "
                + _series(splits, "file_label", "").astype(str)
            )
            splits = splits[searchable.str.contains(station_query, case=False, na=False)]

        split_rate = splits["split_ticket"].mean() if "split_ticket" in splits.columns and len(splits) else np.nan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Units with both ballots", f"{len(splits):,}")
        c2.metric("Split-ticket units", f"{int(splits.get('split_ticket', pd.Series(False)).sum()):,}")
        c3.metric("Split-ticket rate", f"{split_rate * 100:.1f}%" if pd.notna(split_rate) else "N/A")
        c4.metric("Winner pair patterns", f"{len(matrix):,}")
        if not matrix.empty:
            lead = matrix.sort_values("polling_units", ascending=False).iloc[0]
            story_card(
                "Dominant flow",
                f"The most common pattern is {lead['winner_party_constituency']} on constituency ballots flowing to {lead['winner_party_party_list']} on party-list ballots across {int(lead['polling_units']):,} polling units.",
            )

        st.subheader("Split-ticket flow: constituency winner -> party-list winner")
        st.plotly_chart(
            split_flow_figure(matrix, limit=25, title="Split-ticket flow: constituency winner -> party-list winner"),
            width="stretch",
        )

        st.subheader("Dominance frontier")
        if not splits.empty:
            splits["split_label"] = np.where(splits["split_ticket"], "split winner", "same winner")
            fig = px.scatter(
                splits,
                x="winner_share_constituency",
                y="winner_share_party_list",
                color="split_label",
                size="total_ballots" if "total_ballots" in splits.columns else None,
                hover_data=[
                    "unit_label",
                    "area_label",
                    "winner_party_constituency",
                    "winner_party_party_list",
                    "winner_share_constituency",
                    "winner_share_party_list",
                    "winner_margin_share_constituency",
                    "winner_margin_share_party_list",
                    "file_label",
                ],
                title="How strongly each ballot winner dominated the same polling unit",
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(148,163,184,0.7)", dash="dash"))
            fig.update_layout(
                height=650,
                xaxis_tickformat=".0%",
                yaxis_tickformat=".0%",
                xaxis_title="Constituency winner share",
                yaxis_title="Party-list winner share",
                legend_title_text="Winner relationship",
            )
            st.plotly_chart(fig, width="stretch")

        if not dominance.empty:
            st.subheader("Winner dominance by ballot type")
            dominance = add_display_columns(dominance)
            dominance = dominance[dominance["ballot_kind"].isin(kind_sel)]
            if station_query:
                searchable = (
                    dominance["polling_unit_id"].astype(str)
                    + " "
                    + _series(dominance, "unit_label", "").astype(str)
                    + " "
                    + _series(dominance, "area_label", "").astype(str)
                    + " "
                    + _series(dominance, "file_label", "").astype(str)
                )
                dominance = dominance[searchable.str.contains(station_query, case=False, na=False)]
            fig = px.box(
                dominance,
                x="winner_party",
                y="winner_margin_share",
                color="winner_party",
                color_discrete_map=party_color_map(dominance["winner_party"]),
                points="outliers",
                hover_data=["unit_label", "ballot_kind", "runner_up_party", "winner_share", "runner_up_share", "source_file"],
                title="How decisive are each party's wins?",
            )
            fig.update_layout(height=560, yaxis_tickformat=".0%", xaxis_title="Winner party", yaxis_title="Winner margin")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Strongholds and surprising pockets")
        if strongholds.empty:
            st.info("No stronghold rows at current thresholds.")
        else:
            strongholds = add_display_columns(strongholds)
            strongholds = strongholds[strongholds["ballot_kind"].isin(kind_sel)]
            cols = [
                c
                for c in [
                    "unit_label",
                    "area_label",
                    "ballot_kind",
                    "winner_party",
                    "winner_share",
                    "runner_up_party",
                    "runner_up_share",
                    "winner_margin_share",
                    "winner_votes",
                    "winner_margin_votes",
                    "source_file",
                ]
                if c in strongholds.columns
            ]
            presentation_table(
                "Open stronghold table",
                strongholds.sort_values("winner_margin_share", ascending=False)[cols].head(40),
                presentation_mode,
            )

        if not competitiveness.empty:
            st.subheader("Competitiveness landscape")
            competitiveness = add_display_columns(competitiveness.merge(
                df.drop_duplicates("polling_unit_id")[
                    [c for c in ["polling_unit_id", "source_file", "station_id"] if c in df.columns]
                ],
                on="polling_unit_id",
                how="left",
            ))
            competitiveness = competitiveness[competitiveness["ballot_kind"].isin(kind_sel)]
            fig = px.histogram(
                competitiveness,
                x="effective_number_of_parties",
                color="ballot_kind",
                nbins=30,
                title="Effective number of parties per polling unit",
            )
            fig.update_layout(height=430, xaxis_title="Effective number of parties")
            st.plotly_chart(fig, width="stretch")


if "Parties" in tab_lookup:
  with tab_lookup["Parties"]:
    if long_f.empty:
        st.info("No party rows after filters.")
    else:
        agg = (
            long_f.groupby(["ballot_kind", "party"], as_index=False)
            .agg(votes=("votes", "sum"), stations=("polling_unit_id", "nunique"), mean_share=("vote_share", "mean"))
            .sort_values(["ballot_kind", "votes"], ascending=[True, False])
        )
        agg["share_pct"] = agg["votes"] / agg.groupby("ballot_kind")["votes"].transform("sum") * 100
        presentation_table("Open party performance table", agg, presentation_mode)
        for kind, sub in agg.groupby("ballot_kind"):
            top = sub.head(top_n)
            fig = px.bar(
                top.iloc[::-1],
                x="share_pct",
                y="party",
                color="party",
                color_discrete_map=party_color_map(top["party"]),
                orientation="h",
                title=f"Top {top_n} - {kind}",
                hover_data=["votes", "stations", "mean_share"],
            )
            fig.update_layout(height=max(420, 28 * len(top)))
            st.plotly_chart(fig, width="stretch")

if "Geo Map" in tab_lookup:
  with tab_lookup["Geo Map"]:
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
        presentation_table(
            "Open location QA table",
            display_table(loc_view[qa_cols]),
            presentation_mode,
        )

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
                        color_discrete_map=party_color_map(winners["party"]),
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
                        fig = scatter_map(
                            map_ready_tambon,
                            lat="map_lat",
                            lon="map_lon",
                            color="party",
                            color_discrete_map=party_color_map(map_ready_tambon["party"]),
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
                        fig.update_layout(margin=dict(l=0, r=0, t=45, b=0))
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
                presentation_table("Open draft geographic grouping table", summary, presentation_mode)

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
                fig = scatter_map(
                    map_ready,
                    lat="lat",
                    lon="lon",
                    color="winner_party",
                    color_discrete_map=party_color_map(map_ready["winner_party"]),
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
                fig.update_layout(margin=dict(l=0, r=0, t=45, b=0))
                st.plotly_chart(fig, width="stretch")

if "Spatial" in tab_lookup:
  with tab_lookup["Spatial"]:
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
            color_discrete_map=party_color_map(spatial[color_by]) if color_by == "winner_party" else None,
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
        presentation_table("Open pseudo-spatial data table", display_table(spatial), presentation_mode)

if "Networks" in tab_lookup:
  with tab_lookup["Networks"]:
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

with tab_lookup["Drilldown"]:
    station_sel = st.selectbox(
        "Polling unit",
        station_options,
        index=0,
        format_func=lambda value: station_label_lookup.get(value, value),
    )
    st_df = df[df["polling_unit_id"].astype(str) == station_sel]
    st_long = long[long["polling_unit_id"].astype(str) == station_sel] if not long.empty else long
    presentation_table("Open polling-unit source records", display_table(st_df), presentation_mode, expanded=not presentation_mode)
    if not st_long.empty:
        for kind, sub in st_long.groupby("ballot_kind"):
            agg = sub.groupby("party", as_index=False)["votes"].sum().sort_values("votes", ascending=False)
            top = agg.head(25).iloc[::-1]
            st.plotly_chart(
                px.bar(
                    top,
                    x="votes",
                    y="party",
                    color="party",
                    color_discrete_map=party_color_map(top["party"]),
                    orientation="h",
                    title=kind,
                ),
                width="stretch",
            )

if "Outliers" in tab_lookup:
  with tab_lookup["Outliers"]:
    anomaly_path = FIGURES_DIR / "anomaly_records.csv"
    if anomaly_path.exists():
        anomalies = pd.read_csv(anomaly_path)
        presentation_table("Open anomaly records table", display_table(anomalies), presentation_mode)
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
        presentation_table("Open quick MAD flagged rows", display_table(quick), presentation_mode)

if "Data Lab" in tab_lookup:
  with tab_lookup["Data Lab"]:
    st.subheader("Filtered records")
    if presentation_mode:
        st.info("Raw tables are hidden by default in presentation mode. Use the expanders below for audit/export.")
    presentation_table("Open filtered records table", display_table(df_f), presentation_mode)
    st.download_button(
        "Download filtered records",
        df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_election_records.csv",
        mime="text/csv",
    )
    if not long_f.empty:
        st.subheader("Long party rows")
        presentation_table("Open long party rows table", display_table(long_f), presentation_mode)
        st.download_button(
            "Download long party rows",
            long_f.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_party_long.csv",
            mime="text/csv",
        )
