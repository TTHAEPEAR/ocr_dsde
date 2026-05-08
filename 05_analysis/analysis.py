"""
Research-grade election analysis.

The analysis is deliberately quality-aware: confirmed OCR rows are separated
from rows that still need review, vote-share tests use station-level shares
instead of raw counts, and every exported table carries stable polling-unit
keys so downstream reports can trace claims back to source images.

Usage:
    python 05_analysis/analysis.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, FIGURES_DIR, OCR_RAW_DIR, PARTY_NAMES

plt.rcParams["font.family"] = ["Tahoma", "DejaVu Sans", "Arial"]

ADVANCE_FORMS = {"5_16", "5_16_party", "5_17", "5_17_party"}
OUTPUT_DIR = FIGURES_DIR


def _to_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series(df: pd.DataFrame, column: str, default="") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _source_path() -> Path:
    candidates = [
        CLEANED_DIR / "election_results_cleaned.csv",
        OCR_RAW_DIR / "raw_all_forms_split.csv",
        OCR_RAW_DIR / "raw_election_split.csv",
        OCR_RAW_DIR / "raw_election_split_checkpoint.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No analysis input found. Run OCR/clean first.")


def load_election_data() -> tuple[pd.DataFrame, Path]:
    path = _source_path()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")

    if "polling_unit_id" not in df.columns:
        df["polling_unit_id"] = (
            df.get("source_file", pd.Series(range(len(df)), index=df.index))
            .astype(str)
            .str.replace(r"\.pdf$", "", regex=True)
        )
    if "ballot_record_id" not in df.columns:
        kind_key = _series(df, "ballot_kind", None).fillna(_series(df, "form_type", "unknown"))
        df["ballot_record_id"] = df["polling_unit_id"].astype(str) + "__" + kind_key.astype(str)
    if "ballot_kind" not in df.columns:
        form_type = _series(df, "form_type", "")
        df["ballot_kind"] = np.where(
            form_type.astype(str).str.endswith("_party"),
            "party_list",
            "constituency",
        )

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
            df[col] = _num(df[col])

    for col in [
        "ocr_success",
        "needs_review",
        "vote_sum_match",
        "vote_sum_match_good_ballots",
        "vote_sum_match_total_votes",
        "summary_votes_match",
        "ballot_sum_match",
    ]:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _to_bool(v, default=False) if pd.notna(v) else np.nan)

    if "ocr_success" not in df.columns:
        df["ocr_success"] = True
    if "needs_review" not in df.columns:
        df["needs_review"] = False
    if "vote_sum_match" not in df.columns:
        df["vote_sum_match"] = df.get("votes_sum", 0).eq(df.get("good_ballots", -1))
    if "ballot_sum_match" not in df.columns:
        df["ballot_sum_match"] = (
            df.get("good_ballots", 0) + df.get("bad_ballots", 0) + df.get("no_vote_ballots", 0)
        ).eq(df.get("total_ballots", -1))

    df["is_confirmed"] = (
        df["ocr_success"].fillna(False).astype(bool)
        & ~df["needs_review"].fillna(True).astype(bool)
        & df["vote_sum_match"].fillna(False).astype(bool)
        & df["ballot_sum_match"].fillna(False).astype(bool)
    )
    df["review_reason"] = df.apply(review_reason, axis=1)
    return df, path


def review_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not _to_bool(row.get("ocr_success", True), default=True):
        reasons.append("ocr_failed")
    if _to_bool(row.get("needs_review", False), default=False):
        reasons.append("needs_review")
    if not _to_bool(row.get("vote_sum_match", True), default=True):
        reasons.append("vote_sum_mismatch")
    if not _to_bool(row.get("ballot_sum_match", True), default=True):
        reasons.append("ballot_sum_mismatch")
    if pd.notna(row.get("summary_votes_match")) and not _to_bool(row.get("summary_votes_match")):
        reasons.append("good_vs_total_votes_mismatch")
    return ";".join(dict.fromkeys(reasons))


def build_party_long(df: pd.DataFrame) -> pd.DataFrame:
    cand_re = re.compile(r"^candidate_(\d+)_votes$")
    cand_nums = sorted({int(m.group(1)) for c in df.columns if (m := cand_re.match(c))})
    rows: list[dict] = []
    for _, row in df.iterrows():
        ballot_kind = str(row.get("ballot_kind", ""))
        is_party_list = ballot_kind == "party_list" or str(row.get("form_type", "")).endswith("_party")
        good = row.get("good_ballots", np.nan)
        for n in cand_nums:
            v_col = f"candidate_{n}_votes"
            if v_col not in df.columns:
                continue
            votes = pd.to_numeric(pd.Series([row.get(v_col)]), errors="coerce").iloc[0]
            if pd.isna(votes) or votes == 0:
                continue
            if is_party_list:
                party = PARTY_NAMES.get(n, f"party_no_{n}")
            else:
                party = str(row.get(f"candidate_{n}_party", "") or "").strip() or f"candidate_no_{n}"
            rows.append(
                {
                    "source_file": row.get("source_file", ""),
                    "polling_unit_id": row.get("polling_unit_id"),
                    "ballot_record_id": row.get("ballot_record_id"),
                    "station_id": row.get("station_id"),
                    "form_type": row.get("form_type", ""),
                    "ballot_kind": ballot_kind,
                    "is_party_list": is_party_list,
                    "is_advance": row.get("form_type", "") in ADVANCE_FORMS,
                    "candidate_number": n,
                    "party": party,
                    "votes": float(votes),
                    "good_ballots": good,
                    "vote_share": float(votes / good) if pd.notna(good) and good > 0 else np.nan,
                    "is_confirmed": bool(row.get("is_confirmed", False)),
                    "needs_review": bool(row.get("needs_review", False)),
                    "review_reason": row.get("review_reason", ""),
                }
            )
    return pd.DataFrame(rows)


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mad_zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def hhi(shares: pd.Series) -> float:
    shares = shares.dropna().astype(float)
    return float((shares**2).sum()) if len(shares) else np.nan


def enp(shares: pd.Series) -> float:
    val = hhi(shares)
    return float(1 / val) if val and val > 0 else np.nan


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


class ElectionAnalyzer:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.df, self.input_path = load_election_data()
        self.long = build_party_long(self.df)
        self.confirmed = self.df[self.df["is_confirmed"]].copy()
        self.long_confirmed = self.long[self.long["is_confirmed"]].copy()

    def run_all(self) -> None:
        self.data_quality()
        self.party_performance()
        self.polling_unit_summary()
        self.cross_ballot_consistency()
        self.anomaly_detection()
        self.advance_vs_day_test()
        self.station_clustering()
        self.pseudo_spatial_analysis()
        self.network_analysis()
        self.write_report()
        logger.info(f"Research analysis complete: {OUTPUT_DIR}")

    def data_quality(self) -> None:
        rows = []
        for name, sub in [
            ("all", self.df),
            ("confirmed", self.confirmed),
            ("needs_review", self.df[~self.df["is_confirmed"]]),
        ]:
            rows.append(
                {
                    "slice": name,
                    "records": len(sub),
                    "polling_units": sub["polling_unit_id"].nunique(),
                    "constituency_records": int((sub["ballot_kind"] == "constituency").sum()),
                    "party_list_records": int((sub["ballot_kind"] == "party_list").sum()),
                    "ocr_success_rate": sub["ocr_success"].mean(),
                    "vote_sum_match_rate": sub["vote_sum_match"].mean(),
                    "ballot_sum_match_rate": sub["ballot_sum_match"].mean(),
                    "confirmed_rate": sub["is_confirmed"].mean() if len(sub) else np.nan,
                }
            )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
        self.df[~self.df["is_confirmed"]].to_csv(OUTPUT_DIR / "review_records.csv", index=False)

    def party_performance(self) -> None:
        if self.long.empty:
            return
        outputs = []
        for label, long_df in [("all", self.long), ("confirmed", self.long_confirmed)]:
            if long_df.empty:
                continue
            agg = (
                long_df.groupby(["ballot_kind", "party"], as_index=False)
                .agg(
                    votes=("votes", "sum"),
                    stations=("polling_unit_id", "nunique"),
                    mean_station_share=("vote_share", "mean"),
                    median_station_share=("vote_share", "median"),
                )
                .sort_values(["ballot_kind", "votes"], ascending=[True, False])
            )
            totals = agg.groupby("ballot_kind")["votes"].transform("sum")
            agg["vote_share"] = agg["votes"] / totals.replace(0, np.nan)
            ci = [wilson_ci(v, n) for v, n in zip(agg["votes"], totals)]
            agg["share_ci_low"] = [c[0] for c in ci]
            agg["share_ci_high"] = [c[1] for c in ci]
            agg["quality_slice"] = label
            outputs.append(agg)
        out = pd.concat(outputs, ignore_index=True)
        out.to_csv(OUTPUT_DIR / "party_performance.csv", index=False)

        confirmed = out[out["quality_slice"] == "confirmed"]
        for kind, sub in confirmed.groupby("ballot_kind"):
            top = sub.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11, max(5, len(top) * 0.35)))
            ax.barh(top["party"], top["vote_share"] * 100, color="#2563eb")
            ax.set_xlabel("Vote share (%)")
            ax.set_title(f"Confirmed vote share - {kind}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"party_share_confirmed_{kind}.png", dpi=160)
            plt.close()

    def polling_unit_summary(self) -> None:
        if self.df.empty:
            return
        rows = []
        for (unit, kind), sub in self.df.groupby(["polling_unit_id", "ballot_kind"], dropna=False):
            r = sub.iloc[-1]
            rows.append(
                {
                    "polling_unit_id": unit,
                    "ballot_kind": kind,
                    "station_id": r.get("station_id"),
                    "source_file": r.get("source_file"),
                    "is_confirmed": r.get("is_confirmed"),
                    "review_reason": r.get("review_reason"),
                    "good_ballots": r.get("good_ballots"),
                    "bad_ballots": r.get("bad_ballots"),
                    "no_vote_ballots": r.get("no_vote_ballots"),
                    "total_ballots": r.get("total_ballots"),
                    "votes_sum": r.get("votes_sum"),
                    "total_votes_sum": r.get("total_votes_sum"),
                    "vote_sum_match_good_ballots": r.get("vote_sum_match_good_ballots"),
                    "vote_sum_match_total_votes": r.get("vote_sum_match_total_votes"),
                    "summary_votes_match": r.get("summary_votes_match"),
                    "ballot_sum_match": r.get("ballot_sum_match"),
                    "invalid_ballot_ratio": (
                        r.get("bad_ballots") / r.get("total_ballots")
                        if pd.notna(r.get("bad_ballots")) and pd.notna(r.get("total_ballots")) and r.get("total_ballots") > 0
                        else np.nan
                    ),
                    "blank_ballot_ratio": (
                        r.get("no_vote_ballots") / r.get("total_ballots")
                        if pd.notna(r.get("no_vote_ballots")) and pd.notna(r.get("total_ballots")) and r.get("total_ballots") > 0
                        else np.nan
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "polling_unit_summary.csv", index=False)

    def cross_ballot_consistency(self) -> None:
        cols = ["polling_unit_id", "ballot_kind", "good_ballots", "total_ballots", "is_confirmed"]
        if not set(cols).issubset(self.df.columns):
            return
        wide = self.df[cols].pivot_table(
            index="polling_unit_id",
            columns="ballot_kind",
            values=["good_ballots", "total_ballots", "is_confirmed"],
            aggfunc="last",
        )
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        if {"total_ballots_constituency", "total_ballots_party_list"}.issubset(wide.columns):
            wide["total_ballots_delta"] = wide["total_ballots_constituency"] - wide["total_ballots_party_list"]
        if {"good_ballots_constituency", "good_ballots_party_list"}.issubset(wide.columns):
            wide["good_ballots_delta"] = wide["good_ballots_constituency"] - wide["good_ballots_party_list"]
        wide.to_csv(OUTPUT_DIR / "cross_ballot_consistency.csv", index=False)

    def anomaly_detection(self) -> None:
        rows = []
        base = self.confirmed if not self.confirmed.empty else self.df
        metrics = ["total_ballots", "good_ballots", "bad_ballots", "no_vote_ballots"]
        for kind, sub in base.groupby("ballot_kind"):
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                z = mad_zscore(sub[metric])
                flagged = sub.loc[z.abs() > 3.5]
                for idx, r in flagged.iterrows():
                    rows.append(
                        {
                            "polling_unit_id": r.get("polling_unit_id"),
                            "ballot_kind": kind,
                            "metric": metric,
                            "value": r.get(metric),
                            "mad_z": float(z.loc[idx]),
                            "source_file": r.get("source_file"),
                        }
                    )

        if not self.long_confirmed.empty:
            party_totals = self.long_confirmed.groupby("party")["votes"].sum()
            top_parties = party_totals.sort_values(ascending=False).head(10).index
            for (kind, party), sub in self.long_confirmed[
                self.long_confirmed["party"].isin(top_parties)
            ].groupby(["ballot_kind", "party"]):
                z = mad_zscore(sub["vote_share"])
                flagged = sub.loc[z.abs() > 3.5]
                for idx, r in flagged.iterrows():
                    rows.append(
                        {
                            "polling_unit_id": r.get("polling_unit_id"),
                            "ballot_kind": kind,
                            "metric": f"vote_share:{party}",
                            "value": r.get("vote_share"),
                            "mad_z": float(z.loc[idx]),
                            "source_file": r.get("source_file"),
                        }
                    )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "anomaly_records.csv", index=False)

    def advance_vs_day_test(self) -> None:
        if self.long_confirmed.empty:
            return
        rows = []
        for (kind, party), sub in self.long_confirmed.groupby(["ballot_kind", "party"]):
            adv = sub.loc[sub["is_advance"], "vote_share"].dropna()
            day = sub.loc[~sub["is_advance"], "vote_share"].dropna()
            if len(adv) < 3 or len(day) < 3:
                continue
            u, p = stats.mannwhitneyu(adv, day, alternative="two-sided")
            rows.append(
                {
                    "ballot_kind": kind,
                    "party": party,
                    "n_advance": len(adv),
                    "n_day": len(day),
                    "median_share_advance": float(adv.median()),
                    "median_share_day": float(day.median()),
                    "effect_delta_median": float(adv.median() - day.median()),
                    "u_statistic": float(u),
                    "p_value": float(p),
                }
            )
        out = pd.DataFrame(rows)
        if not out.empty:
            out["p_bh_fdr"] = self._bh_adjust(out["p_value"])
            out.sort_values("p_value").to_csv(OUTPUT_DIR / "advance_vs_day_share_tests.csv", index=False)

    @staticmethod
    def _bh_adjust(pvals: pd.Series) -> pd.Series:
        p = pvals.astype(float).to_numpy()
        order = np.argsort(p)
        ranked = np.empty_like(p)
        n = len(p)
        prev = 1.0
        for i in range(n - 1, -1, -1):
            rank = i + 1
            val = min(prev, p[order[i]] * n / rank)
            ranked[order[i]] = val
            prev = val
        return pd.Series(ranked, index=pvals.index).clip(upper=1.0)

    def station_clustering(self) -> None:
        if self.long_confirmed.empty:
            return
        for kind, sub in self.long_confirmed.groupby("ballot_kind"):
            pivot = sub.pivot_table(
                index="polling_unit_id", columns="party", values="votes", aggfunc="sum", fill_value=0
            )
            if len(pivot) < 4:
                continue
            pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            n_clusters = min(4, max(2, len(pivot) // 8))
            scaled = StandardScaler().fit_transform(pivot)
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(scaled)
            out = pivot.copy()
            out["cluster"] = labels
            out.to_csv(OUTPUT_DIR / f"station_clusters_{kind}.csv")

            coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=45, alpha=0.8)
            ax.set_title(f"Station clusters by confirmed party-share profile - {kind}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(sc, ax=ax, label="cluster")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"station_clusters_{kind}.png", dpi=160)
            plt.close()

    def pseudo_spatial_analysis(self) -> None:
        if self.long_confirmed.empty:
            return
        top_party = (
            self.long_confirmed.sort_values("votes", ascending=False)
            .groupby(["polling_unit_id", "ballot_kind"], as_index=False)
            .first()
        )
        totals = self.long_confirmed.groupby(["polling_unit_id", "ballot_kind"], as_index=False).agg(
            total_party_votes=("votes", "sum"),
            party_count=("party", "nunique"),
        )
        rows = top_party.merge(totals, on=["polling_unit_id", "ballot_kind"], how="left", suffixes=("", "_total"))
        rows["locality"] = rows["polling_unit_id"].map(locality_from_unit)
        rows["unit_number"] = rows.apply(lambda r: unit_number(r["polling_unit_id"], r.get("station_id")), axis=1)
        rows["layout_x"] = rows["unit_number"]
        missing_x = rows["layout_x"].isna()
        if missing_x.any():
            rows.loc[missing_x, "layout_x"] = rows[missing_x].groupby("locality").cumcount() + 1
        locality_order = {name: i for i, name in enumerate(sorted(rows["locality"].dropna().unique()))}
        rows["layout_y"] = rows["locality"].map(locality_order).astype(float)
        rows["winner_party"] = rows["party"]
        rows["winner_votes"] = rows["votes"]
        rows["winner_share"] = rows["vote_share"]
        cols = [
            "polling_unit_id",
            "ballot_kind",
            "locality",
            "unit_number",
            "layout_x",
            "layout_y",
            "winner_party",
            "winner_votes",
            "winner_share",
            "total_party_votes",
            "party_count",
            "source_file",
        ]
        rows[cols].to_csv(OUTPUT_DIR / "pseudo_spatial_units.csv", index=False)

        for kind, sub in rows.groupby("ballot_kind"):
            fig, ax = plt.subplots(figsize=(11, max(5, sub["locality"].nunique() * 0.7)))
            labels = {party: i for i, party in enumerate(sub["winner_party"].dropna().unique())}
            colors = sub["winner_party"].map(labels)
            sizes = (sub["winner_share"].fillna(0.05) * 500).clip(lower=30, upper=350)
            sc = ax.scatter(sub["layout_x"], sub["layout_y"], c=colors, s=sizes, cmap="tab20", alpha=0.85)
            ax.set_yticks(list(locality_order.values()))
            ax.set_yticklabels(list(locality_order.keys()))
            ax.set_xlabel("Polling unit order within source/locality")
            ax.set_title(f"Pseudo-spatial winner layout - {kind}")
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"pseudo_spatial_winners_{kind}.png", dpi=160)
            plt.close()

    def network_analysis(self) -> None:
        if self.long_confirmed.empty:
            return
        node_rows = []
        edge_rows = []
        for kind, sub in self.long_confirmed.groupby("ballot_kind"):
            pivot = sub.pivot_table(
                index="polling_unit_id", columns="party", values="vote_share", aggfunc="sum", fill_value=0
            )
            top_parties = sub.groupby("party")["votes"].sum().sort_values(ascending=False).head(18).index
            pivot = pivot.reindex(columns=top_parties, fill_value=0)
            if pivot.empty or pivot.shape[1] < 2:
                continue

            party_totals = sub.groupby("party").agg(votes=("votes", "sum"), stations=("polling_unit_id", "nunique"))
            for party in pivot.columns:
                node_rows.append(
                    {
                        "node_id": f"{kind}::party::{party}",
                        "label": party,
                        "node_type": "party",
                        "ballot_kind": kind,
                        "votes": float(party_totals.loc[party, "votes"]) if party in party_totals.index else 0,
                        "stations": int(party_totals.loc[party, "stations"]) if party in party_totals.index else 0,
                    }
                )

            corr = pivot.corr(method="spearman").fillna(0)
            for i, source in enumerate(corr.columns):
                for target in corr.columns[i + 1 :]:
                    weight = float(corr.loc[source, target])
                    if abs(weight) >= 0.35:
                        edge_rows.append(
                            {
                                "source": f"{kind}::party::{source}",
                                "target": f"{kind}::party::{target}",
                                "source_label": source,
                                "target_label": target,
                                "edge_type": "party_similarity",
                                "ballot_kind": kind,
                                "weight": weight,
                                "abs_weight": abs(weight),
                            }
                        )

            top_by_unit = sub.sort_values("vote_share", ascending=False).groupby("polling_unit_id").head(3)
            for _, row in top_by_unit.iterrows():
                unit_id = f"{kind}::unit::{row['polling_unit_id']}"
                node_rows.append(
                    {
                        "node_id": unit_id,
                        "label": row["polling_unit_id"],
                        "node_type": "polling_unit",
                        "ballot_kind": kind,
                        "votes": np.nan,
                        "stations": 1,
                    }
                )
                edge_rows.append(
                    {
                        "source": unit_id,
                        "target": f"{kind}::party::{row['party']}",
                        "source_label": row["polling_unit_id"],
                        "target_label": row["party"],
                        "edge_type": "unit_top_party",
                        "ballot_kind": kind,
                        "weight": float(row["vote_share"]),
                        "abs_weight": float(abs(row["vote_share"])),
                    }
                )

        nodes = pd.DataFrame(node_rows).drop_duplicates("node_id")
        edges = pd.DataFrame(edge_rows)
        nodes.to_csv(OUTPUT_DIR / "network_nodes.csv", index=False)
        edges.to_csv(OUTPUT_DIR / "network_edges.csv", index=False)

        sim = edges[edges["edge_type"] == "party_similarity"].copy()
        if not sim.empty:
            for kind, sub in sim.groupby("ballot_kind"):
                top = sub.sort_values("abs_weight", ascending=False).head(25).iloc[::-1]
                labels = top["source_label"] + " <-> " + top["target_label"]
                fig, ax = plt.subplots(figsize=(12, max(5, len(top) * 0.32)))
                ax.barh(labels, top["weight"], color=np.where(top["weight"] >= 0, "#2563eb", "#dc2626"))
                ax.axvline(0, color="#111827", linewidth=0.8)
                ax.set_xlabel("Spearman correlation of station vote-share profiles")
                ax.set_title(f"Party similarity network edges - {kind}")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"network_party_similarity_{kind}.png", dpi=160)
                plt.close()

    def write_report(self) -> None:
        quality = pd.read_csv(OUTPUT_DIR / "data_quality_summary.csv")
        perf_path = OUTPUT_DIR / "party_performance.csv"
        lines = [
            "# Election Research Analysis Report",
            "",
            f"Input: `{self.input_path}`",
            "",
            "## Data Quality",
            quality.to_markdown(index=False),
            "",
        ]
        if perf_path.exists():
            perf = pd.read_csv(perf_path)
            confirmed = perf[perf["quality_slice"] == "confirmed"].copy()
            if not confirmed.empty:
                confirmed["vote_share_pct"] = confirmed["vote_share"] * 100
                lines += [
                    "## Top Confirmed Party Shares",
                    confirmed.groupby("ballot_kind").head(10)[
                        ["ballot_kind", "party", "votes", "vote_share_pct", "stations"]
                    ].to_markdown(index=False),
                    "",
                ]
        lines += [
            "## Interpretation Rules",
            "- Treat confirmed rows as the main analytical set.",
            "- Treat needs_review rows as uncertainty, not final evidence.",
            "- Compare parties by station-level vote share when testing behavior across units.",
            "- Use `polling_unit_id` or `ballot_record_id` as keys; `station_id` alone can repeat.",
            "",
        ]
        (OUTPUT_DIR / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ElectionAnalyzer().run_all()
