"""
Phase 4: Research-grade party-aware analysis.

Reads cleaned OCR output, reshapes wide per-candidate columns into a long
party-level frame, then computes:
  - Party totals, vote share, 95% Wilson confidence intervals
  - Effective Number of Parties (Laakso-Taagepera), HHI, Gini
  - Turnout & spoilage descriptives
  - Mann-Whitney U test: advance vs election-day party shares
  - Robust MAD-based anomaly detection on station totals & invalid ratio
  - K-Means clustering of stations by party-share vector

Usage:
    python 05_analysis/analysis.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, FIGURES_DIR, REFERENCE_DIR, PARTY_NAMES

plt.rcParams["font.family"] = "Tahoma"

CONSTITUENCY_FORMS = {"5_18", "5_16", "5_17"}
PARTY_LIST_FORMS = {"5_18_party", "5_16_party", "5_17_party"}
ELECTION_DAY_FORMS = {"5_18", "5_18_party"}
ADVANCE_FORMS = {"5_16", "5_16_party", "5_17", "5_17_party"}


def build_party_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide candidate columns into long party-level rows.

    For constituency forms (5_18 etc.) the party label comes from the
    `candidate_<N>_party` column produced by Stage B. For party-list forms
    (5_18_party etc.) the candidate number maps to the national PARTY_NAMES.
    """
    cand_re = re.compile(r"^candidate_(\d+)_votes$")
    cand_nums = sorted({int(m.group(1)) for c in df.columns
                        if (m := cand_re.match(c))})
    rows = []
    for _, row in df.iterrows():
        form_type = str(row.get("form_type", ""))
        is_party_list = form_type.endswith("_party")
        for n in cand_nums:
            v_col = f"candidate_{n}_votes"
            p_col = f"candidate_{n}_party"
            if v_col not in row.index:
                continue
            v = row[v_col]
            if pd.isna(v) or v == 0:
                continue
            if is_party_list:
                party = PARTY_NAMES.get(n, f"พรรคเบอร์ {n}")
            else:
                party = row[p_col] if p_col in row.index and pd.notna(row[p_col]) else ""
                party = str(party).strip() or f"ผู้สมัครเบอร์ {n}"
            rows.append({
                "form_type": form_type,
                "is_party_list": is_party_list,
                "is_advance": form_type in ADVANCE_FORMS,
                "polling_unit_id": row.get("polling_unit_id", row.get("station_id")),
                "station_id": row.get("station_id"),
                "candidate_number": n,
                "party": party,
                "votes": float(v),
                "good_ballots": row.get("good_ballots", np.nan),
            })
    long_df = pd.DataFrame(rows)
    if not long_df.empty:
        long_df["votes"] = long_df["votes"].astype(float)
    return long_df


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion. Returns (low, high). NaN if n<=0."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index on shares (0..1)."""
    s = np.asarray(shares, dtype=float)
    return float(np.sum(s ** 2))


def enp(shares: np.ndarray) -> float:
    """Effective Number of Parties (Laakso-Taagepera)."""
    h = hhi(shares)
    return float(1 / h) if h > 0 else np.nan


def mad_zscore(x: pd.Series) -> pd.Series:
    """Robust z-score using median absolute deviation."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


class ElectionAnalyzer:
    def __init__(self):
        self.df = self._load_data()
        self.long = build_party_long(self.df)
        self.results: dict = {}
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        path = CLEANED_DIR / "election_results_cleaned.csv"
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} cleaned records")
        return df

    def run_all(self):
        if self.long.empty:
            logger.warning("Long-format party frame is empty. Nothing to analyze.")
            return
        self.party_performance()
        self.concentration_metrics()
        self.turnout_and_spoilage()
        self.advance_vs_day_test()
        self.anomaly_detection()
        self.station_clustering()
        self.comparative_2023()
        self._save_summary()
        logger.info(f"All analyses complete → {FIGURES_DIR}")

    # ── Party performance ──
    def party_performance(self):
        logger.info("Party performance...")
        agg = (self.long.groupby(["is_party_list", "party"], as_index=False)
                       .agg(votes=("votes", "sum"),
                            stations=("polling_unit_id", "nunique")))
        agg["share"] = agg.groupby("is_party_list")["votes"].transform(
            lambda s: s / s.sum() if s.sum() else np.nan)
        total_by_kind = agg.groupby("is_party_list")["votes"].transform("sum")
        ci = [wilson_ci(v, n) for v, n in zip(agg["votes"], total_by_kind)]
        agg["share_ci_low"] = [c[0] for c in ci]
        agg["share_ci_high"] = [c[1] for c in ci]
        agg = agg.sort_values(["is_party_list", "votes"], ascending=[True, False])
        agg.to_csv(FIGURES_DIR / "party_performance.csv", index=False)
        self.results["party_performance"] = agg

        for is_pl, sub in agg.groupby("is_party_list"):
            label = "Party-list" if is_pl else "Constituency"
            top = sub.head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11, 7))
            err_low = (top["share"] - top["share_ci_low"]) * 100
            err_high = (top["share_ci_high"] - top["share"]) * 100
            ax.barh(top["party"], top["share"] * 100,
                    xerr=[err_low, err_high], color="#6366f1",
                    error_kw={"ecolor": "#1e293b", "capsize": 3})
            ax.set_xlabel("Vote share % (95% Wilson CI)")
            ax.set_title(f"{label} — top 15 parties")
            plt.tight_layout()
            slug = "party_list" if is_pl else "constituency"
            plt.savefig(FIGURES_DIR / f"party_share_{slug}.png", dpi=150)
            plt.close()

    # ── Concentration metrics ──
    def concentration_metrics(self):
        logger.info("Concentration metrics (ENP, HHI, Gini)...")
        rows = []
        for is_pl, sub in self.long.groupby("is_party_list"):
            shares = sub.groupby("party")["votes"].sum()
            shares = shares / shares.sum() if shares.sum() else shares
            rows.append({
                "kind": "party_list" if is_pl else "constituency",
                "n_parties": int((shares > 0).sum()),
                "ENP_Laakso_Taagepera": enp(shares.values),
                "HHI": hhi(shares.values),
                "Gini_votes": gini(sub.groupby("party")["votes"].sum().values),
            })
        out = pd.DataFrame(rows)
        out.to_csv(FIGURES_DIR / "concentration_metrics.csv", index=False)
        self.results["concentration"] = out

    # ── Turnout & spoilage ──
    def turnout_and_spoilage(self):
        logger.info("Turnout & spoilage...")
        if "total_ballots" not in self.df.columns:
            return
        cols = [c for c in ["total_ballots", "good_ballots", "bad_ballots",
                            "turnout_valid_ratio", "invalid_ballot_ratio"]
                if c in self.df.columns]
        desc = self.df[cols].describe().T
        desc.to_csv(FIGURES_DIR / "turnout_descriptive.csv")
        self.results["turnout"] = desc

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        self.df["total_ballots"].hist(bins=30, ax=axes[0], color="#10b981")
        axes[0].set_title("Ballots per station")
        axes[0].set_xlabel("Total ballots")
        if "invalid_ballot_ratio" in self.df.columns:
            self.df["invalid_ballot_ratio"].hist(bins=30, ax=axes[1], color="#f59e0b")
            axes[1].set_title("Invalid ballot ratio")
            axes[1].set_xlabel("Ratio")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "turnout_distribution.png", dpi=150)
        plt.close()

    # ── Advance vs election-day Mann-Whitney ──
    def advance_vs_day_test(self):
        logger.info("Advance vs election-day Mann-Whitney U test on station-level vote shares...")
        rows = []
        share_df = self.long.copy()
        denom = pd.to_numeric(share_df["good_ballots"], errors="coerce")
        fallback = share_df.groupby("polling_unit_id")["votes"].transform("sum")
        share_df["station_share"] = share_df["votes"] / denom.where(denom > 0, fallback).replace(0, np.nan)

        for party, sub in share_df.groupby("party"):
            adv = sub.loc[sub["is_advance"], "station_share"].dropna().values
            day = sub.loc[~sub["is_advance"], "station_share"].dropna().values
            if len(adv) < 3 or len(day) < 3:
                continue
            try:
                u, p = stats.mannwhitneyu(adv, day, alternative="two-sided")
            except ValueError:
                continue
            rows.append({
                "party": party,
                "n_advance": len(adv),
                "n_day": len(day),
                "median_share_advance": float(np.median(adv)),
                "median_share_day": float(np.median(day)),
                "U": float(u),
                "p_value": float(p),
            })
        if not rows:
            logger.warning("Not enough advance/day data for U-test.")
            return
        out = pd.DataFrame(rows).sort_values("p_value")
        # Bonferroni
        out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
        out.to_csv(FIGURES_DIR / "advance_vs_day_mwu.csv", index=False)
        self.results["mwu"] = out

    # ── Anomaly detection (robust MAD) ──
    def anomaly_detection(self):
        logger.info("Anomaly detection (MAD)...")
        unit_col = "polling_unit_id" if "polling_unit_id" in self.df.columns else "station_id"
        if unit_col not in self.df.columns:
            return
        flags = []
        if "total_ballots" in self.df.columns:
            z = mad_zscore(self.df["total_ballots"].fillna(0))
            self.df["zscore_total_ballots"] = z
            flagged = self.df.loc[z.abs() > 3.5]
            for _, r in flagged.iterrows():
                flags.append({"station_id": r.get("station_id"),
                              "polling_unit_id": r.get("polling_unit_id", r.get("station_id")),
                              "form_type": r.get("form_type"),
                              "metric": "total_ballots",
                              "value": r.get("total_ballots"),
                              "mad_z": r.get("zscore_total_ballots")})
        if "invalid_ballot_ratio" in self.df.columns:
            z = mad_zscore(self.df["invalid_ballot_ratio"].fillna(0))
            flagged = self.df.loc[z.abs() > 3.5]
            for _, r in flagged.iterrows():
                flags.append({"station_id": r.get("station_id"),
                              "polling_unit_id": r.get("polling_unit_id", r.get("station_id")),
                              "form_type": r.get("form_type"),
                              "metric": "invalid_ballot_ratio",
                              "value": r.get("invalid_ballot_ratio"),
                              "mad_z": float(z.loc[r.name])})
        if flags:
            pd.DataFrame(flags).to_csv(FIGURES_DIR / "anomalies.csv", index=False)
            logger.info(f"Flagged {len(flags)} anomalies (|MAD-z| > 3.5)")

    # ── Station clustering on party-share vector ──
    def station_clustering(self):
        logger.info("Station clustering on party shares...")
        if self.long.empty:
            return
        pivot = (self.long.pivot_table(index="polling_unit_id", columns="party",
                                       values="votes", aggfunc="sum")
                          .fillna(0))
        pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        if len(pivot) < 2:
            logger.warning("Not enough stations to cluster.")
            return
        n_clusters = min(4, len(pivot))
        scaled = StandardScaler().fit_transform(pivot.values)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        pivot["cluster"] = labels
        pivot.to_csv(FIGURES_DIR / "station_clusters.csv")

        if scaled.shape[1] >= 2:
            coords = PCA(n_components=2).fit_transform(scaled)
            fig, ax = plt.subplots(figsize=(9, 7))
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                            cmap="viridis", alpha=0.7, s=40)
            ax.set_title("Polling-station clusters by party-share profile")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            plt.colorbar(sc, label="Cluster")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "station_clusters.png", dpi=150)
            plt.close()

    # ── 2023 comparison ──
    def comparative_2023(self):
        ref = REFERENCE_DIR / "election_2023.csv"
        if not ref.exists():
            return
        logger.info("2023 vs 2026 comparison...")
        df_23 = pd.read_csv(ref)
        cur = (self.long.groupby("party")["votes"].sum()
                        .sort_values(ascending=False).head(10))
        cur_share = cur / cur.sum() if cur.sum() else cur
        cur_share.to_frame("share_2026").to_csv(
            FIGURES_DIR / "top10_2026.csv")
        # 2023 file format unknown; skip plotting unless schema matches
        try:
            v23 = [c for c in df_23.columns if c.endswith("_votes")]
            prev = df_23[v23].sum()
            prev_share = prev / prev.sum() if prev.sum() else prev
            prev_share.to_frame("share_2023").to_csv(
                FIGURES_DIR / "top10_2023.csv")
        except Exception as e:
            logger.warning(f"2023 schema mismatch: {e}")

    def _save_summary(self):
        path = FIGURES_DIR / "analysis_summary.txt"
        lines = ["Thailand Election 2026 — Analysis Summary", "=" * 50, ""]
        if "concentration" in self.results:
            lines.append("Concentration metrics:")
            lines.append(self.results["concentration"].to_string(index=False))
            lines.append("")
        if "party_performance" in self.results:
            top = (self.results["party_performance"]
                   .groupby("is_party_list").head(5))
            lines.append("Top parties by ballot kind:")
            lines.append(top.to_string(index=False))
            lines.append("")
        if "mwu" in self.results:
            sig = self.results["mwu"].query("p_bonferroni < 0.05")
            lines.append(f"Mann-Whitney U: {len(sig)} parties significant after Bonferroni")
        path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ElectionAnalyzer().run_all()
