#!/usr/bin/env python3
"""Generate manuscript tables and figures directly from benchmark artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.metrics import rmse, safe_spearman
from kinase_ligand_ranking.sequence_identity import nearest_train_identity
from kinase_ligand_ranking.splits import target_family


RESULTS_DIR = BASE_DIR / "results" / "benchmark"
EXTERNAL_RESULTS_DIR = BASE_DIR / "results" / "external_validation" / "bindingdb"
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark"
ANALYSIS_DIR = BASE_DIR / "results" / "benchmark_analysis"
FIGURES_DIR = BASE_DIR / "figures"
PAPER_FRAGMENT = BASE_DIR / "paper" / "generated_results.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark analysis, tables, and figures")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--external-results-dir", type=Path, default=EXTERNAL_RESULTS_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--paper-fragment", type=Path, default=PAPER_FRAGMENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.results_dir / "summary.csv")
    best_by_split = (
        summary_df.sort_values(["split", "rmse", "model"])
        .groupby("split", as_index=False)
        .first()
    )
    best_by_split.to_csv(args.analysis_dir / "best_by_split.csv", index=False)
    summary_df.to_csv(args.analysis_dir / "benchmark_overview.csv", index=False)

    leakage_target_df, leakage_summary_df = build_leakage_analysis(args.benchmark_dir)
    leakage_target_df.to_csv(args.analysis_dir / "leakage_per_target.csv", index=False)
    leakage_summary_df.to_csv(args.analysis_dir / "leakage_summary.csv", index=False)

    mutation_df = build_mutation_family_analysis(args.benchmark_dir, args.results_dir / "mutation_holdout")
    mutation_df.to_csv(args.analysis_dir / "mutation_family_analysis.csv", index=False)

    external_summary_df = pd.DataFrame()
    if (args.external_results_dir / "summary.csv").exists():
        external_summary_df = pd.read_csv(args.external_results_dir / "summary.csv")
        external_summary_df.to_csv(args.analysis_dir / "external_validation_summary.csv", index=False)

    plot_rmse_by_split(summary_df, args.figures_dir / "split_rmse_comparison.png")
    plot_spearman_by_split(summary_df, args.figures_dir / "split_spearman_comparison.png")
    plot_leakage_summary(leakage_summary_df, args.figures_dir / "sequence_identity_leakage.png")
    if not mutation_df.empty:
        plot_mutation_family_deltas(mutation_df, args.figures_dir / "mutation_family_delta.png")
    if not external_summary_df.empty:
        plot_external_validation(external_summary_df, args.figures_dir / "external_validation_rmse.png")

    args.paper_fragment.write_text(
        build_paper_fragment(
            summary_df=summary_df,
            best_by_split=best_by_split,
            leakage_summary_df=leakage_summary_df,
            mutation_df=mutation_df,
            external_summary_df=external_summary_df,
            analysis_dir=args.analysis_dir,
            figures_dir=args.figures_dir,
        )
    )

    print(f"Saved analysis to: {args.analysis_dir}")
    print(f"Saved figures to: {args.figures_dir}")
    print(f"Saved paper fragment to: {args.paper_fragment}")


def build_leakage_analysis(benchmark_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity_path = benchmark_dir / "sequence_identity" / "target_sequence_identity.csv"
    if not identity_path.exists():
        raise FileNotFoundError(
            "Expected exact identity table at data/benchmark/sequence_identity/target_sequence_identity.csv. "
            "Regenerate sequence_identity splits first."
        )
    identity_table = pd.read_csv(identity_path)

    per_target_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for split_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
        train_df = pd.read_csv(split_dir / "train.csv")
        test_df = pd.read_csv(split_dir / "test.csv")
        leakage_df = nearest_train_identity(
            identity_table,
            train_targets=train_df["target_id"].astype(str).unique(),
            eval_targets=test_df["target_id"].astype(str).unique(),
        )
        leakage_df["split"] = split_dir.name
        per_target_rows.append(leakage_df)
        summary_rows.append(
            {
                "split": split_dir.name,
                "test_targets": int(len(leakage_df)),
                "mean_nearest_train_identity": float(leakage_df["nearest_train_sequence_identity"].mean()),
                "median_nearest_train_identity": float(leakage_df["nearest_train_sequence_identity"].median()),
                "max_nearest_train_identity": float(leakage_df["nearest_train_sequence_identity"].max()),
            }
        )

    per_target_df = pd.concat(per_target_rows, ignore_index=True).sort_values(["split", "target_id"])
    summary_df = pd.DataFrame(summary_rows).sort_values("split").reset_index(drop=True)
    return per_target_df, summary_df


def build_mutation_family_analysis(benchmark_dir: Path, mutation_results_dir: Path) -> pd.DataFrame:
    if not mutation_results_dir.exists():
        return pd.DataFrame()

    identity_table = pd.read_csv(benchmark_dir / "sequence_identity" / "target_sequence_identity.csv")
    train_df = pd.read_csv(benchmark_dir / "mutation_holdout" / "train.csv").copy()
    test_df = pd.read_csv(benchmark_dir / "mutation_holdout" / "test.csv").copy()
    test_leakage_df = nearest_train_identity(
        identity_table,
        train_targets=train_df["target_id"].astype(str).unique(),
        eval_targets=test_df["target_id"].astype(str).unique(),
    )
    leakage_map = dict(
        zip(
            test_leakage_df["target_id"].astype(str),
            test_leakage_df["nearest_train_sequence_identity"].astype(float),
        )
    )

    train_df["mutation_family"] = train_df["target_id"].map(target_family)
    train_df["is_variant"] = train_df["target_id"].astype(str) != train_df["mutation_family"].astype(str)

    rows: List[Dict[str, object]] = []
    for model_dir in sorted(path for path in mutation_results_dir.iterdir() if path.is_dir()):
        predictions_path = model_dir / "test_predictions.csv"
        if not predictions_path.exists():
            continue

        prediction_df = pd.read_csv(predictions_path).copy()
        prediction_df["mutation_family"] = prediction_df["target_id"].map(target_family)
        prediction_df["nearest_train_sequence_identity"] = prediction_df["target_id"].map(leakage_map)

        for family, group in prediction_df.groupby("mutation_family"):
            wildtype_train = train_df[(train_df["mutation_family"] == family) & (~train_df["is_variant"])]
            rows.append(
                {
                    "model": model_dir.name,
                    "mutation_family": family,
                    "rows": int(len(group)),
                    "targets": int(group["target_id"].nunique()),
                    "variant_targets": int(group["target_id"].nunique()),
                    "wildtype_train_rows": int(len(wildtype_train)),
                    "wildtype_train_targets": int(wildtype_train["target_id"].nunique()),
                    "rmse": rmse(group["p_activity"].to_numpy(), group["predicted_p_activity"].to_numpy()),
                    "spearman": safe_spearman(
                        group["p_activity"].to_numpy(),
                        group["predicted_p_activity"].to_numpy(),
                    ),
                    "variant_mean_p_activity": float(group["p_activity"].mean()),
                    "wildtype_train_mean_p_activity": float(wildtype_train["p_activity"].mean())
                    if not wildtype_train.empty
                    else None,
                    "variant_minus_wildtype_p_activity": float(group["p_activity"].mean() - wildtype_train["p_activity"].mean())
                    if not wildtype_train.empty
                    else None,
                    "mean_nearest_train_identity": float(group["nearest_train_sequence_identity"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "rmse", "mutation_family"]).reset_index(drop=True)


def plot_rmse_by_split(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10.5, 5))
    sns.barplot(data=summary_df, x="split", y="rmse", hue="model")
    plt.ylabel("RMSE")
    plt.xlabel("Split")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_spearman_by_split(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10.5, 5))
    sns.barplot(data=summary_df, x="split", y="spearman", hue="model")
    plt.ylabel("Spearman")
    plt.xlabel("Split")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_leakage_summary(leakage_summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=leakage_summary_df, x="split", y="mean_nearest_train_identity", color="#2F4858")
    plt.ylabel("Mean nearest-train identity")
    plt.xlabel("Split")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_mutation_family_deltas(mutation_df: pd.DataFrame, output_path: Path) -> None:
    top_families = (
        mutation_df.groupby("mutation_family", as_index=False)["rows"]
        .sum()
        .sort_values("rows", ascending=False)
        .head(10)["mutation_family"]
        .tolist()
    )
    plot_df = mutation_df[mutation_df["mutation_family"].isin(top_families)]
    plt.figure(figsize=(10.5, 5.5))
    sns.barplot(data=plot_df, x="mutation_family", y="variant_minus_wildtype_p_activity", hue="model")
    plt.ylabel("Variant minus wild-type mean pActivity")
    plt.xlabel("Mutation family")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_external_validation(external_summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8.5, 4.5))
    sns.barplot(data=external_summary_df, x="model", y="rmse", color="#7C9885")
    plt.ylabel("External-validation RMSE")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_paper_fragment(
    *,
    summary_df: pd.DataFrame,
    best_by_split: pd.DataFrame,
    leakage_summary_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
    external_summary_df: pd.DataFrame,
    analysis_dir: Path,
    figures_dir: Path,
) -> str:
    literature_df = summary_df[
        summary_df["model"].isin(["ridge_ensemble", "deepdta_exact", "graphdta_gcn_exact", "dual_tower_uq"])
    ].copy()
    rmse_matrix = literature_df.pivot(index="split", columns="model", values="rmse").round(3)
    spearman_matrix = literature_df.pivot(index="split", columns="model", values="spearman").round(3)
    mutation_preview = mutation_df.head(12).round(3) if not mutation_df.empty else pd.DataFrame()

    lines = [
        "# Generated Benchmark Evidence",
        "",
        "All tables below are generated from result files under `results/`.",
        "",
        "## Primary Tables",
        "",
        best_by_split[
            ["split", "model", "rmse", "spearman", "roc_auc", "mean_per_target_spearman"]
        ].round(3).to_csv(index=False),
        rmse_matrix.to_csv(),
        spearman_matrix.to_csv(),
        leakage_summary_df.round(3).to_csv(index=False),
    ]
    if not mutation_preview.empty:
        lines.append(mutation_preview.to_csv(index=False))
    if not external_summary_df.empty:
        lines.append(external_summary_df.round(3).to_csv(index=False))

    lines.extend(
        [
            "## Analysis Artifacts",
            "",
            f"- [best_by_split.csv]({(analysis_dir / 'best_by_split.csv').resolve()})",
            f"- [benchmark_overview.csv]({(analysis_dir / 'benchmark_overview.csv').resolve()})",
            f"- [leakage_summary.csv]({(analysis_dir / 'leakage_summary.csv').resolve()})",
            f"- [leakage_per_target.csv]({(analysis_dir / 'leakage_per_target.csv').resolve()})",
            f"- [mutation_family_analysis.csv]({(analysis_dir / 'mutation_family_analysis.csv').resolve()})",
        ]
    )
    if not external_summary_df.empty:
        lines.append(
            f"- [external_validation_summary.csv]({(analysis_dir / 'external_validation_summary.csv').resolve()})"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Split RMSE]({(figures_dir / 'split_rmse_comparison.png').resolve()})",
            f"![Split Spearman]({(figures_dir / 'split_spearman_comparison.png').resolve()})",
            f"![Identity Leakage]({(figures_dir / 'sequence_identity_leakage.png').resolve()})",
        ]
    )
    if not mutation_df.empty:
        lines.append(f"![Mutation Family Delta]({(figures_dir / 'mutation_family_delta.png').resolve()})")
    if not external_summary_df.empty:
        lines.append(f"![External Validation RMSE]({(figures_dir / 'external_validation_rmse.png').resolve()})")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
