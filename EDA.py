"""Exploratory Data Analysis for Project 1.

Run with: python EDA.py
"""

import csv

import numpy as np

from helpers import load_csv_data
from mlproject.config import DATA_DIR, FIGURES_DIR
from mlproject.plots import (
    plot_missingness,
    plot_missingness_with_cutoff,
    save_figure,
)

CANDIDATE_SPECIAL_CODES = [
    7,
    9,
    77,
    99,
    777,
    999,
    7777,
    9999,
    77777,
    99999,
    777777,
    999999,
]


def analyze_special_codes(x, feature_names, output_path, codes=None):
    """Export occurrences of candidate special codes without changing x.

    Output columns: feature, code, count, percentage.
    Use the codebook to decide which feature/code pairs truly mean missing.
    """
    if codes is None:
        codes = CANDIDATE_SPECIAL_CODES

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["feature", "code", "count", "percentage"])

        n_rows = x.shape[0]
        for j, feature in enumerate(feature_names):
            col = x[:, j]
            observed = col[~np.isnan(col)]

            for code in codes:
                count = int(np.sum(observed == code))
                if count > 0:
                    percentage = 100.0 * count / n_rows
                    writer.writerow([feature, code, count, percentage])

    print(f"Candidate special-code analysis saved to: {output_path}")


def load_feature_names():
    """Read feature names from x_train.csv, excluding the ID column."""
    with open(DATA_DIR / "x_train.csv", encoding="utf-8") as f:
        feature_names = f.readline().strip().split(",")

    # load_csv_data removes the first column (ID).
    feature_names = feature_names[1:]

    return feature_names


def inspect_features(x, feature_names):
    """Print descriptive statistics for every feature."""
    print("\n===== FEATURE DIAGNOSTICS =====")

    for j, name in enumerate(feature_names):
        col = x[:, j]
        valid = col[~np.isnan(col)]
        missing_count = int(np.isnan(col).sum())
        missing_ratio = missing_count / len(col)

        if valid.size == 0:
            print(f"{name:12s} | missing=100.00% | no observed values")
            continue

        print(
            f"{name:12s} | "
            f"missing={missing_ratio:7.2%} | "
            f"unique={len(np.unique(valid)):4d} | "
            f"min={np.min(valid):10.2f} | "
            f"median={np.median(valid):10.2f} | "
            f"max={np.max(valid):10.2f}"
        )


def export_feature_analysis(x, feature_names, output_path):
    """Export feature statistics and selected code frequencies to CSV."""
    special_codes = [7, 9, 77, 99, 777, 999]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")

        writer.writerow(
            [
                "feature",
                "missing_count",
                "missing_ratio",
                "unique_values",
                "min",
                "median",
                "max",
                "mean",
                "std",
                "count_7",
                "count_9",
                "count_77",
                "count_99",
                "count_777",
                "count_999",
            ]
        )

        for j, name in enumerate(feature_names):
            col = x[:, j]
            valid = col[~np.isnan(col)]
            missing_count = int(np.isnan(col).sum())
            missing_ratio = missing_count / len(col)

            if valid.size == 0:
                writer.writerow(
                    [
                        name,
                        missing_count,
                        missing_ratio,
                        0,
                        "",
                        "",
                        "",
                        "",
                        "",
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
                continue

            stats = [
                name,
                missing_count,
                missing_ratio,
                len(np.unique(valid)),
                np.min(valid),
                np.median(valid),
                np.max(valid),
                np.mean(valid),
                np.std(valid),
            ]

            code_counts = [int(np.sum(valid == code)) for code in special_codes]

            writer.writerow(stats + code_counts)

    print(f"\nFeature analysis saved to: {output_path}")


def main():
    """Run the exploratory data analysis."""
    # 1. Load dataset
    print("Loading dataset...")

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(str(DATA_DIR))

    print("\n===== DATASET SHAPES =====")
    print("x_train:", x_train.shape)
    print("x_test: ", x_test.shape)
    print("y_train:", y_train.shape)
    print("Number of features:", x_train.shape[1])

    # 2. Read and validate feature names
    feature_names = load_feature_names()

    if len(feature_names) != x_train.shape[1]:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not "
            f"match x_train columns ({x_train.shape[1]})."
        )

    # 3. Analyze missing values
    nan_mask = np.isnan(x_train)
    nan_ratio = np.mean(nan_mask, axis=0)

    # Plot the distribution of missing values
    fig = plot_missingness(x_train, feature_names)
    save_figure(fig, "missing_values_distribution")

    fig = plot_missingness_with_cutoff(x_train, feature_names, threshold=0.80)
    save_figure(fig, "missing_values_distribution_cutoff")

    print("\n===== MISSING VALUES =====")
    print("Total missing values:", int(np.sum(nan_mask)))
    print("Overall missing ratio:", f"{np.mean(nan_mask):.2%}")
    print("Features with at least one NaN:", int(np.sum(nan_ratio > 0)))

    thresholds = [0.01, 0.10, 0.30, 0.50, 0.70, 0.90, 0.97]

    print("\nFeatures exceeding missing-value thresholds:")
    for threshold in thresholds:
        count = int(np.sum(nan_ratio >= threshold))
        print(f"  At least {threshold:.0%} missing: {count}")

    # 4. Inspect the 20 features with the most missing values
    print("\n===== TOP 20 FEATURES WITH MOST MISSING VALUES =====")

    # most_missing = np.argsort(nan_ratio)[::-1]

    # for j in most_missing[:20]:
    # print(f"{feature_names[j]:12s} | missing={nan_ratio[j]:.2%}")

    # 5. Inspect all features
    # inspect_features(x_train, feature_names)

    # 6. Detect constant features among observed values
    constant_features = []

    for j, name in enumerate(feature_names):
        col = x_train[:, j]
        observed = col[~np.isnan(col)]

        if observed.size > 0 and np.all(observed == observed[0]):
            constant_features.append(name)

    print("\n===== CONSTANT FEATURES =====")
    print("Constant features among observed values:", len(constant_features))

    for name in constant_features:
        print(" ", name)

    # 7. Analyze class imbalance
    labels, counts = np.unique(y_train, return_counts=True)

    print("\n===== CLASS DISTRIBUTION =====")

    for label, count in zip(labels, counts):
        print(f"Label {label}: {count} ({count / len(y_train):.2%})")

    # 8. Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reports_dir = DATA_DIR.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 11. Export feature analysis to CSV
    output_path = reports_dir / "feature_analysis.csv"

    export_feature_analysis(
        x_train,
        feature_names,
        output_path,
    )

    # 12. Count candidate special codes without replacing any values
    special_codes_path = reports_dir / "special_codes_analysis.csv"
    analyze_special_codes(
        x_train,
        feature_names,
        special_codes_path,
    )

    print("\n===== END EDA =====")


if __name__ == "__main__":
    main()
