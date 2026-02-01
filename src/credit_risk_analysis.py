import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "default.payment.next.month"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    summary = df.describe().transpose()
    summary.to_csv(output_dir / "summary_statistics.csv")

    missing = df.isna().sum().to_frame(name="missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df)) * 100
    missing.to_csv(output_dir / "missing_values.csv")

    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts(normalize=True).to_frame(name="pct")
        target_dist.to_csv(output_dir / "target_distribution.csv")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )


def evaluate_model(name: str, model, X_test, y_test, output_dir: Path) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }

    report_path = output_dir / f"{name}_classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")

    return metrics


def run_modeling(df: pd.DataFrame, output_dir: Path) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            "Verify the dataset matches the Kaggle UCI Credit Card CSV."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        ),
    }

    metrics = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        metrics.append(evaluate_model(name, pipeline, X_test, y_test, output_dir))

    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistical analysis and baseline credit risk modeling."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the UCI Credit Card CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write analysis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    df = load_data(args.data_path)
    write_summary_statistics(df, args.output_dir)
    plot_correlation_heatmap(df, args.output_dir)
    run_modeling(df, args.output_dir)


if __name__ == "__main__":
    main()
