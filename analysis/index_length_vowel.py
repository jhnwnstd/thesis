from pathlib import Path

import numpy as np
import pandas as pd
from config import DATASET_PATHS
from sklearn.linear_model import (  # type: ignore[import-not-found,import-untyped]
    LogisticRegression,
)
from sklearn.metrics import (  # type: ignore[import-not-found,import-untyped]
    accuracy_score,
    classification_report,
    log_loss,
)
from sklearn.model_selection import (  # type: ignore[import-not-found,import-untyped]
    train_test_split,
)
from utils import is_vowel


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for model training."""
    data["Word_Length"] = data["Original_Word"].apply(len)
    data["Missing_Letter_Position"] = data["Tested_Word"].str.find("_")
    data["Relative_Position"] = (
        data["Missing_Letter_Position"] / data["Word_Length"]
    )
    data["Missing_Letter"] = data.apply(
        lambda row: (
            row["Original_Word"][row["Missing_Letter_Position"]]
            if row["Missing_Letter_Position"] != -1
            else ""
        ),
        axis=1,
    )
    data["Is_Vowel"] = data["Missing_Letter"].apply(is_vowel).astype(int)
    data["Top1_Is_Accurate"] = data["Top1_Is_Accurate"].astype(int)
    return data


def logistic_regression_analysis(dataset_name: str, file_path: Path) -> None:
    """Run logistic regression analysis on the dataset."""
    data = pd.read_csv(file_path)
    data = preprocess_data(data)

    X = data[["Relative_Position", "Word_Length", "Is_Vowel"]]
    y = data["Top1_Is_Accurate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    log_likelihood_model = -log_loss(y_test, y_pred_proba, normalize=False)
    log_likelihood_null = -log_loss(
        y_test, np.ones_like(y_test) * y_test.mean(), normalize=False
    )

    pseudo_r_squared = 1 - (log_likelihood_model / log_likelihood_null)

    print(f"\n{dataset_name} Dataset Analysis:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Pseudo R-squared: {pseudo_r_squared:.4f}")
    print("Classification Report:")
    print(report)
    print("Model Coefficients:")
    print(f"  Relative Position: {coefficients[0]:.4f}")
    print(f"  Word Length: {coefficients[1]:.4f}")
    print(f"  Is Vowel: {coefficients[2]:.4f}")
    print(f"  Intercept: {intercept:.4f}")


def main():
    for dataset_name, file_path in DATASET_PATHS.items():
        logistic_regression_analysis(dataset_name, file_path)


if __name__ == "__main__":
    main()
