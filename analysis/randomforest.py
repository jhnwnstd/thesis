import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from config import DATASET_PATHS
from sklearn.ensemble import (  # type: ignore[import-not-found,import-untyped]
    RandomForestClassifier,
)
from sklearn.feature_selection import (  # type: ignore[import-not-found,import-untyped]
    RFECV,
)
from sklearn.metrics import (  # type: ignore[import-not-found,import-untyped]
    accuracy_score,
)
from sklearn.model_selection import (  # type: ignore[import-not-found,import-untyped]
    train_test_split,
)
from utils import is_vowel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_and_preprocess(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and preprocess it for model training.

    Args:
        path (Path): Path to the dataset.

    Returns:
        tuple: A tuple containing the features DataFrame and target series.
    """
    df = pd.read_csv(path)
    df["Word_Length"] = df["Original_Word"].str.len()
    df["Missing_Letter_Index"] = df["Original_Word"].str.find("_")
    df["Relative_Position"] = df["Missing_Letter_Index"] / df[
        "Word_Length"
    ].clip(lower=1)
    df["Top1_Predicted_Letter_is_Vowel"] = df["Top1_Predicted_Letter"].apply(
        is_vowel
    )
    df["Correct_Letter_is_Vowel"] = df["Correct_Letters"].apply(is_vowel)

    features = df[
        [
            "Word_Length",
            "Relative_Position",
            "Top1_Predicted_Letter_is_Vowel",
            "Correct_Letter_is_Vowel",
        ]
    ]
    target = df["Top1_Is_Accurate"]

    return features, target


def train_and_evaluate_model(
    features: pd.DataFrame, target: pd.Series
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train and evaluate the Random Forest model with RFECV for feature selection.

    Args:
        features (pd.DataFrame): The features for model training.
        target (pd.Series): The target variable for the model.

    Returns:
        tuple: A tuple containing the model's accuracy, selected features, and feature importances.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    rf_model = RandomForestClassifier(random_state=42)
    rfecv = RFECV(
        estimator=rf_model, step=1, cv=5, scoring="accuracy", n_jobs=-1
    )
    rfecv.fit(X_train, y_train)
    y_pred = rfecv.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    selected_features = rfecv.support_
    feature_importances = (
        rfecv.estimator_.feature_importances_  # type: ignore[attr-defined]
    )

    return accuracy, selected_features, feature_importances


def main() -> None:
    results = {}
    feature_names = [
        "Word Length",
        "Relative Position",
        "Predicted Letter is Vowel",
        "Correct Letter is Vowel",
    ]

    for title, path in DATASET_PATHS.items():
        logger.info(f"Processing dataset: {title}")
        features, target = load_and_preprocess(path)
        accuracy, selected_features, feature_importances = (
            train_and_evaluate_model(features, target)
        )
        results[title] = {
            "accuracy": accuracy,
            "selected_features": selected_features,
            "feature_importances": feature_importances,
        }

    for title, info in results.items():
        logger.info(f"\n{title}:")
        logger.info(f"Accuracy = {info['accuracy']:.4f}")
        logger.info("Selected Features and Their Importances:")
        selected_features = np.asarray(info["selected_features"])
        feature_importances = np.asarray(info["feature_importances"])
        for name, selected, importance in zip(
            feature_names,
            selected_features,
            feature_importances,
        ):
            logger.info(
                f"    {name}: {'Selected' if selected else 'Not Selected'}, Importance: {importance:.4f}"
            )


if __name__ == "__main__":
    main()
