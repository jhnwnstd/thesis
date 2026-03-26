import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import statsmodels.formula.api as smf  # type: ignore[import-not-found,import-untyped]
from config import DATASET_PATHS, VOWELS_EXTENDED
from statsmodels.discrete.discrete_model import (  # type: ignore[import-not-found,import-untyped]
    LogitResults,
)
from utils import is_vowel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def preprocess_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Preprocess data for analysis."""
    try:
        data = pd.read_csv(
            file_path,
            usecols=[
                "Tested_Word",
                "Original_Word",
                "Top1_Is_Accurate",
                "Correct_Letters",
            ],
        )
        logger.info(f"Data loaded successfully from {file_path}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

    data["Missing_Letter_Position"] = data["Tested_Word"].str.find("_")
    data["Word_Length"] = data["Original_Word"].str.len()
    data["Normalized_Missing_Letter_Position"] = (
        data["Missing_Letter_Position"] / (data["Word_Length"] - 1)
    ).fillna(0)

    if "Correct_Letters" in data.columns:
        data["is_vowel"] = (
            data["Correct_Letters"]
            .str[0]
            .apply(lambda c: is_vowel(c, VOWELS_EXTENDED))
            .astype(int)
        )
    else:
        data["is_vowel"] = 0

    data["Top1_Is_Accurate"] = pd.to_numeric(
        data["Top1_Is_Accurate"], errors="coerce"
    ).astype("Int64")
    data = data.dropna(
        subset=[
            "Top1_Is_Accurate",
            "is_vowel",
            "Word_Length",
            "Normalized_Missing_Letter_Position",
        ]
    )

    logger.info(f"Data shape after preprocessing: {data.shape}")
    logger.info(f"Unique values in 'is_vowel': {data['is_vowel'].unique()}")

    return data


def run_logistic_regression(
    data: pd.DataFrame, dataset_name: str
) -> Optional[LogitResults]:
    """Run logistic regression analysis on the data."""
    if len(data["is_vowel"].unique()) < 2:
        logger.warning(
            f"Insufficient variability in 'is_vowel' for {dataset_name} dataset. Skipping regression."
        )
        return None

    formula = "Top1_Is_Accurate ~ is_vowel + Word_Length + Normalized_Missing_Letter_Position"
    try:
        model = smf.logit(formula=formula, data=data).fit(disp=0)
        logger.info(
            f"Regression Summary for {dataset_name}:\n{model.summary().tables[1]}"
        )
        return model
    except Exception as e:
        logger.error(
            f"Error in logistic regression for {dataset_name}: {str(e)}"
        )
        return None


def process_dataset(
    dataset_name: str, file_path: Path
) -> Tuple[str, Optional[pd.DataFrame], Optional[LogitResults]]:
    """Process a single dataset."""
    data = preprocess_data(file_path)
    if data is not None and not data.empty:
        model = run_logistic_regression(data, dataset_name)
        return dataset_name, data, model
    return dataset_name, None, None


def main():
    results = {}
    for name, path in DATASET_PATHS.items():
        dataset_name, data, model = process_dataset(name, path)
        if data is not None and model is not None:
            results[dataset_name] = {"data": data, "model": model}

    # Compare results across datasets
    for dataset_name, result in results.items():
        logger.info(f"\nResults for {dataset_name}:")
        logger.info(f"Model AIC: {result['model'].aic:.2f}")
        logger.info(f"Top Predictor: {result['model'].params.abs().idxmax()}")
        logger.info(f"Pseudo R-squared: {result['model'].prsquared:.4f}")


if __name__ == "__main__":
    main()
