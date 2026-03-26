import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf  # type: ignore[import-not-found,import-untyped]
from config import VOWELS_BASIC

logger = logging.getLogger(__name__)


def is_vowel(letter: str, vowel_set: frozenset = VOWELS_BASIC) -> bool:
    """Check if a letter is a vowel."""
    return letter.lower() in vowel_set


def load_csv(
    file_path: Path, usecols: Optional[list] = None
) -> Optional[pd.DataFrame]:
    """Load a CSV file with optional column selection. Returns None on error."""
    try:
        data = pd.read_csv(file_path, usecols=usecols)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def load_and_prepare_index_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and prepare data with Normalized_Missing_Index for analysis.

    Used by: letter_index.py, letter_index_plot.py
    """
    try:
        data = pd.read_csv(
            file_path, usecols=["Tested_Word", "Top1_Is_Accurate"]
        )
        logger.info(f"Data loaded successfully from {file_path}")

        data["Word_Length"] = data["Tested_Word"].str.len()
        data["Normalized_Missing_Index"] = data["Tested_Word"].str.find(
            "_"
        ) / (data["Word_Length"] - 1)
        data["Top1_Is_Accurate"] = pd.to_numeric(
            data["Top1_Is_Accurate"], errors="coerce"
        )

        data.replace(
            {"Normalized_Missing_Index": {np.inf: np.nan, -np.inf: np.nan}},
            inplace=True,
        )
        data.dropna(
            subset=["Top1_Is_Accurate", "Normalized_Missing_Index"],
            inplace=True,
        )
        data["Top1_Is_Accurate"] = data["Top1_Is_Accurate"].astype(int)

        return data
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None


def calculate_missing_index_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical measures for normalized missing index and accuracy."""
    stats = (
        data.groupby("Normalized_Missing_Index")
        .agg({"Top1_Is_Accurate": ["mean", "std", "count", "sem"]})
        .reset_index()
    )
    stats.columns = [
        "Normalized_Missing_Index",
        "Accuracy_Mean",
        "Accuracy_Std",
        "Sample_Count",
        "Accuracy_SEM",
    ]
    return stats


def perform_logistic_regression(
    data: pd.DataFrame, formula: str
) -> Optional[Any]:
    """Perform logistic regression analysis with the given formula."""
    try:
        model = smf.logit(formula, data=data).fit()
        logger.info("Logistic regression completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error in logistic regression: {str(e)}")
        return None


def perform_linear_regression(
    data: pd.DataFrame, formula: str
) -> Optional[Any]:
    """Perform linear regression analysis with the given formula."""
    try:
        model = smf.ols(formula, data=data).fit()
        logger.info("Linear regression completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error in linear regression: {str(e)}")
        return None
