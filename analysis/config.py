from pathlib import Path
from typing import Dict

# Shared dataset paths used across all analysis scripts
DATASET_PATHS: Dict[str, Path] = {
    "CLMET3": Path(
        "main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv"
    ),
    "Lampeter": Path(
        "main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv"
    ),
    "Edges": Path(
        "main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv"
    ),
    "CMU": Path(
        "main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv"
    ),
    "Brown": Path(
        "main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv"
    ),
}

# Vowel/consonant sets -- basic ASCII set used by most scripts
VOWELS_BASIC = frozenset("aeiouy")

# Extended set with accented/historical characters used by vowel_or_consonant.py
VOWELS_EXTENDED = frozenset("aeèéiîouyæœ")
CONSONANTS_EXTENDED = frozenset("bcdfghjklmnpqrstvwxzȝ")

# Morpheme data path (add_features.py)
MORPHEME_DATA_PATH = Path("main/data/MorphoLEX_en.xlsx")
