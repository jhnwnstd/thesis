# English Orthography Predictability Analysis

This repository contains the code for my thesis project on analyzing the predictability of English orthography using n-gram models. The project aims to investigate factors influencing letter prediction accuracy in English words, providing insights into the regularity of English spelling.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [KenLM Dependency](#kenlm-dependency)
- [Usage](#usage)
  - [Running Experiments with qgram_type.py](#running-experiments-with-qgram_typepy)
  - [Running Experiments with qgram_token.py](#running-experiments-with-qgram_tokenpy)
  - [Analyzing Results](#analyzing-results)
- [Project Structure](#project-structure)
- [Large Files and Dependencies](#large-files-and-dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This thesis project explores the predictability of English orthography using computational methods. It employs n-gram models to predict missing letters in English words, analyzing various factors such as word length, missing letter position, and vowel presence. The research utilizes a diverse range of datasets, including the CMU Pronouncing Dictionary, Brown Corpus, and several historical corpora, to provide a comprehensive view of English orthographic patterns across different time periods and text types.

Key features of this project include:
- Implementation of n-gram models for letter prediction
- Analysis of word types to minimize frequency biases and focus on underlying orthographic structures
- Visualization techniques and statistical analysis of results

## Prerequisites

Ensure you have the following software installed on your system:

- [Git](https://git-scm.com/) (version 2.25.0 or newer)
- [Python](https://www.python.org/) (version 3.11 or newer)
- [Git Large File Storage (LFS)](https://git-lfs.github.com/) (version 2.13.0 or newer)

It's recommended to use a Unix-like environment (Linux or macOS) for optimal compatibility. If using Windows, consider using Windows Subsystem for Linux (WSL) or Git Bash.

## Installation

Follow these steps to set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/jhnwnstd/thesis.git
   cd thesis
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install KenLM as a Python module:
   ```bash
   export MAX_ORDER=7 
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```
   Note: The `MAX_ORDER` environment variable controls the maximum n-gram order. Adjust this value if needed, but ensure it matches the value used in KenLM compilation (see next section).

## KenLM Dependency

This project heavily relies on KenLM, a fast and efficient language model toolkit developed by Kenneth Heafield. To set up KenLM:

1. Clone the KenLM repository:
   ```bash
   git clone https://github.com/kpu/kenlm.git
   cd kenlm
   ```

2. Create a build directory and compile with optimizations:
   ```bash
   mkdir -p build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=7
   make -j$(nproc)
   ```
   Note: The `-DKENLM_MAX_ORDER=7` flag sets the maximum n-gram order to 7. Adjust this value if you need a different maximum order, but ensure it matches the `MAX_ORDER` used in the Python module installation.

3. Add the KenLM `build` directory to your system PATH or update your project's configuration to point to the KenLM installation directory.

For more detailed information on KenLM usage, refer to the [KenLM GitHub repository](https://github.com/kpu/kenlm) and the [official KenLM website](https://kheafield.com/code/kenlm/).

## Usage

The main scripts for this project are located in the `main` directory. There are two primary scripts you can run:

### Running Experiments with `qgram_type.py`

The `qgram_type.py` script is the main script for running q-gram analysis on word types. To run an experiment:

1. Open `qgram_type.py` in a text editor or IDE.
2. Locate the `Config` class and modify the desired values.
3. Save the file.
4. Run the script from the command line:
   ```bash
   python main/qgram_type.py
   ```

#### Changing Configuration Values

To run experiments with different settings, modify the following values in the `Config` class:

1. `seed`: Set the random seed for reproducibility (default: 42)
2. `q_range`: Set the range of q-gram sizes to analyze [min, max] (default: [7, 7])
3. `split_config`: Set the train-test split ratio (default: 0.5)
4. `vowel_replacement_ratio`: Set the ratio of vowels to replace (default: 0.2)
5. `consonant_replacement_ratio`: Set the ratio of consonants to replace (default: 0.8)
6. `min_word_length`: Set the minimum word length to consider (default: 3)
7. `prediction_method_name`: Set the prediction method to use (default: 'context_sensitive')
8. `num_replacements`: Set the number of character replacements (default: 1)
9. `log_level`: Set the logging level (default: logging.INFO)

#### Experiment Examples

Here are some example modifications you can make to run different experiments:

1. Analyze different q-gram sizes:
   ```python
   self.q_range = [5, 8]  # This will analyze q-grams from size 5 to 8
   ```

2. Change the train-test split:
   ```python
   self.split_config = 0.7  # This will use 70% of the data for training
   ```

3. Adjust character replacement ratios:
   ```python
   self.vowel_replacement_ratio = 0.3
   self.consonant_replacement_ratio = 0.7
   ```

4. Change the prediction method:
   ```python
   self.prediction_method_name = 'another_prediction_method'  # Replace with an actual method name from your code
   ```

5. Increase the number of character replacements:
   ```python
   self.num_replacements = 2  # This will replace 2 characters instead of 1
   ```

### Running Experiments with qgram_token.py

The `qgram_token.py` script is similar to `qgram_type.py` but runs the analysis on all words (tokens) instead of just word types. Use this script when you need to analyze all word occurrences rather than unique word types. The configuration and usage are similar to `qgram_type.py`.

### Analyzing Results

After running either script:

1. Check the console output for immediate results.
2. Examine the log file in the `data/logs` directory for detailed information.
3. Review the CSV files in the `data/outputs/csv` directory for prediction details.
4. Check the text files in the `data/outputs/texts` directory for summary statistics.

## Project Structure

```
thesis/
├── analysis/
│ ├── 1_DATA_CLEANER.py       # Must be run first before any analysis
│ ├── GAMs_model.py
│ ├── histogram_confidences.py
│ ├── letter_accuracies.py
│ ├── letter_index_plot.py
│ ├── randomforest.py
│ ├── vowel_or_consonant.py
│ ├── bootstrap_confidences.py
│ ├── heat_map.py
│ ├── index_length_vowel.py
│ ├── letter_importances.py
│ ├── letter_index.py
│ ├── totalvsaccurate.py
│ └── word_length.py
├── main/
│ ├── data/
│ │ ├── corpora/
│ │ ├── logs/
│ │ ├── models/
│ │ └── outputs/
│ ├── corpus_class.py
│ ├── evaluation_class.py
│ ├── predictions_class.py
│ ├── qgram_token.py
│ └── qgram_type.py        # main script to run
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Main Directory Scripts

- `corpus_class.py`: Defines the Corpus class for handling text data.
- `evaluation_class.py`: Contains the Evaluation class for assessing model performance.
- `predictions_class.py`: Implements the Predictions class for making and managing predictions.
- `qgram_analysis.py`: Provides utility functions for q-gram analysis.
- `qgram_token.py`: Performs q-gram analysis on all words (tokens) in the corpus.
- `qgram_type.py`: Performs q-gram analysis on unique word types in the corpus.

### Analysis Directory

The analysis directory contains scripts for various data analysis tasks. Before running any analysis, ensure the data is formatted correctly by first executing `1_DATA_CLEANER.py`.

- `1_DATA_CLEANER.py`: Cleans and prepares raw data for further analysis. This script must be run first.
- `G

AMs_model.py`: Fits and analyzes Generalized Additive Models (GAMs).
- `randomforest.py`: Trains and evaluates Random Forest models.
- `histogram_confidences.py`: Creates histograms of confidence scores for predictions.
- `letter_index_plot.py`: Plots related to the position of letters.
- Other scripts perform various analyses and generate visualizations for the project.

## Large Files and Ignored Files

This repository uses Git LFS to handle large files. The following file types are tracked by Git LFS:

- Text files in the corpora directory (`main/data/corpora/*.txt`)

The following file types are excluded using `.gitignore`:

- KenLM models (`*.klm`)
- ARPA files (`*.arpa`)
- PNG files (`*.png`)
- CSV files (`*.csv`)

## Troubleshooting

If you encounter issues:

1. Ensure all prerequisites are correctly installed and up to date.
2. Ensure that KenLM is properly installed and configured. Check that the `DKENLM_MAX_ORDER` value used during compilation matches the `MAX_ORDER` environment value specified during the Python module installation.
3. Verify that input data files are in the correct locations within the `main/data/corpora/` directory.
4. Check the log files in the `data/logs` directory for detailed error messages and stack traces.
5. Ensure your Python virtual environment is correctly activated and all dependencies are installed.
6. If you're having issues with specific analyses, try running the `1_DATA_CLEANER.py` script again to ensure your data is properly formatted.

For persistent problems, please open an issue on the GitHub repository with a detailed description of the error, steps to reproduce it, and relevant parts of the log files.

## Contributing

Contributions are welcome. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for the full license text.