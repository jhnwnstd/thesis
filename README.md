# Thesis Project

This repository contains all the code and resources related to my thesis. Below, you'll find detailed instructions on how to set up and use this repository, including handling large files and managing dependencies.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [KenLM Dependency](#kenlm-dependency)
- [Usage](#usage)
- [Large Files](#large-files)
- [Managing Dependencies](#managing-dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Ensure you have the following installed:

- [Git](https://git-scm.com/) (2.25.0 or newer)
- [Python](https://www.python.org/) (3.8 or newer)
- [Git Large File Storage (LFS)](https://git-lfs.github.com/) (2.13.0 or newer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jhnwnstd/thesis.git
   cd thesis
   ```

2. Set up Git LFS:
   ```bash
   git lfs install
   git lfs track "main/data/corpora/*.txt"
   ```

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## KenLM Dependency

This project requires KenLM, a faster and smaller language model query library developed by Kenneth Heafield. To set up KenLM:

1. Clone the KenLM repository:
   ```bash
   git clone https://github.com/kpu/kenlm.git
   cd kenlm
   ```

2. Create a build directory and compile:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j 4
   ```

3. To use KenLM as a Python module:
   ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```
   Note: When installing via pip, you can set the `MAX_ORDER` environment variable to control the max order with which KenLM is built. This is essential to run the analysis for more than 6 character grams. If you run into difficulties compiling it, then default to 6.

4. Add the KenLM `build` directory to your system PATH or update your project's configuration to point to the KenLM installation directory.

For more detailed information on usage refer to the [KenLM GitHub repository](https://github.com/kpu/kenlm) and the [official KenLM website](https://kheafield.com/code/kenlm/).

## Usage

The main scripts for this project are located in the `main` directory. There are two primary scripts you can run:

1. `qgram_type.py` (Main script):
   This script runs the analysis on word types.

2. `qgram_token.py`:
   This script is similar to `qgram_type.py` but runs the analysis on all words (tokens) instead of just word types.

### Running Experiments with qgram_type.py

The `qgram_type.py` script is the main script for running q-gram analysis on word types. You can modify various parameters in the `Config` class to run different experiments.

#### Changing Configuration Values

To run experiments with different settings, modify the following values in the `_set_values` method of the `Config` class:

1. `self.seed`: Set the random seed for reproducibility (default: 42)
2. `self.q_range`: Set the range of q-gram sizes to analyze [min, max] (default: [7, 7])
3. `self.split_config`: Set the train-test split ratio (default: 0.5)
4. `self.vowel_replacement_ratio`: Set the ratio of vowels to replace (default: 0.2) # Vowel and Consonant ratio must sum to 1.0
5. `self.consonant_replacement_ratio`: Set the ratio of consonants to replace (default: 0.8) # Vowel and Consonant ratio must sum to 1.0
6. `self.min_word_length`: Set the minimum word length to consider (default: 3)
7. `self.prediction_method_name`: Set the prediction method to use (default: 'context_sensitive')
8. `self.num_replacements`: Set the number of character replacements (default: 1)
9. `self.log_level`: Set the logging level (default: logging.INFO)

#### Running the Script

To run the script with modified values:

1. Open `qgram_type.py` in a text editor.
2. Locate the `Config` class and modify the desired values in the `_set_values` method.
3. Save the file.
4. Run the script from the command line:

```bash
python main/qgram_type.py
```

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

#### Analyzing Results

After running the script:

1. Check the console output for immediate results.
2. Examine the log file in the `data/logs` directory for detailed information.
3. Review the CSV files in the `data/outputs/csv` directory for prediction details.
4. Check the text files in the `data/outputs/texts` directory for summary statistics.

### Additional Notes

- `qgram_type.py` is the primary script for this project and should be used for most analyses.
- Use `qgram_token.py` when you need to analyze all word occurrences rather than just unique word types.
- Ensure that all necessary data files and dependencies are in place before running the scripts.
- You may need to adjust parameters or input files within the scripts depending on your specific analysis needs.

If you encounter any issues or need to modify the analysis, refer to the comments within each script for guidance on customization and troubleshooting.

Remember to document any changes you make to the configuration when reporting your results to ensure reproducibility of your experiments.

## Large Files

This repository uses Git LFS to handle large files. The following file types are tracked by Git LFS:

- Text files in the corpora directory (`main/data/corpora/*.txt`)

The following file types are excluded using `.gitignore`:

- KenLM models (`*.klm`)
- ARPA files (`*.arpa`)
- PNG files (`*.png`)
- CSV files (`*.csv`)

### Removing Large Files from History

If you need to remove large files from the repository history:

1. Install `git-filter-repo`:
   ```bash
   pip install git-filter-repo
   ```

2. Remove specific files:
   ```bash
   git filter-repo --path main/data/corpora/all_tokens_clmet.txt --path main/data/corpora/all_tokens_openEdges.txt --invert-paths --force
   ```

## Managing Dependencies

This project uses `pip` for package management. To update dependencies:

1. Ensure your virtual environment is activated.
2. Update dependencies:
   ```bash
   pip install -r requirements.txt
   ```

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
│ ├── totalvsaccruate.py
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
│ ├── qgram_analysis.py
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

#### Modeling and Analysis:
- `GAMs_model.py`: Fits and analyzes Generalized Additive Models (GAMs).
- `randomforest.py`: Trains and evaluates Random Forest models.

#### Data Visualization and Plotting:
- `histogram_confidences.py`: Creates histograms of confidence scores for predictions.
- `letter_index_plot.py`: Plots related to the position of letters

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for details.