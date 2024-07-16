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
   Note: When installing via pip, you can set the `MAX_ORDER` environment variable to control the max order with which KenLM is built.

4. Add the KenLM `build` directory to your system PATH or update your project's configuration to point to the KenLM installation directory.

### Additional Notes:

- KenLM supports reading ARPA files in compressed formats if compiled with the appropriate options:
  - HAVE_ZLIB: Supports gzip (link with -lz)
  - HAVE_BZLIB: Supports bzip2 (link with -lbz2)
  - HAVE_XZLIB: Supports xz (link with -llzma)

- KenLM offers two main data structures for querying: probing (fastest, uses most memory) and trie (least memory, slightly slower).

- For large-scale applications, you can create a binary format for faster loading:
  ```bash
  ./build_binary input.arpa output.binary
  ```

- KenLM has been tested on various platforms including x86_64, x86, PPC64, and ARM, and runs on Linux, OS X, Cygwin, and MinGW.

For more detailed information on usage, estimation, filtering, and benchmarks, refer to the [KenLM GitHub repository](https://github.com/kpu/kenlm) and the [official KenLM website](https://kheafield.com/code/kenlm/).

## Usage

The main scripts for this project are located in the `main` directory. There are two primary scripts you can run:

1. `qgram_type.py` (Main script):
   This script runs the analysis on word types.

2. `qgram_token.py`:
   This script is similar to `qgram_type.py` but runs the analysis on all words (tokens) instead of just word types.

To run the main script:

```bash
python main/qgram_type.py
```

To run the token-based analysis:

```bash
python main/qgram_token.py
```

Make sure you are in the root directory of the project when running these commands.

### Notes:
- `qgram_type.py` is the primary script for this project and should be used for most analyses.
- Use `qgram_token.py` when you need to analyze all word occurrences rather than just unique word types.
- Ensure that all necessary data files and dependencies are in place before running the scripts.
- You may need to adjust parameters or input files within the scripts depending on your specific analysis needs.

If you encounter any issues or need to modify the analysis, refer to the comments within each script for guidance on customization and troubleshooting.

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

## Project Structure

```
thesis/
├── csv_processing/
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
│ └── qgram_type.py
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

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for details.