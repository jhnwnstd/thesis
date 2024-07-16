# Thesis Project

Welcome to the repository for my thesis project. This repository contains all the code and works related to my thesis. Below, you'll find detailed instructions on how to set up and use this repository, including handling large files and managing dependencies.

## Table of Contents

- [Prerequisites](#prerequisites)
- [KenLM Dependency](#kenlm-dependency)
- [Installation](#installation)
- [Usage](#usage)
- [Large Files](#large-files)
- [Managing Dependencies](#managing-dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- [Git](https://git-scm.com/)
- [Python 3.x](https://www.python.org/)
- [Git Large File Storage (LFS)](https://git-lfs.github.com/)

## KenLM Dependency

This project requires KenLM, a faster and smaller language model query library. Follow these steps to set up KenLM:

1. Visit the [official KenLM website](https://kheafield.com/code/kenlm/) for detailed information.

2. Clone the KenLM repository:
   ```bash
   git clone https://github.com/kpu/kenlm.git
   cd kenlm
   ```

3. Build KenLM:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j 4
   ```

4. Add KenLM to your system PATH or update your project's configuration to point to the KenLM installation directory.

For more detailed instructions and advanced configuration options, please take a look at the [KenLM GitHub repository](https://github.com/kpu/kenlm).

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

## Usage

### Running the Code

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run your scripts:
   ```bash
   python path/to/your_script.py
   ```

## Large Files

This repository uses Git LFS to handle large files. The following file types are excluded using `.gitignore`:

- KenLM models (`*.klm`)
- ARPA files (`*.arpa`)
- PNG files (`*.png`)
- CSV files (`*.csv`)

### Removing Large Files

To remove large files from the repository history:

1. Install `git-filter-repo`:
   ```bash
   pip install git-filter-repo
   ```

2. Remove specific files:
   ```bash
   git filter-repo --path main/data/corpora/all_tokens_clmet.txt --path main/data/corpora/all_tokens_openEdges.txt --invert-paths --force
   ```

## Managing Dependencies

This project uses `pyenv` and `pip` to manage Python environments and dependencies.

1. Activate your `pyenv` environment:
   ```bash
   pyenv activate your-environment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
thesis/
├── main/
│   ├── data/
│   │   ├── corpora/
│   │   └── ...
│   ├── scripts/
│   └── ...
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request
