# MNIST Ensemble Benchmark

A reproducible machine learning benchmark for handwritten digit classification. The project compares Decision Tree, Random Forest, Extra Trees, and XGBoost models on a 10,000-row MNIST sample, selects hyperparameters with stratified cross-validation, and evaluates the selected models on an untouched test set.

## Highlights

- Validates the input schema, labels, pixel ranges, and missing values.
- Uses stratified train/test splitting and cross-validation to prevent test-set leakage.
- Supports vectorized image translation for optional data augmentation.
- Saves metrics, classification reports, plots, and the best fitted model.
- Provides a command-line interface for reproducible experiments.
- Includes unit tests for data validation and augmentation.

## Technologies

- Python 3.10+
- pandas and NumPy
- scikit-learn
- XGBoost
- Matplotlib
- pytest and Ruff

## Project structure

```text
mnist-ensemble-benchmark/
|-- data/
|   `-- mnist_sample.csv
|-- src/mnist_benchmark/
|   |-- cli.py
|   |-- data.py
|   `-- experiment.py
|-- tests/
|-- outputs/               # Generated locally and ignored by Git
|-- pyproject.toml
`-- README.md
```

## Installation

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/YOUR_USERNAME/mnist-ensemble-benchmark.git
   cd mnist-ensemble-benchmark
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the project:

   ```bash
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

## Usage

Run a quick Decision Tree experiment:

```bash
mnist-benchmark --models decision_tree --sample-limit 1000
```

Run the complete benchmark:

```bash
mnist-benchmark --models decision_tree random_forest extra_trees xgboost
```

Run all models with translation-based data augmentation:

```bash
mnist-benchmark --augment
```

Useful options:

```text
--data PATH          Use another MNIST-format CSV file
--output-dir PATH    Select the artifact directory
--sample-limit N     Run on a stratified subset
--cv-folds N         Set the number of cross-validation folds
--n-jobs N           Control parallel model training
--random-state N     Make the experiment reproducible
```

The command creates `outputs/metrics.json`, `outputs/metrics.csv`, diagnostic plots, a classification report, and `outputs/best_model.joblib`.

## Data format

The included CSV contains 10,000 rows and 785 columns without a header. The first column is the digit label (`0`-`9`), followed by 784 grayscale pixel values (`0`-`255`) representing a `28 x 28` image.

## Quality checks

```bash
pytest
ruff check .
```

## Project status

Portfolio-ready. The pipeline is complete and can be extended with additional estimators, experiment tracking, or a larger MNIST dataset.
