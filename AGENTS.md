# Repository Guidelines

## Project Structure & Module Organization

- `data/` holds training data, validation inputs, prediction templates, and assessment notes in `README.md`.
- `notebooks/data.ipynb` contains initial data exploration. Its relative paths expect execution from `notebooks/`.
- `notebooks/score.py` validates submission CSVs and generates the December prediction chart; it does not compute accuracy metrics.
- `requirements.txt` declares NumPy, pandas, and Matplotlib dependencies. There is no package, build system, or test directory yet.

Use actual filenames: `train-test.csv`, `validation.csv`, `validation-predictions-template.csv`, and `december-chart-inputs.csv`. The README uses outdated underscore filenames and references an assessment PDF absent from this workspace.

## Build, Test, and Development Commands

Run setup from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebook also imports Seaborn and Plotly, which are currently undeclared. To run it locally:

```bash
python -m pip install jupyter seaborn plotly
cd notebooks
jupyter notebook data.ipynb
```

Validate completed prediction files from the repository root:

```bash
python notebooks/score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

This creates `scorer_results/candidate_december.png`. Save completed December predictions separately to preserve the supplied inputs.

## Coding Style & Naming Conventions

Follow the scorer's Python style: four-space indentation, `snake_case` functions and variables, `UPPER_SNAKE_CASE` constants, and type hints for reusable functions. Use `pathlib.Path` for filesystem paths. Keep notebooks compact and executable in order; set random seeds for reproducible splits and models. No formatter or linter is configured.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Re-run notebooks from a fresh kernel and validate generated submissions with the scorer. Validation output requires exactly 12,000 unique expected load IDs and positive finite rates. December output requires 31 daily rows with the original column order and fixed route inputs. Report model quality separately using a documented, leakage-aware split of training data.

## Commit & Pull Request Guidelines

This workspace has no Git history to establish commit conventions. When versioned, use short imperative messages such as `Add baseline rate model`. PRs should describe changes, dependencies, validation commands, split strategy, and measured results. Include chart previews when changing plotting behavior; avoid committing virtual environments or temporary outputs.
