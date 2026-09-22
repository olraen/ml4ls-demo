# ml4ls-demo
[![DOI](https://zenodo.org/badge/1380130357.svg)](https://doi.org/10.5281/zenodo.22895773)

# BMI prediction with logistic Lasso

Simple machine-learning example for predicting BMI > 30 from
age, gender and gene-expression measurements.

## Data

`data/data_bmi.tsv`

Variables:

- `id`: sample identifier
- `BMI>30`: binary outcome
- `age`: age in years
- `gender`: categorical variable
- remaining columns: normalized gene-expression measurements

The dataset is a teaching dataset.

## Analysis

1. Split data into 80% training and 20% test data.
2. Standardize numeric predictors.
3. Encode gender.
4. Fit logistic Lasso.
5. Select the regularization parameter using 5-fold cross-validation.
6. Select the five genes with the largest non-zero coefficients.
7. Fit a final logistic regression using age, gender and the five genes.
8. Evaluate the final model on the held-out test set.

## Reproduce the analysis

Install dependencies:
```bash
uv sync
````

Train the model
```bash
uv run python train_lasso.py
```

Run the Streamlit application
```bash
uv run streamlit run app.py
```

## Files

- explore_data.py – basic data exploration
- train_lasso.py – model training and evaluation
- app.py – Streamlit prediction application
- bmi_model.joblib – saved fitted model
- pyproject.toml – project dependencies
- uv.lock – exact dependency versions

## Software

Python dependencies and exact versions are recorded in `pyproject.toml` and `uv.lock`.

## License

MIT License.
