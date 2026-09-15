# Loan Default Risk Predictor
3MTT Capstone Project

## What this project does
Predicts whether a loan applicant is likely to default, using a machine
learning classification pipeline built in Python.

## Project structure
```
loan_default_predictor/
├── data/
│   └── loan_data.csv        <- put your downloaded dataset here (see below)
├── src/
│   ├── 01_eda.py                     <- explore the data
│   ├── 02_preprocessing.py           <- clean + engineer features
│   ├── 03_train_model.py             <- train baseline + stronger models
│   ├── 04_evaluate.py                <- evaluate and interpret results
│   └── 05_predict_new_applicant.py   <- score a single new applicant
├── outputs/                          <- saved models, plots, cleaned data land here
├── Loan_Default_Predictor.ipynb      <- presentation notebook (already run, with outputs)
├── requirements.txt
└── README.md
```

## Step 1 — Dataset
The real dataset is already in place at `data/loan_data.csv` (from
https://www.kaggle.com/datasets/nikhil1e9/loan-default — 255,347 rows,
no missing values, ~11.6% default rate). The pipeline and notebook have
both been run on it already; see Results below.

If you want to try a different dataset later, drop it in as
`data/loan_data.csv` and check the column names against `TARGET_COLUMN`
and `ID_COLUMNS` at the top of each script in `src/`.

A small synthetic sample (`data/sample_data.csv`) is also included for
quick pipeline testing.

## Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

## Step 3 — Run the pipeline in order
```bash
python src/01_eda.py
python src/02_preprocessing.py
python src/03_train_model.py
python src/04_evaluate.py
```

Each script prints its findings to the console and saves any plots/outputs
to the `outputs/` folder.

## Scoring a new applicant
Once you've trained the models (steps above), edit the `new_applicant`
dictionary near the top of `src/05_predict_new_applicant.py` with real
details, then run:
```bash
python src/05_predict_new_applicant.py
```
It prints a risk probability and recommendation from both models.

## Presentation notebook
`Loan_Default_Predictor.ipynb` walks through the entire project — EDA,
cleaning, training, evaluation, feature importance — in one notebook with
narrative explanations, already executed with all plots embedded. Open it
with `jupyter notebook` or `jupyter lab`, or use it directly for your
capstone presentation/demo. It currently runs on the synthetic
`sample_data.csv`; swap the data-loading cell to your real Kaggle file and
re-run all cells once you have it.

## Results (on the real dataset)
255,347 applicants, 11.6% default rate.

| Model | ROC-AUC | Recall (Default) | Precision (Default) |
|---|---|---|---|
| Logistic Regression | 0.762 | 0.70 | 0.23 |
| Random Forest | 0.755 | 0.63 | 0.25 |

Logistic Regression catches more actual defaulters (higher recall) at
the cost of more false alarms; Random Forest is slightly more precise
but misses more real defaulters. Which trade-off is "better" depends on
what a missed default costs a lender versus a false alarm — worth
discussing in your writeup.

Top predictors of default (Random Forest feature importance): **Age**,
**InterestRate**, **loan_to_income_ratio**, **MonthsEmployed**, and
**Income**.

## Why each step matters (for your writeup)
- **EDA**: understand class imbalance and spot data quality issues before
  modeling — skipping this is the #1 beginner mistake.
- **Preprocessing**: models can't handle missing values or raw text
  categories, and unscaled features can distort some algorithms.
- **Feature engineering**: ratios like debt-to-income often carry more
  signal than raw income or raw debt alone.
- **Two models**: Logistic Regression gives an interpretable baseline;
  Random Forest usually captures non-linear patterns and improves recall
  on the minority (default) class.
- **Evaluation beyond accuracy**: with an imbalanced target, a model that
  always predicts "no default" can still score 80%+ accuracy while being
  useless. Precision, recall, F1, and ROC-AUC tell the real story.
