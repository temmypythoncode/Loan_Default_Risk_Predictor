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

## Step 1 — Get the dataset
Download one of these from Kaggle and save it as `data/loan_data.csv`:

- https://www.kaggle.com/datasets/nikhil1e9/loan-default (recommended — clean, beginner-friendly)
- https://www.kaggle.com/datasets/yasserh/loan-default-dataset
- https://www.kaggle.com/datasets/himelsarder/loan-default-risk-prediction-dataset

The scripts expect a CSV with a binary target column indicating default
(commonly named `Default`, `default`, or `loan_status`). Open the file
once you download it and check the column name — you may need to update
`TARGET_COLUMN` at the top of `02_preprocessing.py`.

**Note:** a small synthetic sample (`data/sample_data.csv`) is included so
you can run the whole pipeline right now and confirm everything works,
before swapping in the real Kaggle data.

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
