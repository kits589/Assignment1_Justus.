# Car Price Drivers: An Analysis Pipeline

Uganda Christian University · Faculty of Engineering, Design and Technology
**Data Science Lifecycle (MSDS-1) — Advent 2026, Assignment One**


Exploratory data analysis and feature selection on `automobileEDA.csv` (201 cars,
29 attributes) to identify the variables that most strongly influence car price.

The notebook is structured as a **reproducible pipeline** rather than a script.
Everything configurable lives in one `CONFIG` dictionary; each stage is a function
that takes a dataframe and returns a dataframe; and every decision a stage makes is
written to a run log printed in full at the end of section 1.

```
load  ->  profile  ->  repair  ->  derive  ->  audit  ->  publish
                                                            |
                              describe -> group -> test -> model
```

## Contents

| Path | Description |
|---|---|
| `Assignment1_Justus.ipynb` | Main notebook — all five questions, code and a written interpretation of every output |
git | `automobileEDA.csv` | Dataset as provided |
| `Ariho_Data.csv` | Cleaned dataset produced by section 1 (27 columns) |
| `data/raw/` | Untouched copy of the source data |
| `data/processed/` | Cleaned data written by the notebook |
| `figures/` | 21 charts exported as PNG (`chart_01` … `chart_21`) |
| `requirements.txt` | Python dependencies |

## Notebook structure

1. **Pipeline stages** — load, profile with a role assertion, repair missing values,
   derive the horsepower band, audit duplicates / outliers / redundancy, publish
   `Ariho_Data.csv` with a printed decision log.
2. **Descriptive statistics** — summary statistics with coefficient of variation,
   all 67 categorical levels as one tidy frame, distributions, correlation with
   strength bands.
3. **Grouping** — average price by group with bootstrap 95% confidence intervals, a
   two-way price grid, and ANOVA with both eta squared and bias-corrected omega
   squared.
4. **Effect of the independent attributes on price** — continuous variables with
   quartile step analysis, categorical variables with effect sizes, and
   `num-of-cylinders` against `aspiration`.
5. **Main price drivers** — a univariate filter (`SelectKBest`) and a model-based
   wrapper (permutation importance on gradient-boosted trees) compared, plus
   multicollinearity and a held-out evaluation.

## Methods used

| Step | Method |
|---|---|
| Missing `stroke` | Manufacturer median — the donor grouping was **chosen by measurement**, after a bore-based regression was tested and rejected at r = −0.055 |
| Horsepower band | Rebuilt with `pd.qcut` into four quantile bands (53 / 49 / 51 / 48 cars) |
| Outlier detection | 1st/99th percentile rule, distribution-free |
| Redundant columns | `city-L/100km` and `gas` **dropped** — both are provable functions of other columns |
| Group testing | ANOVA with eta squared and bias-corrected omega squared |
| Feature selection | `SelectKBest` (ANOVA F) and permutation importance on a `GradientBoostingRegressor`, measured on held-out data |

## Headline results

- **Both selection methods agree completely on the top four features**: engine size,
  curb weight, horsepower, highway fuel economy. Shuffling engine size alone costs
  the model a **0.53 drop in R²** — more than every other feature combined.
- **The methods disagree informatively below that.** `stroke` ranks 27th by F
  statistic and 5th by permutation importance; `peak-rpm` ranks 30th and 7th. A
  filter-only approach would have discarded both.
- Gradient boosting beats linear regression by **12 points of R² on a single
  feature**, because the relationships are curved — eight variables have a Spearman
  correlation materially higher than their Pearson.
- The four quantile horsepower bands separate price monotonically: 7,160 / 10,089 /
  12,836 / 23,463.

### A caveat the notebook states rather than hides

The final model reports a held-out R² of 0.979 and an MAE of 689. **Section 5.7
argues this is too good and explains why**: three of the seven specification-identical
pairs found in section 1.5 straddle the train/test split, the 51-car test set has a
mean price 1,640 below the dataset's, and a single split on 201 rows is a
high-variance estimate. The defensible claim is an R² in the high 0.80s to low 0.90s,
and a grouped split is named as the first fix.

## Running it

```bash
pip install -r requirements.txt
jupyter lab Assignment1_Justus.ipynb
```

Run all cells top to bottom. Every threshold is in `CONFIG` at the top, and all
stochastic steps use `CONFIG["seed"] = 314`, so re-running reproduces every number.
Changing a cleaning threshold means editing one line, not hunting through the
notebook.

## Dataset note

`automobileEDA.csv` is a pre-processed version of the UCI Automobile dataset. The
pipeline's audit stages document what had already been done to it: `length` and
`width` divided by their column maxima, `city-L/100km` equal to 235 / `city-mpg` to
within 4.4e-09, `diesel` and `gas` summing to 1 on all 201 rows, and mean-substituted
values in `normalized-losses` (122 on 41 rows) and two Renaults' `horsepower` and
`peak-rpm`.

**All 201 rows are retained; the column count falls from 29 to 27.** No rows were
deleted — only two cars exceed the 99th percentile of price (a BMW at 41,315 and a
Mercedes at 45,400) and both are genuine, while the extreme compression ratios are
the 20 diesels, whose range does not overlap the petrol cars' at all. Two columns
*were* dropped, because both are deterministic functions of columns that remain:
`city-L/100km` and `gas`. `city-mpg` was deliberately **not** dropped at this stage
despite correlating 0.97 with `highway-mpg` — 0.97 is a strong empirical relationship,
not an identity, so that exclusion belongs to the modelling step in section 5.1,
where it is a modelling decision rather than a claim about the data.
