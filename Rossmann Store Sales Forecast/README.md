# Rossmann Store Sales Forecast

## Project Overview

This project explores and models daily sales for Rossmann stores. The goal is to understand the main drivers of sales, identify useful patterns across holidays, promotions, stores, and dates, and compare several machine learning models for predicting sales.

The project is built around two notebooks:

- `Data Exploratory.ipynb`: cleans the training data and investigates sales patterns.
- `Models.ipynb`: trains classification and regression models to estimate sales.

The strongest conclusion from the analysis is that customer traffic is the dominant sales driver. Promotions also increase both customer count and sales, while public holidays, Easter, Christmas, and Sundays are associated with very low sales.

## Codes and Resources Used

Editor Used: Jupyter Notebook

Python Version: Python `3.11.6`

### Python Packages Used

General Purpose:

- `os` or standard Python utilities may be used for local file handling if the project is extended.

Data Manipulation:

- `pandas`
- `numpy`

Data Visualization:

- `matplotlib`
- `seaborn`
- `mlxtend.plotting`

Machine Learning:

- `scikit-learn`
- `LinearRegression`
- `KNeighborsClassifier`
- `RandomForestRegressor`
- `RandomForestClassifier`
- `LassoCV`
- `RidgeCV`
- `train_test_split`
- `r2_score`
- `accuracy_score`
- `mean_absolute_error`
- `mean_squared_error`
- `confusion_matrix`

## Data

### Source Data

The project uses Rossmann store sales data from the Kaggle Rossmann Store Sales competition:

- Source: <https://www.kaggle.com/c/rossmann-store-sales>

Files included in this project:

- `Rossmann_train.csv`: historical store-level daily sales data. This is the main dataset used in both notebooks.
- `Rossman_test.csv`: test data without the `Sales` target. This file is included but is not used in the current notebook modeling workflow.
- `Rossmann_store.csv`: store metadata, including store type, assortment, competition information, and Promo2 fields. This file is included but is not merged into the current model notebooks.
- `rossmann_train_clean.csv`: cleaned output generated from the exploratory notebook.

Main fields used in the analysis:

- `Store`: unique store identifier.
- `Date`: sales date.
- `Sales`: daily turnover and the main prediction target.
- `Customers`: number of customers on a given day.
- `Open`: whether the store was open.
- `Promo`: whether the store was running a promotion.
- `StateHoliday`: state holiday indicator.
- `SchoolHoliday`: school holiday indicator.
- `DayOfWeek`: day-of-week numeric indicator.

The training data covers `2013-01-01` to `2015-07-31`, with `1,017,209` rows across `1,115` stores.

### Data Acquisition

The dataset can be obtained from Kaggle by downloading the Rossmann Store Sales competition files. After downloading, place the files in this folder with the same names used by the notebooks:

- `Rossmann_train.csv`
- `Rossman_test.csv`
- `Rossmann_store.csv`

The notebooks currently assume the CSV files are stored in the same folder as the notebooks.

### Data Preprocessing

The exploratory notebook applies these preprocessing steps:

- Converts `Date` to datetime format.
- Extracts `year`, `month`, `day`, and `day_of_week_name`.
- Creates `SalePerCustomer = Sales / Customers`.
- One-hot encodes `StateHoliday`.
- One-hot encodes `Promo` in the exploratory notebook.
- Fills missing values with `0`.
- Saves the cleaned dataset as `rossmann_train_clean.csv`.

Important preprocessing note: rows with `Customers = 0` create missing or infinite sales-per-customer values, so the notebook fills those derived missing values with `0`.

## Code Structure

- `Data Exploratory.ipynb`: performs data cleaning, feature creation, holiday analysis, promotion analysis, store-level analysis, top and bottom 20% store segmentation, time-based sales analysis, histograms, and correlation heatmaps.
- `Models.ipynb`: builds a KNN sales-band classifier, linear regression models, random forest regression, LassoCV, RidgeCV, and a final random forest section for short-window sales comparison.
- `Rossmann_train.csv`: original training data with the `Sales` target.
- `Rossman_test.csv`: test file without the `Sales` target.
- `Rossmann_store.csv`: supplemental store metadata.
- `rossmann_train_clean.csv`: cleaned training data exported by the exploratory notebook.
- `README.md`: project summary, setup notes, data description, results, and future work.

## Results and Evaluation

### Exploratory Results

Store open and closed behavior:

- `844,338` rows show stores open with positive sales.
- `54` rows show stores open but recording zero sales.
- `172,817` rows show stores closed with zero sales.
- No rows show stores closed with positive sales.

Holiday effects:

- Average sales on non-public-holiday rows: `5,885.25`.
- Average sales on public holidays: `290.74`.
- Average sales on Easter-holiday rows: `214.31`.
- Average sales on Christmas-holiday rows: `168.73`.
- Average sales on school-holiday rows: `6,476.52`.
- Average sales on non-school-holiday rows: `5,620.98`.

Promotion effects:

- Average sales without promo: `4,406.05`.
- Average sales with promo: `7,991.15`.
- Average customers without promo: `517.82`.
- Average customers with promo: `820.10`.

Store-level results:

- Highest total sales store: `Store 262` with `19,516,842`.
- Lowest total sales store: `Store 307` with `2,114,322`.
- Highest average sales-per-customer store: `Store 842` at `16.16`.
- Lowest average sales-per-customer store: `Store 769` at `3.51`.

Top and bottom store segments:

- Top 20% average-sales cutoff: `7,019`.
- Bottom 20% average-sales cutoff: `4,224`.
- Stores above the top cutoff: `220`.
- Stores below the bottom cutoff: `223`.

Time-based patterns:

- Best sales month: December, with average sales of `6,826.61`.
- Weakest sales month: January, with average sales of `5,465.40`.
- Best day of month: day `30`, with average sales of `7,297.27`.
- Weakest day of month: day `1`, with average sales of `4,658.45`.
- Best weekday: Monday, with average sales of `7,809.04`.
- Weakest weekday: Sunday, with average sales of `204.18`.

Correlation results:

- `Customers` has the strongest correlation with `Sales`: `0.895`.
- `Open` is also strongly related to `Sales`: `0.678`.
- `SalePerCustomer` has a positive relationship with `Sales`: `0.657`.
- `DayOfWeek` has a notable negative relationship with `Sales`: `-0.462`.

### Model Results

KNN sales-band classifier:

- Features used: `Customers`, `Promo`.
- Sales bands: bad, medium, and excellent.
- Accuracy with `k = 9`: `0.9793`.

Linear Regression:

- Training R^2: `0.8549`.
- Test R^2: `0.8529`.
- MAE: `985.43`.
- RMSE: `1,474.89`.

Reduced Linear Regression:

- Removes separate `Public`, `Easter`, and `Christmas` variables while keeping `No_holiday`.
- Training R^2: `0.8548`.
- Test R^2: `0.8529`.
- MAE: `986.70`.
- RMSE: `1,475.11`.

Random Forest Regressor:

- Training R^2: `0.9930`.
- Test R^2: `0.9555`.
- MAE: `482.14`.
- RMSE: `811.32`.
- Most important feature: `Customers`, with feature importance of `0.8596`.

LassoCV:

- Optimal alpha: `0.01`.
- Training R^2: `0.8548`.
- Test R^2: `0.8529`.
- MAE: `986.70`.
- RMSE: `1,475.11`.

RidgeCV:

- Optimal alpha: `10.0`.
- Training R^2: `0.8548`.
- Test R^2: `0.8529`.
- MAE: `986.71`.
- RMSE: `1,475.11`.

The best model tested is the Random Forest Regressor, which clearly outperforms the linear models on test R^2, MAE, and RMSE.

### Evaluation Methodology

The notebooks mainly use an `80/20` train-test split with `random_state = 42`.

Regression models are evaluated using:

- R^2 score
- Mean Absolute Error
- Root Mean Squared Error

The KNN classifier is evaluated using:

- Accuracy score

Important caveat: the current models use `Customers`, which is highly predictive but may not be known in advance for a true future-sales forecast. Because of this, the current models are best interpreted as sales estimation models unless future customer counts are also forecast or removed from the feature set.

## Future Work

- Merge `Rossmann_store.csv` into the training data to use store type, assortment, competition distance, and Promo2 information.
- Build a true future-sales forecast model that does not rely on `Customers`.
- Replace random train-test splitting with time-based validation to better simulate real forecasting.
- Engineer lag features, rolling averages, holiday proximity features, and store-level historical sales features.
- Tune the Random Forest Regressor hyperparameters with cross-validation.
