# Melbourne Rain Forecast

## Project Overview

This project explores Melbourne weather observations and builds models to forecast whether rain occurs. The analysis starts with high-frequency weather records, cleans and resamples them into 30-minute intervals, validates rainfall and temperature patterns against Bureau of Meteorology monthly files, and compares classification and time-series approaches for rain forecasting.

The project is built around two notebooks:

- `Data Exploratory.ipynb`: cleans the weather observations, validates rainfall and temperature against monthly Bureau of Meteorology data, and explores weather patterns.
- `Models.ipynb`: creates a binary rain/no-rain target and compares KNN, Random Forest, Logistic Regression, and ARIMA approaches.

The main modeling task is binary classification:

- `0`: no rain, where `RainTrace = 0`.
- `1`: rain, where `RainTrace > 0`.

The best overall classifier in the notebooks is Logistic Regression, which reaches `0.813` accuracy on the 2023 test period and improves precision compared with the other tested classifiers.

## Installation and Setup

1. Clone or download this repository.
2. Open the project folder:

```powershell
cd "C:\Users\Loc-kun\Documents\GitHub\Loc-s-projects-\Melbourne Rain Forecast"
```

3. Create and activate a Python environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

4. Install the required packages:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend statsmodels xgboost jupyter
```

5. Launch Jupyter Notebook:

```powershell
jupyter notebook
```

6. Run the notebooks in this order:

- `Data Exploratory.ipynb`
- `Models.ipynb`

The notebooks assume the CSV files are stored in the same folder as the notebooks.

## Codes and Resources Used

Editor Used: Jupyter Notebook

Python Version: Python `3.11.6`

### Python Packages Used

General Purpose:

- Standard Python utilities for local file handling if the project is extended.

Data Manipulation:

- `pandas`
- `numpy`

Data Visualization:

- `matplotlib`
- `seaborn`
- `matplotlib.ticker`
- `mlxtend.plotting`

Machine Learning:

- `scikit-learn`
- `KNeighborsClassifier`
- `RandomForestClassifier`
- `RandomForestRegressor`
- `LogisticRegression`
- `SVC`
- `XGBClassifier`
- `StandardScaler`
- `train_test_split`
- `accuracy_score`
- `precision_score`
- `recall_score`
- `confusion_matrix`
- `mean_absolute_error`
- `mean_squared_error`

Time-Series Modeling:

- `statsmodels`
- `adfuller`
- `ARIMA`
- `plot_acf`
- `plot_pacf`

Local Helper Code:

- `logitplots.py`: contains helper functions for plotting confusion matrices, decision boundaries, scatter plots, and correlation matrices.

## Data

### Source Data

The project uses Melbourne weather data for Bureau of Meteorology station `086338`, Melbourne (Olympic Park).

Official source references:

- Bureau of Meteorology climate statistics for station `086338`: <https://www.bom.gov.au/climate/averages/tables/cw_086338_All.shtml>
- Bureau of Meteorology latest observations page for Melbourne (Olympic Park): <https://www.bom.gov.au/products/IDV60801/IDV60801.95936.shtml>
- Bureau of Meteorology Climate Data Online: <http://www.bom.gov.au/climate/data/>

Files included in this project:

- `Melbourne01.csv`: high-frequency Melbourne weather observations used as the main modeling dataset.
- `IDCJAC0001_086338_Data1.csv`: monthly precipitation total data from the Bureau of Meteorology.
- `IDCJAC0002_086338_Data1.csv`: monthly mean maximum temperature data from the Bureau of Meteorology.

Main fields used from `Melbourne01.csv`:

- `Year`: year of the observation.
- `Month`: month of the observation.
- `Day`: day of the observation.
- `Hour`: hour of the observation.
- `Min`: minute of the observation.
- `Temp`: air temperature.
- `AppTemp`: apparent temperature.
- `DewPt`: dew point.
- `RelHum`: relative humidity.
- `WindDir`: wind direction.
- `WindSpeed`: wind speed.
- `WindGust`: wind gust.
- `PressMSL`: mean sea level pressure.
- `RainTrace`: rainfall trace in millimetres.

The raw `Melbourne01.csv` file contains `1,072,135` rows and `14` columns. After dropping duplicates, cleaning invalid records, resampling to 30-minute intervals, and interpolation, the modeling dataset contains `213,619` rows from `2011-01-01 00:00:00` to `2023-03-09 09:00:00`.

### Data Acquisition

The monthly validation files can be obtained from the Bureau of Meteorology Climate Data Online service for Melbourne (Olympic Park), station `086338`.

To reproduce the current notebooks:

- Place `Melbourne01.csv` in the project folder.
- Place `IDCJAC0001_086338_Data1.csv` in the project folder.
- Place `IDCJAC0002_086338_Data1.csv` in the project folder.
- Keep the filenames unchanged, because the notebooks read these names directly.

### Data Preprocessing

The notebooks apply these preprocessing steps:

- Drop duplicate weather observation rows.
- Remove rows where `WindDir` is not one of the expected compass directions.
- Remove invalid `RainTrace` rows where the value is `-`.
- Remove rows with invalid sentinel values such as `-9999.0` in wind speed, wind gust, and pressure.
- Remove rows where `RelHum = 0.0`.
- Convert `RainTrace` from object/string values to numeric values.
- Convert wind direction labels into degrees.
- Create a `date_time` index from `Year`, `Month`, `Day`, `Hour`, and `Min`.
- Resample observations into 30-minute intervals.
- Interpolate missing values after resampling.
- Create a binary target column called `bin_code`, where `0` means no rain and `1` means rain.

Key preprocessing results:

- Rows after removing duplicates: `536,576`.
- Rows after filtering invalid weather records: `527,142`.
- Rows after 30-minute resampling: `213,619`.
- Missing 30-minute intervals before interpolation: `5,819`.
- Missing values after interpolation: `0`.
- No-rain rows after resampling: `155,959`, or about `73.0%`.
- Rain rows after resampling: `57,660`, or about `27.0%`.

## Code Structure

- `Data Exploratory.ipynb`: performs cleaning, resampling, interpolation, monthly rainfall comparison, monthly temperature comparison, and exploratory charts.
- `Models.ipynb`: creates the binary rain target, splits training and testing data by year, trains KNN, Random Forest, Logistic Regression, and ARIMA models, and evaluates predictions.
- `logitplots.py`: helper plotting functions used for confusion matrices, decision boundaries, and correlation heatmaps.
- `Melbourne01.csv`: main high-frequency weather observation dataset.
- `IDCJAC0001_086338_Data1.csv`: Bureau of Meteorology monthly precipitation totals.
- `IDCJAC0002_086338_Data1.csv`: Bureau of Meteorology monthly mean maximum temperature.
- `README.md`: project summary, setup notes, data description, results, and future work.

## Results and Evaluation

### Exploratory Results

Rain distribution:

- Raw data no-rain share: about `67.6%`.
- Resampled no-rain share: about `73.0%`.
- Resampled rain share: about `27.0%`.
- Average `RainTrace` after resampling: `0.802` mm.
- Maximum `RainTrace` after resampling: `54.6` mm.

Monthly validation against Bureau of Meteorology data:

- Monthly rainfall comparison rows: `116`.
- Correlation between calculated monthly `RainTrace` and BOM monthly precipitation total: `0.989`.
- Mean absolute difference between calculated monthly `RainTrace` and BOM monthly precipitation total: `1.58` mm.
- Monthly temperature comparison rows: `118`.
- Correlation between calculated monthly temperature and BOM monthly mean maximum temperature: `0.986`.
- Mean absolute difference between calculated monthly temperature and BOM monthly mean maximum temperature: `0.96` degrees Celsius.

Correlation with `RainTrace`:

- `RelHum`: `0.250`
- `DewPt`: `0.115`
- `WindDir`: `0.094`
- `Temp`: `-0.101`
- `PressMSL`: `-0.187`

The strongest positive relationship with rain is relative humidity. Mean sea level pressure has the strongest negative relationship, which supports the notebook insight that rainfall is more common when pressure is lower.

Pressure investigation:

- Rain observations with `PressMSL < 1000`: `1,811`.
- No-rain observations with `PressMSL < 1000`: `1,599`.
- Rain observations with `PressMSL > 1000`: `55,134`.
- No-rain observations with `PressMSL > 1000`: `151,730`.

The notebook concludes that rainfall occurs more often under low-pressure conditions, especially below roughly `1000` hPa.

### Model Results

Training and testing split:

- Training period: all rows where `Year < 2023`.
- Testing period: all rows where `Year >= 2023`.
- Training rows: `210,384`.
- Testing rows: `3,235`.
- Training no-rain rows: `153,383`.
- Training rain rows: `57,001`.
- Testing no-rain rows: `2,576`.
- Testing rain rows: `659`.

KNN classifier:

- Uses all weather features except `RainTrace` and `bin_code`.
- Best notebook setting: `n_neighbors = 8`.
- Accuracy: `0.829`.

Reduced KNN classifier:

- Uses only `RelHum` and `PressMSL`.
- Accuracy: `0.767`.

Random Forest classifier:

- Tested `50`, `100`, `150`, and `200` estimators.
- Accuracy scores: `0.827`, `0.829`, `0.830`, `0.830`.
- Notebook conclusion: increasing the number of estimators did not materially change performance.
- Final reported Random Forest accuracy using `50` estimators: `0.827`.

Random Forest feature importance:

- `RelHum`: `0.135`
- `PressMSL`: `0.131`
- `WindDir`: `0.093`
- `Temp`: `0.091`
- `Day`: `0.079`
- `AppTemp`: `0.078`
- `DewPt`: `0.077`
- `Hour`: `0.068`

Reduced Random Forest using `RelHum` and `PressMSL`:

- Accuracy: `0.734`.
- Precision: `0.360`.
- Recall: `0.395`.
- Daily check over `65` sampled 8:30am observations had `17` wrong predictions, or about `26.2%`.

Logistic Regression:

- Uses standardized weather features.
- Solver: `lbfgs`.
- Penalty: `l2`.
- `C = 5`.
- Accuracy: `0.813`.
- Precision: `0.555`.
- Recall: `0.419`.
- Daily check over `65` sampled 8:30am observations had `14` wrong predictions, or about `21.5%`.

ARIMA:

- Augmented Dickey-Fuller p-value on training `RainTrace`: `0.0`.
- Forecast MAE: `1.001`.
- Forecast RMSE: `1.526`.
- The notebook concludes that ARIMA is not useful for this task because rainfall depends heavily on weather variables, not only the time trend of `RainTrace`.

### Evaluation Methodology

The classification models use a time-based split:

- Train on observations before 2023.
- Test on observations from 2023.

Classification models are evaluated using:

- Accuracy
- Precision
- Recall
- Confusion matrices

ARIMA is evaluated using:

- Mean Absolute Error
- Root Mean Squared Error
- Visual comparison of predicted and actual `RainTrace`

Important caveat: the dataset is imbalanced, with no-rain observations forming about `73%` of the resampled data. Accuracy alone can therefore overstate model quality. Precision and recall are important because the practical goal is identifying rain events, not simply predicting the majority no-rain class.

## Future Work

- Use class balancing, class weights, or resampling to improve rain-event recall.
- Tune KNN, Random Forest, and Logistic Regression hyperparameters with cross-validation.
- Add lag features for pressure, humidity, temperature, wind, and recent rainfall.
- Create rolling-window features such as previous 1-hour, 3-hour, and 24-hour rainfall.
- Compare additional models such as XGBoost, LightGBM, CatBoost, and calibrated probabilistic classifiers.
- Evaluate models with precision-recall curves, ROC-AUC, F1 score, and recall at useful probability thresholds.
- Forecast daily rain occurrence separately from half-hour rain occurrence.
- Export key plots from the notebooks and embed them in this README.
