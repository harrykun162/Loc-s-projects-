# Monash Solar Power Output Forecasting

## Project overview

This project investigates how weather conditions relate to, and can help predict, solar power output. The analysis uses timestamped solar generation data and Monash weather observations, then compares regression models across six solar series (Solar0–Solar5).

The work is contained in three Jupyter notebooks. It includes data preparation, exploratory analysis, feature screening, model fitting, held-out evaluation, and plots comparing predicted and observed output. The results show that a compact weather-based linear model is a strong baseline, while random forests provide small improvements on some series at the cost of a larger train/test performance gap.

## Project contents

| File | Purpose |
| --- | --- |
| `Data Exploratory.ipynb` | Loads and cleans the data, joins solar output to weather, and explores time patterns and relationships. |
| `ALL Models.ipynb` | Compares linear regression, random forest, and a neural network on six solar series. |
| `Best model.ipynb` | Consolidates model analysis, feature screening, and random forest experiments. |
| `phase_2_data.tsf` | Solar output time-series data used to construct the training data. |
| `nov_data.tsf` | Separate solar output time-series data used for held-out evaluation. |
| `monash_weather.csv` | Weather observations aligned to the solar timestamps. |
| `phase_1_data.tsf`, `finaltest.csv`, `NOV_test_data.csv` | Additional project data files used or retained with the analysis. |

## Analysis workflow

### 1. Prepare the time series

The notebooks convert the `.tsf` time-series files into timestamped observations and split the six Solar series into individual dataframes. Solar output and weather are aligned by timestamp. Weather timestamps are shifted by 10 hours in the model notebook, resampled to 15-minute intervals, rounded to one decimal place, and linearly interpolated where values are missing.

The training data is trimmed to remove an early period with large gaps in power output for Solar1–Solar5. The notebook retains Solar0 from its available start date and uses 1 October 2019 as the starting point for the other five series. The separate November dataset is prepared as the held-out evaluation period.

### 2. Explore patterns and candidate predictors

Exploratory plots examine the time series, weather/output relationships, and correlation heatmaps. Additional plots aggregate output and weather by time of day and month to inspect daily and seasonal patterns. Feature screening uses scikit-learn's `SelectKBest` with the regression F-test (`f_regression`).

The analysis highlights three weather variables as especially relevant to output:

- Surface solar radiation
- Relative humidity
- Temperature

These variables were also used as a compact three-feature input set in the reduced-feature model experiments.

### 3. Fit and evaluate models

The notebooks compare:

- **Linear regression**, applied through a standardization pipeline.
- **Random forest regression**, evaluated with a fixed random seed in the main comparison.
- **A feed-forward neural network**, trained separately for each series.

For the scikit-learn models, inputs exclude the target (`power_output`) and identifier/location fields (`solar_no`, UTC offset, and model elevation). Performance is reported using held-out R², RMSE, and MAE. Predictions are plotted against actual values, and negative predicted outputs are clipped to zero in comparison plots.

## Model results

### Held-out performance by series

The table reports saved held-out results from the notebooks. RMSE and MAE are in the same output units as `power_output` in the source data.

| Series | Linear R² | Linear RMSE | Linear MAE | Random forest R² | Random forest RMSE | Random forest MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solar0 | 0.816 | 6.981 | 5.039 | 0.785 | 7.541 | 4.408 |
| Solar1 | 0.850 | 1.651 | 1.212 | 0.835 | 1.735 | 1.171 |
| Solar2 | **0.861** | **1.589** | 1.045 | 0.860 | 1.596 | **0.985** |
| Solar3 | 0.781 | 1.785 | 1.180 | 0.730 | 1.983 | 1.283 |
| Solar4 | 0.842 | 1.004 | 0.661 | **0.857** | **0.956** | **0.597** |
| Solar5 | 0.853 | 4.938 | 3.256 | **0.855** | **4.890** | **2.716** |

### Neural-network held-out R²

The neural-network notebook prints R² under the label “Test accuracy.” Interpreted as R², its saved scores were:

| Series | Neural-network R² |
| --- | ---: |
| Solar0 | 0.827 |
| Solar1 | 0.819 |
| Solar2 | 0.843 |
| Solar3 | 0.685 |
| Solar4 | 0.825 |
| Solar5 | 0.243 |

The Keras `accuracy` value displayed alongside the loss is not a useful regression metric; the R² values above are the relevant comparison.

## Insights from the results

### Weather provides useful predictive signal

Linear regression achieved held-out R² values from **0.781 to 0.861** across all six series. The model captures a substantial share of output variation using aligned weather features, with the highest score on Solar2 and lowest on Solar3. This gives the project a clear, interpretable baseline for subsequent model work.

### More complex models do not improve every series

Random forest achieved its highest held-out R² on Solar4 (**0.857**) and Solar5 (**0.855**), narrowly outperforming linear regression on those series. Linear regression scored higher on Solar0, Solar1, and Solar3; Solar2 was effectively tied (0.861 vs. 0.860). Therefore, model choice should be made per series rather than assuming one algorithm will perform best across every installation.

### Random forest shows a generalization gap

Random forest training R² ranges from approximately **0.989 to 0.991**, while held-out R² ranges from **0.730 to 0.860**. This large difference is consistent with overfitting: the model fits the training observations very closely but does not carry all of that fit into the held-out period. Its small gains on Solar4 and Solar5 should be weighed against this gap.

### Neural-network performance is inconsistent

The neural network reached an R² of **0.843** on Solar2 and **0.827** on Solar0, but fell to **0.685** on Solar3 and **0.243** on Solar5. It does not provide a consistent improvement over linear regression, particularly for Solar5, where the saved result is substantially weaker.

### The leading weather features offer a focused next step

The exploratory analysis and feature-screening work point to surface solar radiation, humidity, and temperature. These variables provide a practical reduced feature set for future tuning and comparison. The existing notebook plots compare predictions from all-feature and three-feature random forests, although the saved notebooks do not report a corresponding reduced-feature metric table.

## Running the notebooks

Open the notebooks in Jupyter or Google Colab and make the repository data files available in the working directory. Run `Data Exploratory.ipynb` to review data preparation and exploratory plots, then use `ALL Models.ipynb` for the main model comparison. `Best model.ipynb` contains the additional feature-screening and model analysis. The notebooks use relative file paths and include saved outputs for the reported results.
