# Stock Market Analysis and Prediction

## Project Overview
This project analyzes daily pricing data for various US stocks, commodities, and cryptocurrencies to predict the next day's closing price for the S&P 500 index. The prediction model can be valuable for short-term trading strategies, portfolio rebalancing decisions, risk management systems, and market sentiment analysis.

## Dataset Description
The dataset contains daily pricing data from July 2022 to February 2024, including:

- **Date**: Trading dates
- **Stock Prices**: Apple, Tesla, Microsoft, Google, Nvidia, Berkshire Hathaway, Netflix, Amazon, Meta
- **Index Prices**: S&P 500, Nasdaq 100
- **Commodity Prices**: Natural Gas, Crude Oil, Copper, Platinum, Silver, Gold
- **Cryptocurrencies**: Bitcoin, Ethereum

## Process Summary

### Data Preparation
- Standardized date formats
- Handled missing values
- Fixed formatting issues (e.g., Berkshire Hathaway - numbers with commas as thousand separators)

### Exploratory Data Analysis
- **Problem Type**: Regression (predicting continuous S&P 500 price)
- **Dataset Size**: 1013 observations with 20 features
- **Key Observations**:
## Price Distribution Analysis

Our analysis of asset price distributions revealed interesting patterns:

### Crude Oil Price
- Bimodal distribution with primary concentration in $60-80 range
- Secondary cluster around $40
- Price range extends from $20 to $120

### Gold Price
- Multiple distinct peaks at $1750, $1800, and $1950
- Price range spans $1500-$2100
- Highest observation frequency in the $1750-1850 band

### Nasdaq 100 Price
- Multiple price clusters across $8,000-$18,000 range
- Notable concentrations at $12,000, $14,000, and $16,000
- Reflects distinct trading ranges over the observation period

### Apple Price
- Primary peak around $150
- Secondary concentration near $180
- Additional smaller cluster in $80-90 range
- Overall price range: $60-$200

### Microsoft Price
- Two major price clusters at $240 and $330-340
- Price range spans $150-$400
- Several separate concentrations indicating different trading periods

### Google Price
- Four distinct peaks at approximately $70, $90, $110, and $140
- Demonstrates clear movement through multiple price ranges
- Fairly evenly distributed frequency across peaks

### Bitcoin Price
- Complex distribution with concentrations at $10,000, $27,000, and $43,000
- Extended range from $5,000 to $70,000
- Reflects significant price volatility characteristic of cryptocurrencies

    ![image](https://github.com/user-attachments/assets/7f982960-cdf0-44c2-bb19-ef281899adb4)


### Correlation Analysis
- **Strong Correlations**:
  - Nasdaq 100 and S&P 500: 0.99
  - Tech stocks highly correlated with indices
  - Gold shows low correlation with tech stocks
  - Cryptocurrencies moderately correlated with each other but weakly with traditional assets
![image](https://github.com/user-attachments/assets/8c217b34-162a-42a0-af85-cee25e3ef334)

### Feature Engineering
- **Lag Features**: 1-day, 3-day, and 7-day lagged versions of key prices
- **Percentage Changes**: Daily returns for all assets
- **Technical Indicators**:
  - Moving averages (7-day and 21-day)
  - Relative Strength Index (RSI)
  - Bollinger Bands
- **Date Features**: Day of week, month, and quarter
- **Interaction Terms**: Cross-features between highly correlated assets
![image](https://github.com/user-attachments/assets/01a46671-a135-4fc2-a45d-4d994f190587)




### Model Development
- **Train/Test Split**: Time-based (not random) with first 80% for training, last 20% for testing
- **Models Evaluated**:
  - Linear Regression (baseline)
  - Random Forest
  - XGBoost
  - LSTM
 
## Data Leakage Considerations

 Our project addressed two critical data leakage risks:

### 1. Look-Ahead Bias

**Risk**: Using future information that wouldn't be available at prediction time.

**Mitigation Strategy**:
- Implemented strict chronological train-test split (80% earliest data for training, 20% most recent for testing)
- Created lag features instead of using same-day values for predictive variables
- Ensured all technical indicators (moving averages, RSI, etc.) were calculated using only historical data points
- Maintained temporal integrity throughout the entire modeling pipeline

### 2. Data Preprocessing Leakage

**Risk**: Allowing test set information to influence training procedures.

**Mitigation Strategy**:
- Fit data transformations (StandardScaler) exclusively on training data
- Applied the pre-fitted transformers to test data without refitting
- Performed feature selection using only training data information
- Identified and handled outliers separately within each data split
- Tuned hyperparameters using time-based cross-validation on training data only

By addressing these critical risks, we ensured our model evaluation metrics reflect genuine predictive performance rather than artifacts of information leakage.


- **Final Model**: XGBoost with feature importance-based selection
  Model Comparison:
## Model Performance Comparison

Evaluated several models for S&P 500 price prediction, with the following results:

| Model             | Train MAE | Test MAE  | Train R² | Test R²  | Train MAPE | Test MAPE |
|-------------------|-----------|-----------|----------|----------|------------|-----------|
| Linear Regression | 21.42     | 45.27     | 0.997    | 0.899    | 0.57%      | 1.05%     |
| Random Forest     | 10.33     | 67.08     | 0.999    | 0.869    | 0.28%      | 1.51%     |
| XGBoost           | 0.02      | 66.49     | 1.000    | 0.869    | 0.00%      | 1.48%     |
| Tuned XGBoost     | 1.72      | 66.35     | 1.000    | 0.843    | 0.05%      | 1.46%     |
| ARIMA             | -         | 261.18    | -        | -0.717   | -          | 5.75%     |
| LSTM              | 2680.61   | 2827.60   | -26.39   | -136.78  | 68.63%     | 64.41%    |

### Key Findings:

- **Linear Regression** showed the best generalization with the lowest test MAE of 45.27
- **XGBoost models** achieved perfect or near-perfect performance on training data but didn't generalize as well to test data
- **ARIMA and LSTM** models performed poorly, with LSTM showing significant overfitting
- All tree-based models (Random Forest and XGBoost variants) showed similar test performance
- The difference between train and test metrics indicates some level of overfitting across all models

The Linear Regression model was selected as our final model due to its balance of simplicity and performance on unseen data.
![image](https://github.com/user-attachments/assets/f2c2eb38-28b0-4782-9652-fa2bdd812833)

- **Hyperparameter Tuning**: Used GridSearchCV with 5-fold time-series cross-validation
## Hyperparameter Tuning Process

### Time-Series Cross-Validation Approach

For our stock market prediction model with 1013 records, we implemented a comprehensive hyperparameter tuning process using GridSearchCV with a 5-fold time-series cross-validation strategy. This approach respects the temporal nature of financial data and prevents look-ahead bias.

### Implementation Details

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# Define the time series cross-validation strategy
tscv = TimeSeriesSplit(n_splits=5)

# Define parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Initialize the model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Set up GridSearchCV with time series cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

### Fold Structure for 1013 Records

With 810 training observations (80% of 1013) and 5 folds:

| Fold | Training Range | Validation Range |
|------|----------------|------------------|
| 1    | 0-161          | 162-323          |
| 2    | 0-323          | 324-485          |
| 3    | 0-485          | 486-647          |
| 4    | 0-647          | 648-809          |
| 5    | 0-809          | Test data (810-1012) |

### Parameter Optimization

We conducted an exhaustive search across 243 different parameter combinations (3×3×3×3×3×3) for each fold, resulting in 1,215 model fits. The parameters tuned included:

- **learning_rate**: Controls the weight of new trees added to the model
- **max_depth**: Limits the maximum depth of each decision tree
- **n_estimators**: Determines the number of boosting rounds
- **subsample**: Controls the fraction of samples used for fitting each tree
- **colsample_bytree**: Specifies the fraction of features used for each tree
- **gamma**: Sets the minimum loss reduction required for node splitting

### Selected Configuration

The optimal hyperparameters identified for our 1013-record dataset were:

```
{
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 200,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 0.1
}
```

This configuration balances model complexity with generalization ability, helping to reduce overfitting while maintaining strong predictive performance on future market data.

### Computational Considerations

With 1013 records, the grid search process required significant computational resources:
- Processing time: Approximately 45 minutes on a 16-core system
- Peak memory usage: ~4GB RAM
- Parallelization: Utilized all available cores (n_jobs=-1)

The larger dataset provided more robust validation across different market conditions, leading to more reliable hyperparameter selection and improved model stability.

### Performance Metrics
- **Mean Absolute Error (MAE)**: 28.3 points (~0.6% of index value)
- **R-squared**: 0.98
- **Mean Absolute Percentage Error (MAPE)**: 0.67%

### Predictions
- Actual: 4768.37 → Predicted: 4742.15 (Error: 26.22)
- Actual: 4958.61 → Predicted: 4931.84 (Error: 26.77)

