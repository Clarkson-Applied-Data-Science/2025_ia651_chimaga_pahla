# S&P 500 Time Series Forecasting Project

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Process Summary](#process-summary)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Regularized Linear Regression Models](#regularized-linear-regression-models)
- [Performance Comparison](#performance-comparison)
- [Overfitting Mitigation Strategies](#overfitting-mitigation-strategies)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Appendix: Code](#appendix-code)

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

## Exploratory Data Analysis

### Price Distribution Analysis

Our analysis of asset price distributions revealed interesting patterns:

- **Crude Oil Price**: Bimodal distribution ($60-80 range and $40 range)
- **Gold Price**: Multiple distinct peaks at $1750, $1800, and $1950
- **Nasdaq 100 Price**: Multiple price clusters across $8,000-$18,000 range
- **Apple Price**: Primary peak around $150, secondary near $180
- **Microsoft Price**: Two major price clusters at $240 and $330-340
- **Google Price**: Four distinct peaks at approximately $70, $90, $110, and $140
- **Bitcoin Price**: Complex distribution with concentrations at $10,000, $27,000, and $43,000

### Correlation Analysis
- **Strong Correlations**:
  - Nasdaq 100 and S&P 500: 0.99
  - Tech stocks highly correlated with indices
  - Gold shows low correlation with tech stocks
  - Cryptocurrencies moderately correlated with each other but weakly with traditional assets

## Feature Engineering

- **Lag Features**: 1-day, 3-day, and 7-day lagged versions of key prices
- **Percentage Changes**: Daily returns for all assets
- **Technical Indicators**:
  - Moving averages (7-day and 21-day)
  - Relative Strength Index (RSI)
  - Bollinger Bands
- **Date Features**: Day of week, month, and quarter
- **Interaction Terms**: Cross-features between highly correlated assets

## Model Development

- **Train/Test Split**: Time-based (not random) with first 80% for training, last 20% for testing
- **Models Evaluated**:
  - Linear Regression (baseline)
  - Ridge Regression (new)
  - Lasso Regression (new)
  - Random Forest
  - XGBoost
  - ARIMA

### Data Leakage Considerations

Our project addressed two critical data leakage risks:

1. **Look-Ahead Bias**: Mitigated through chronological train-test split, lag features, and ensuring all technical indicators used only historical data
   
2. **Data Preprocessing Leakage**: Addressed by fitting transformations only on training data, performing feature selection using only training data, and using time-based cross-validation

## Regularized Linear Regression Models

To address overfitting observed in our previous models, we implemented two regularized linear regression approaches:

### Ridge Regression

Ridge regression adds an L2 penalty (sum of squared coefficients) to the linear regression cost function, helping to reduce the impact of multicollinearity and prevent overfitting.

**Implementation Approach**:
- Used `TimeSeriesSplit` with 5 folds for proper time series validation
- Tested alpha values ranging from 0.001 to 1000 (logspace) using grid search
- Selected optimal alpha value that minimized validation error

### Lasso Regression

Lasso regression adds an L1 penalty (sum of absolute coefficients) that can shrink some coefficients to zero, effectively performing feature selection.

**Implementation Approach**:
- Used `TimeSeriesSplit` with 5 folds for proper time series validation
- Tested alpha values ranging from 0.001 to 1000 (logspace) using grid search
- Selected optimal alpha value that minimized validation error
- Examined which features were retained by the model

### Optimal Hyperparameters

- **Ridge optimal alpha**: 10.0
- **Lasso optimal alpha**: 0.1

### Feature Selection Results with Lasso

The Lasso model with optimal alpha selected 35 features out of the original 87 engineered features. The most important retained features were:

1. Nasdaq_100_Price_lag_1
2. SNP500_lag_1
3. Microsoft_Price_lag_1
4. Apple_Price_lag_1
5. Gold_Price_lag_1

This significant reduction in features helps to prevent overfitting by creating a more parsimonious model.

## Performance Comparison

After implementing regularized linear models and additional overfitting mitigation techniques, here are the updated results:

| Model | Train MAE | Test MAE | Train R² | Test R² | Train MAPE | Test MAPE |
|-------|-----------|----------|----------|---------|------------|-----------|
| Linear Regression | 21.42 | 45.27 | 0.997 | 0.899 | 0.57% | 1.05% |
| Ridge Regression | 21.95 | 42.18 | 0.996 | 0.912 | 0.58% | 0.98% |
| Lasso Regression | 22.31 | 43.45 | 0.996 | 0.905 | 0.59% | 1.01% |
| Random Forest | 10.33 | 67.08 | 0.999 | 0.869 | 0.28% | 1.51% |
| XGBoost | 0.02 | 66.49 | 1.000 | 0.869 | 0.00% | 1.48% |
| Tuned XGBoost | 1.72 | 66.35 | 1.000 | 0.843 | 0.05% | 1.46% |
| ARIMA | - | 261.18 | - | -0.717 | - | 5.75% |

### Key Observations

1. **Ridge Regression** emerged as the best-performing model with the lowest test MAE (42.18) and highest test R² (0.912)
2. **Lasso Regression** performed slightly worse than Ridge but better than the baseline Linear Regression
3. **Regularized models** showed reduced gap between training and testing performance, indicating better generalization
4. **Tree-based models** still showed signs of overfitting despite tuning efforts
5. **ARIMA** continued to underperform compared to regression-based approaches

## Overfitting Mitigation Strategies

We implemented several strategies to mitigate overfitting:

1. **Regularization**
   - Ridge regression's L2 penalty helped control coefficient magnitudes
   - Lasso regression's L1 penalty performed feature selection

2. **Time Series Cross-Validation**
   - Used `TimeSeriesSplit` instead of random k-fold cross-validation
   - Respected temporal order of observations
   - Prevented data leakage from future to past

3. **Feature Selection**
   - Lasso's built-in feature selection capability reduced model complexity
   - Removed multicollinear features
   - Selected features based on importance thresholds

4. **Hyperparameter Tuning**
   - Grid search with time series cross-validation
   - Systematically identified optimal regularization strength

5. **Ensemble Methods**
   - Created a simple ensemble by averaging predictions from multiple models
   - Reduced variance in predictions

## Limitations

Despite our efforts to build accurate and robust models, several limitations must be acknowledged:

1. **Market Unpredictability**
   - Financial markets are influenced by unpredictable events (geopolitical tensions, policy changes)
   - Models cannot account for "black swan" events
   - Market regime changes can render historical patterns less relevant

2. **Feature Limitations**
   - Our models rely primarily on price-based features
   - Missing important factors:
     - Trading volume
     - Market sentiment (news, social media)
     - Macroeconomic indicators (interest rates, inflation)
     - Sector-specific developments

3. **Temporal Stability**
   - Model performance tends to degrade over time
   - Market dynamics evolve and relationships between variables change
   - Periodic retraining is necessary but introduces complexity

4. **Overfitting Concerns**
   - Despite mitigation strategies, some models still show signs of overfitting
   - Test performance remains significantly worse than training performance
   - More sophisticated regularization may be needed

5. **Prediction Horizon**
   - Current models focus on next-day predictions
   - Longer-term forecasts would require different approaches
   - Uncertainty compounds with prediction distance

## Future Scope

Several promising directions could enhance this project in the future:

1. **Enhanced Feature Engineering**
   - Incorporate sentiment analysis from financial news and social media
   - Add macroeconomic indicators (GDP, unemployment, interest rates)
   - Include options market data (implied volatility, put-call ratio)
   - Develop non-linear feature transformations

2. **Advanced Modeling Approaches**
   - Implement Bayesian models for uncertainty quantification
   - Explore Gaussian Processes for time series forecasting
   - Develop multi-task learning to predict multiple indices simultaneously
   - Create hybrid models combining statistical and machine learning approaches

3. **Production Implementation**
   - Develop automated data pipeline for daily updates
   - Implement model monitoring for performance degradation
   - Create alert systems for prediction anomalies
   - Build a retraining schedule based on performance metrics

4. **Risk Analysis**
   - Add confidence intervals to predictions
   - Implement Monte Carlo simulations for stress testing
   - Develop risk-adjusted return metrics
   - Create prediction-based trading strategies with proper risk management

5. **Interpretability Enhancements**
   - Develop more sophisticated feature importance analysis
   - Implement SHAP (SHapley Additive exPlanations) values
   - Create scenario analysis tools
   - Build visualization dashboards for model behavior

## Conclusion

This project demonstrates the effectiveness of regularized linear regression models for predicting S&P 500 prices. Ridge Regression emerged as the optimal model, balancing complexity and performance with a test MAE of 42.18 (approximately 0.98% error rate).

The significant gap between the performance of simpler regularized models and more complex tree-based models highlights the importance of focusing on generalization in financial forecasting. The regularization techniques effectively reduced overfitting while maintaining strong predictive accuracy.

Our feature selection results emphasize the importance of recent price history (particularly 1-day lag features) and cross-asset relationships (especially between the S&P 500, Nasdaq 100, and major tech stocks). These findings align with financial market theory about the importance of price momentum and sector correlations.

For practical implementation, we recommend using the Ridge Regression model with daily retraining to maintain forecast accuracy. The model's simplicity makes it easier to interpret, update, and monitor compared to more complex alternatives.

## Acknowledgments

We would like to express our sincere gratitude to Lecturer Michael Gilbert for his guidance and valuable insights throughout this project. His expertise in financial modeling and machine learning significantly contributed to the development of our approach and the interpretation of our results.

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Science & Business Media.

2. Hyndman, R.J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice. OTexts.

3. De Prado, M.L. (2018). Advances in Financial Machine Learning. John Wiley & Sons.

4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

5. Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied Soft Computing, 90, 106181.

6. Brownlee, J. (2018). Introduction to Time Series Forecasting with Python. Machine Learning Mastery.

7. Gilbert, M. (2024). Lecture Notes on Time Series Analysis and Financial Forecasting. University Course Materials.

## Appendix: Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Main function to run the project
def main():
    # Load and preprocess data
    data = load_data("US_Stock_Data.xlsx")
    
    # Engineer features
    processed_data = engineer_features(data)
    
    # Split data (80% train, 20% test)
    train_size = int(len(processed_data) * 0.8)
    train_data = processed_data.iloc[:train_size]
    test_data = processed_data.iloc[train_size:]
    
    # Prepare features and target
    features = [col for col in processed_data.columns if col != 'SNP500']
    X_train = train_data[features]
    y_train = train_data['SNP500']
    X_test = test_data[features]
    y_test = test_data['SNP500']
    
    # Train models and get results
    linear_models, linear_results, _ = train_linear_models(X_train, y_train, X_test, y_test)
    tree_models, tree_results = train_tree_models(X_train, y_train, X_test, y_test)
    arima_model, arima_preds, arima_results = train_arima_model(data, 'SNP500', len(test_data))
    
    # Print final performance comparison
    print_performance_comparison(linear_results, tree_results, arima_results)
