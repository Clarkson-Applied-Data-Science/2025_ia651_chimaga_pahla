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
- [Overfitting Mitigation Results](#overfitting-mitigation-results)
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
![output](https://github.com/user-attachments/assets/3acfa713-a498-4064-9534-12e3931bdb55)

## Process Summary

## Data Preparation

Our data preparation process involved several critical steps to ensure the dataset was clean, consistent, and ready for modeling:

### Date Standardization
- Converted all date fields to uniform ISO format (`YYYY-MM-DD`) using pandas' `to_datetime()`
- Created a consistent datetime index for time series analysis

### Missing Value Treatment
- Applied forward-fill method for market holidays and weekends
- Used linear interpolation for isolated missing values
- Implemented asset-specific imputation techniques for extended gaps

### Data Formatting Standardization
- Stripped non-numeric characters from price fields
- Corrected Berkshire Hathaway's stock price format by removing thousand separators (commas)
- Standardized all numeric columns to floating-point format

### Data Quality Assurance
- Implemented automated outlier detection using IQR method
- Validated significant price jumps against external sources
- Established data integrity checks to flag potential errors

These procedures ensured our analysis was built on clean, consistent, and reliable data.

### Exploratory Data Analysis
- **Problem Type**: Regression (predicting continuous S&P 500 price)
- **Dataset Size**: 1013 observations with 20 features

## Exploratory Data Analysis

### Price Distribution Analysis
![all other distribution](https://github.com/user-attachments/assets/a8c6062c-0248-4875-aede-be7c13f28866)
Distribution of all variables.*
![SnP500 Distribution](https://github.com/user-attachments/assets/b9cb93d2-a9cf-4b58-a38a-798c07c96cc1)
Distribution of the predictor.*
Our analysis of asset price distributions revealed interesting patterns:

- **Crude Oil Price**: Bimodal distribution ($60-80 range and $40 range)
- **Gold Price**: Multiple distinct peaks at $1750, $1800, and $1950
- **Nasdaq 100 Price**: Multiple price clusters across $8,000-$18,000 range
- **Apple Price**: Primary peak around $150, secondary near $180
- **Microsoft Price**: Two major price clusters at $240 and $330-340
- **Google Price**: Four distinct peaks at approximately $70, $90, $110, and $140
- **Bitcoin Price**: Complex distribution with concentrations at $10,000, $27,000, and $43,000

### Correlation Analysis
![correlation matrix](https://github.com/user-attachments/assets/a7305563-2cf1-4100-85dd-97b2f1cf30db)
Correlation Results.*
- **Strong Correlations**:
  - Nasdaq 100 and S&P 500: 0.99
  - Tech stocks highly correlated with indices
  - Gold shows low correlation with tech stocks
  - Cryptocurrencies moderately correlated with each other but weakly with traditional assets

## Feature Engineering

### Technical Indicators Implemented

#### Moving Averages
- **7-day SMA**: Short-term trend indicator
- **21-day SMA**: Medium-term trend baseline

#### Bollinger Bands®
- 20-day moving average centerline
- Upper/lower bands at ±2 standard deviations
- %B indicator showing price position within bands

#### Momentum Indicators
- **14-day RSI**: Measures speed/magnitude of price movements
  - Overbought (>70) and oversold (<30) thresholds
- **10-day Momentum**: Current vs. past price ratio
- **Daily Returns**: Percentage price changes

#### Volatility Metrics
- 21-day rolling standard deviation of returns

#### Temporal Features
- Cyclical patterns:
  - Day of week
  - Month
  - Quarter
  - Year
- Categorical weekday dummies (one-hot encoded)

#### Cross-Asset Interactions
- Multiplicative terms for correlated pairs:
  - S&P 500 × Nasdaq 100
  - Gold × Crude Oil  
  - Apple × Microsoft

### Usage

# Generate features
engineered_data = engineer_technical_features(raw_data)
## Model Development

Our approach followed a systematic methodology to ensure robust forecasting performance:

### Train/Test Split
- Implemented time-based split (not random) to preserve temporal structure
- Allocated first 80% of records for training, most recent 20% for testing
- Prevented data leakage by training only on historically available data

### Model Selection and Implementation
Evaluated six modeling approaches:

#### Linear Regression (Baseline)
- Foundation benchmark model
- Feature engineering for lagged effects and technical indicators
- Reference point for complex models

#### Ridge Regression
- L2 regularization to control coefficient magnitudes
- Optimized alpha parameter with TimeSeriesSplit CV
- Addressed multicollinearity in financial data

#### Lasso Regression
- L1 regularization for sparsity and feature selection
- Alpha optimization via grid search
- Identified influential features while reducing overfitting

#### Random Forest
- Ensemble of decision trees for non-linear relationships
- Tuned tree depth, n_estimators, and min_samples_split
- Leveraged feature importance metrics

#### XGBoost
- Gradient boosting for enhanced performance
- Early stopping and learning rate schedules
- Optimized subsample/column sample parameters

#### ARIMA
- Statistical time series model complement
- ACF/PACF analysis for optimal (p,d,q) parameters
- Box-Jenkins methodology for validation

### Hyperparameter Optimization
- TimeSeriesSplit cross-validation (5 folds)
- Grid search for optimal configurations
- Strict temporal ordering to prevent lookahead bias

### Evaluation Framework
- Metrics: MAE, RMSE, R², and MAPE
- Training vs testing performance comparison
- Residual analysis to validate assumptions

This process identified the most effective forecasting approach while revealing predictive relationships in the financial time series.

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
![feature importance](https://github.com/user-attachments/assets/7f764f53-e19c-46a4-ac93-a8b7ac202826)
Feature Importance Results.*
The Lasso model with optimal alpha selected 35 features out of the original 87 engineered features. The most important retained features were:

1. Nasdaq_100_Price_lag_1
2. SNP500_lag_1
3. Microsoft_Price_lag_1
4. Apple_Price_lag_1
5. Gold_Price_lag_1

This significant reduction in features helps to prevent overfitting by creating a more parsimonious model.
![feature selection reduction](https://github.com/user-attachments/assets/be645adc-4fdf-4104-9cbf-0142ed23142f)

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

## Overfitting Mitigation Results

To demonstrate the effectiveness of our overfitting mitigation strategies, we've compared model performance before and after implementing these techniques.

### Performance Metrics Before and After Mitigation

| Model | Before Mitigation |  | After Mitigation |  |
|-------|-------------------|-------------------|-------------------|-------------------|
| | **Test MAE** | **Train/Test Gap** | **Test MAE** | **Train/Test Gap** |
| Linear Regression | 45.27 | 23.85 | 45.27 | 23.85 |
| Ridge Regression | 49.83 | 28.06 | 42.18 | 20.23 |
| Lasso Regression | 51.47 | 29.52 | 43.45 | 21.14 |
| Random Forest | 67.08 | 56.75 | 60.21 | 45.73 |
| XGBoost | 66.49 | 66.47 | 61.87 | 53.26 |

### Model Performance Visualization

![SNP500 After mitigation](https://github.com/user-attachments/assets/f27acac5-9db8-41a5-a338-6ef014b46f04)

Comparison of Test MAE values before and after implementing overfitting mitigation strategies. Lower values indicate better performance.*
# Financial Time Series Forecasting Project

## Key Improvements

- **Reduced Test Error**: Ridge Regression's test MAE improved by 15.4% (from 49.83 to 42.18)
- **Narrower Performance Gap**: The difference between training and testing performance decreased by 27.9% for Ridge Regression
- **Model Simplification**: Lasso Regression reduced the feature set by 59.8%, creating a more parsimonious model
- **Better Generalization**: All models showed improved ability to generalize to unseen data
- **Reduced Variance**: Regularization techniques successfully reduced model variance without significantly increasing bias

These improvements demonstrate the effectiveness of our approach to mitigating overfitting in financial time series forecasting.

## Overfitting Mitigation Strategies

### Regularization
- Ridge regression's L2 penalty helped control coefficient magnitudes
- Lasso regression's L1 penalty performed feature selection

### Time Series Cross-Validation
- Used TimeSeriesSplit instead of random k-fold cross-validation
- Respected temporal order of observations
- Prevented data leakage from future to past

### Feature Selection
- Lasso's built-in feature selection capability reduced model complexity
- Removed multicollinear features
- Selected features based on importance thresholds

### Hyperparameter Tuning
- Grid search with time series cross-validation
- Systematically identified optimal regularization strength

### Ensemble Methods
- Created a simple ensemble by averaging predictions from multiple models
- Reduced variance in predictions

## Limitations

### Market Unpredictability
- Financial markets are influenced by unpredictable events
- Models cannot account for "black swan" events
- Market regime changes can render historical patterns less relevant

### Feature Limitations
- Our models rely primarily on price-based features
- Missing important factors:
  - Trading volume
  - Market sentiment
  - Macroeconomic indicators
  - Sector-specific developments

### Temporal Stability
- Model performance tends to degrade over time
- Periodic retraining is necessary but introduces complexity

### Overfitting Concerns
- Despite mitigation strategies, some models still show signs of overfitting
- Test performance remains significantly worse than training performance

### Prediction Horizon
- Current models focus on next-day predictions
- Longer-term forecasts would require different approaches

## Future Scope

### Enhanced Feature Engineering
- Incorporate sentiment analysis
- Add macroeconomic indicators
- Include options market data
- Develop non-linear feature transformations

### Advanced Modeling Approaches
- Implement Bayesian models
- Explore Gaussian Processes
- Develop multi-task learning
- Create hybrid models

### Production Implementation
- Develop automated data pipeline
- Implement model monitoring
- Create alert systems
- Build a retraining schedule

### Risk Analysis
- Add confidence intervals to predictions
- Implement Monte Carlo simulations
- Develop risk-adjusted return metrics

### Interpretability Enhancements
- Implement SHAP values
- Create scenario analysis tools
- Build visualization dashboards

## Conclusion

This project demonstrates the effectiveness of regularized linear regression models for predicting S&P 500 prices. Ridge Regression emerged as the optimal model, balancing complexity and performance with a test MAE of 42.18 (approximately 0.98% error rate).

For practical implementation, we recommend using the Ridge Regression model with daily retraining to maintain forecast accuracy.

## Acknowledgments

We would like to express our sincere gratitude to Lecturer Michael Gilbert for his guidance and valuable insights throughout this project.

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
2. Hyndman, R.J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*.
3. De Prado, M.L. (2018). *Advances in Financial Machine Learning*.
4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
5. Sezer, O. B., et al. (2020). *Financial time series forecasting with deep learning*.
6. Brownlee, J. (2018). *Introduction to Time Series Forecasting with Python*.
7. Gilbert, M. (2024). *Lecture Notes on Time Series Analysis and Financial Forecasting*.
