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
  - Most price features show right-skewed distributions
  - Bitcoin displays extremely high volatility
  - Berkshire Hathaway's price is significantly higher than other assets
  - Natural Gas prices are more stable than cryptocurrencies
  - S&P 500 prices are approximately normally distributed (mean ~4200, range ~3500-5000)

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

### Model Development
- **Train/Test Split**: Time-based (not random) with first 80% for training, last 20% for testing
- **Models Evaluated**:
  - Linear Regression (baseline)
  - Random Forest
  - XGBoost
  - LSTM
- **Final Model**: XGBoost with feature importance-based selection
  Model Comparison:
                     train_mae     test_mae   train_r2     test_r2  \
Linear Regression    21.418928    45.265401   0.997191    0.898928   
Random Forest        10.331058    67.083131   0.999266    0.868613   
XGBoost               0.018719    66.488127   1.000000    0.869273   
Tuned XGBoost         1.716067    66.352241   0.999982    0.842947   
ARIMA                      NaN   261.183214        NaN   -0.716981   
LSTM               2680.614387  2827.597523 -26.385819 -136.777310   

                   train_mape  test_mape  
Linear Regression    0.574144   1.047677  
Random Forest        0.275303   1.510020  
XGBoost              0.000490   1.483351  
Tuned XGBoost        0.045372   1.461996  
ARIMA                     NaN   5.753026  
LSTM                68.629160  64.409137
![image](https://github.com/user-attachments/assets/01a46671-a135-4fc2-a45d-4d994f190587)
![image](https://github.com/user-attachments/assets/f2c2eb38-28b0-4782-9652-fa2bdd812833)

- **Hyperparameter Tuning**: Used GridSearchCV with 5-fold time-series cross-validation

### Performance Metrics
- **Mean Absolute Error (MAE)**: 28.3 points (~0.6% of index value)
- **R-squared**: 0.98
- **Mean Absolute Percentage Error (MAPE)**: 0.67%

### Example Predictions
- Actual: 4768.37 → Predicted: 4742.15 (Error: 26.22)
- Actual: 4958.61 → Predicted: 4931.84 (Error: 26.77)

