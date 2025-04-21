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
- Fixed formatting issues (e.g., numbers with commas as thousand separators)

### Exploratory Data Analysis
- **Problem Type**: Regression (predicting continuous S&P 500 price)
- **Dataset Size**: 402 observations with 20 features
- **Key Observations**:
  - Most price features show right-skewed distributions
  - Bitcoin displays extremely high volatility
  - Berkshire Hathaway's price is significantly higher than other assets
  - Natural Gas prices are more stable than cryptocurrencies
  - S&P 500 prices are approximately normally distributed (mean ~4200, range ~3500-5000)

### Correlation Analysis
- **Strong Correlations**:
  - Nasdaq 100 and S&P 500: 0.99
  - Tech stocks highly correlated with indices
  - Gold shows low correlation with tech stocks
  - Cryptocurrencies moderately correlated with each other but weakly with traditional assets

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
- **Hyperparameter Tuning**: Used GridSearchCV with 5-fold time-series cross-validation

### Performance Metrics
- **Mean Absolute Error (MAE)**: 28.3 points (~0.6% of index value)
- **R-squared**: 0.98
- **Mean Absolute Percentage Error (MAPE)**: 0.67%

### Example Predictions
- Actual: 4768.37 → Predicted: 4742.15 (Error: 26.22)
- Actual: 4958.61 → Predicted: 4931.84 (Error: 26.77)

## Deployment Considerations

### Production Advice
- Implement monitoring for concept drift
- Retrain model periodically with new data
- Use as one input among many for trading decisions
- Implement circuit breakers for extreme market conditions

### Precautions
- Not suitable for long-term predictions
- Performance may degrade during market crises
- Should not be used without human oversight

## Future Improvements

### Additional Data Sources
- Incorporate fundamental data (interest rates, economic indicators)
- Add sentiment analysis from news/social media
- Include more historical data (10+ years)

### Advanced Modeling
- RNNs/LSTMs for better sequence modeling
- Transformer architectures
- Ensemble approaches combining different model types

### Enhanced Feature Engineering
- Fourier transforms for seasonality
- Wavelet transforms
- Change point detection

## Getting Started
*(Add instructions for setting up and running the project)*

## Dependencies
*(List required libraries and tools)*

## License
*(Add license information)*

## Contributors
*(Add team member information)*
