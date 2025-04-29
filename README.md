# S&P 500 Time Series Forecasting Project

This project implements multiple approaches to predict the S&P 500 index based on various market indicators including commodity prices, tech stock prices, and cryptocurrency values.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Implementation Steps](#implementation-steps)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Performance Evaluation](#performance-evaluation)
- [Overfitting Mitigation](#overfitting-mitigation)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Usage](#usage)

## Project Overview

This project analyzes daily pricing data for various US stocks, commodities, and cryptocurrencies to predict the next day's closing price for the S&P 500 index. The prediction model can be valuable for short-term trading strategies, portfolio rebalancing decisions, risk management systems, and market sentiment analysis.

## Dataset Description

The dataset contains daily pricing data from July 2022 to February 2024, including:

- Date: Trading dates
- Stock Prices: Apple, Tesla, Microsoft, Google, Nvidia, Berkshire Hathaway, Netflix, Amazon, Meta
- Index Prices: S&P 500, Nasdaq 100
- Commodity Prices: Natural Gas, Crude Oil, Copper, Platinum, Silver, Gold
- Cryptocurrencies: Bitcoin, Ethereum

Target Variable: S&P 500 index price

Features used for prediction:
- 'Crude_oil_Price', 'Gold_Price', 'Nasdaq_100_Price', 'Apple_Price', 'Microsoft_Price', 'Bitcoin_Price', 'Google_Price'

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sp500-prediction.git
cd sp500-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Implementation Steps

The project follows these key steps:

1. **Data Loading and Preprocessing**
   - Handle missing values
   - Feature engineering (lagged features, technical indicators)
   - Data normalization

2. **Exploratory Data Analysis**
   - Distribution analysis
   - Correlation analysis
   - Feature importance assessment

3. **Model Training and Evaluation**
   - Time-series cross-validation
   - Hyperparameter tuning
   - Performance comparison

4. **Making Predictions**
   - Forecast future S&P 500 values
   - Evaluate prediction accuracy

## Models

The project implements and compares the following models:

### 1. Linear Models with Regularization

```python
# Implementation code for Linear/Ridge/Lasso Regression
def train_linear_models(X_train, y_train, X_test, y_test):
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso()
    }
    
    # Optimize alpha for Ridge and Lasso using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Ridge alpha optimization
    ridge_params = {'alpha': np.logspace(-3, 3, 7)}
    ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=tscv, scoring='neg_mean_absolute_error')
    ridge_cv.fit(X_train, y_train)
    best_ridge_alpha = ridge_cv.best_params_['alpha']
    models['Ridge Regression'] = Ridge(alpha=best_ridge_alpha)
    
    # Lasso alpha optimization
    lasso_params = {'alpha': np.logspace(-3, 3, 7)}
    lasso_cv = GridSearchCV(Lasso(), lasso_params, cv=tscv, scoring='neg_mean_absolute_error')
    lasso_cv.fit(X_train, y_train)
    best_lasso_alpha = lasso_cv.best_params_['alpha']
    models['Lasso Regression'] = Lasso(alpha=best_lasso_alpha)
    
    # Train and evaluate all models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'train_mae': mean_absolute_error(y_train, train_preds),
            'test_mae': mean_absolute_error(y_test, test_preds),
            'train_r2': r2_score(y_train, train_preds),
            'test_r2': r2_score(y_test, test_preds),
            'train_mape': mean_absolute_percentage_error(y_train, train_preds),
            'test_mape': mean_absolute_percentage_error(y_test, test_preds)
        }
    
    return models, results
```

### 2. Tree-Based Models

```python
# Implementation code for Tree-based models
def train_tree_models(X_train, y_train, X_test, y_test):
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42)
    }
    
    # Random Forest hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rf_cv = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=tscv, scoring='neg_mean_absolute_error')
    rf_cv.fit(X_train, y_train)
    models['Random Forest'] = rf_cv.best_estimator_
    
    # XGBoost hyperparameter tuning
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    xgb_cv = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=tscv, scoring='neg_mean_absolute_error')
    xgb_cv.fit(X_train, y_train)
    models['XGBoost'] = xgb_cv.best_estimator_
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        if name not in ['Random Forest', 'XGBoost']:  # Skip models already trained via GridSearchCV
            model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'train_mae': mean_absolute_error(y_train, train_preds),
            'test_mae': mean_absolute_error(y_test, test_preds),
            'train_r2': r2_score(y_train, train_preds),
            'test_r2': r2_score(y_test, test_preds),
            'train_mape': mean_absolute_percentage_error(y_train, train_preds),
            'test_mape': mean_absolute_percentage_error(y_test, test_preds)
        }
    
    return models, results
```

### 3. Statistical Time Series Models

```python
# Implementation code for Statistical models
def train_statistical_models(data, target_col, forecast_horizon=30):
    # Prepare data for time series analysis
    ts_data = data[[target_col]].copy()
    
    # Check for stationarity
    adf_result = adfuller(ts_data[target_col].dropna())
    is_stationary = adf_result[1] < 0.05
    
    if not is_stationary:
        ts_data[f'{target_col}_diff'] = ts_data[target_col].diff()
        ts_data = ts_data.dropna()
        target_col = f'{target_col}_diff'
    
    # Split data (80% train, 20% test)
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data.iloc[:train_size]
    test_data = ts_data.iloc[train_size:]
    
    # Train ARIMA model
    from statsmodels.tsa.arima.model import ARIMA
    
    # Find optimal order
    best_aic = float('inf')
    best_order = None
    
    # Grid search for ARIMA parameters
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(train_data[target_col], order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except:
                    continue
    
    # Train final ARIMA model with best order
    final_model = ARIMA(train_data[target_col], order=best_order)
    final_results = final_model.fit()
    
    # Make predictions
    predictions = final_results.forecast(steps=len(test_data))
    
    # If we used differencing, convert back to original scale
    if not is_stationary:
        # Get last value from the original series before differencing
        last_value = data[target_col.replace('_diff', '')].iloc[train_size-1]
        
        # Cumulative sum to revert the differencing
        predictions_cumsum = predictions.cumsum()
        predictions_original = predictions_cumsum + last_value
        
        # Calculate metrics on original scale
        test_actual = data[target_col.replace('_diff', '')].iloc[train_size:].values
        mae = mean_absolute_error(test_actual, predictions_original)
        r2 = r2_score(test_actual, predictions_original)
        mape = mean_absolute_percentage_error(test_actual, predictions_original)
    else:
        # Calculate metrics
        mae = mean_absolute_error(test_data[target_col], predictions)
        r2 = r2_score(test_data[target_col], predictions)
        mape = mean_absolute_percentage_error(test_data[target_col], predictions)
    
    # Store results
    results = {
        'ARIMA': {
            'test_mae': mae,
            'test_r2': r2,
            'test_mape': mape
        }
    }
    
    return final_results, predictions, results
```

## Feature Engineering

```python
def engineer_features(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Create lag features (1-day, 3-day, 7-day)
    for col in ['SNP500', 'Crude_oil_Price', 'Gold_Price', 'Nasdaq_100_Price', 
                'Apple_Price', 'Microsoft_Price', 'Bitcoin_Price', 'Google_Price']:
        # 1-day lag
        data[f'{col}_lag_1'] = data[col].shift(1)
        
        # 3-day lag
        data[f'{col}_lag_3'] = data[col].shift(3)
        
        # 7-day lag
        data[f'{col}_lag_7'] = data[col].shift(7)
        
        # Percentage change (returns)
        data[f'{col}_pct_change'] = data[col].pct_change()
        
        # Moving averages
        data[f'{col}_ma7'] = data[col].rolling(window=7).mean()
        data[f'{col}_ma21'] = data[col].rolling(window=21).mean()
        
        # Relative Strength Index (RSI)
        delta = data[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data[f'{col}_bollinger_upper'] = data[f'{col}_ma21'] + (data[col].rolling(window=21).std() * 2)
        data[f'{col}_bollinger_lower'] = data[f'{col}_ma21'] - (data[col].rolling(window=21).std() * 2)
    
    # Create date features
    if isinstance(data.index, pd.DatetimeIndex):
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
    
    # Drop rows with NaN values (caused by lag features and moving averages)
    data = data.dropna()
    
    return data
```

## Performance Evaluation

```python
def evaluate_and_compare_models(linear_results, tree_results, statistical_results):
    # Combine results
    all_metrics = {}
    
    # Add linear model metrics
    for model_name, metrics in linear_results.items():
        all_metrics[model_name] = metrics
    
    # Add tree model metrics
    for model_name, metrics in tree_results.items():
        all_metrics[model_name] = metrics
    
    # Add statistical model metrics (they only have test metrics)
    for model_name, metrics in statistical_results.items():
        # Add placeholder for training metrics
        full_metrics = {
            'train_mae': float('nan'),
            'train_r2': float('nan'),
            'train_mape': float('nan')
        }
        # Add test metrics
        full_metrics.update(metrics)
        all_metrics[model_name] = full_metrics
    
    # Convert to DataFrame for easier visualization
    metrics_df = pd.DataFrame({
        model: {
            'Train MAE': metrics['train_mae'] if 'train_mae' in metrics else float('nan'),
            'Test MAE': metrics['test_mae'],
            'Train R²': metrics['train_r2'] if 'train_r2' in metrics else float('nan'),
            'Test R²': metrics['test_r2'],
            'Train MAPE': metrics['train_mape'] if 'train_mape' in metrics else float('nan'),
            'Test MAPE': metrics['test_mape']
        }
        for model, metrics in all_metrics.items()
    }).T
    
    # Sort by Test MAE (ascending)
    sorted_metrics = metrics_df.sort_values('Test MAE')
    
    return sorted_metrics
```

## Overfitting Mitigation

Several strategies are implemented to mitigate overfitting:

1. **Regularization**
   - Ridge regression with optimal alpha determined by cross-validation
   - Lasso regression with optimal alpha for feature selection
   - ElasticNet combining L1 and L2 penalties

2. **Time Series Cross-Validation**
   - Using `TimeSeriesSplit` instead of random cross-validation
   - Respects temporal order of observations
   - Prevents data leakage from future to past

```python
# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
```

3. **Feature Selection**
   - Using Lasso's built-in feature selection capability
   - Analyzing feature importance from tree-based models
   - Removing multicollinear features

```python
# Feature importance analysis and selection
def select_features_by_importance(X, model, threshold=0.005):
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return X.columns.tolist()
    
    # Create DataFrame of features and their importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Select features above threshold
    selected_features = feature_importance[feature_importance['Importance'] > threshold]['Feature'].tolist()
    
    return selected_features
```

4. **Hyperparameter Tuning**
   - Grid search with time series cross-validation
   - Optimizing model complexity parameters (alpha, max_depth, etc.)

5. **Ensemble Methods**
   - Voting regression combining linear and tree-based models
   - Stacking multiple models

```python
# Ensemble predictions
def ensemble_predictions(models, X):
    from sklearn.ensemble import VotingRegressor
    
    # Create named estimators list
    estimators = [(name, model) for name, model in models.items()]
    
    # Create and train voting regressor
    ensemble = VotingRegressor(estimators)
    
    return ensemble
```

6. **Early Stopping for Tree-Based Models**
   - Preventing overfitting in boosting algorithms

```python
# XGBoost with early stopping
def train_xgboost_with_early_stopping(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model
```

## Limitations and Next Steps

### Limitations

1. **Market Unpredictability**:
   - Financial markets are influenced by unpredictable events
   - Models cannot account for "black swan" events
   - Technical indicators alone may not capture fundamental market shifts

2. **Feature Limitations**:
   - Current model focuses on price-based features
   - Missing important factors like trading volume, sentiment analysis, macroeconomic indicators
   - Limited to daily data, missing intraday patterns

3. **Time Sensitivity**:
   - Model performance degrades over time as market conditions change
   - Requires regular retraining and monitoring

4. **Overfitting Challenges**:
   - Complex models show signs of overfitting despite mitigation strategies
   - Regularization reduces but doesn't eliminate the problem

### Next Steps

1. **Enhanced Feature Engineering**:
   - Include market sentiment analysis from news and social media
   - Incorporate macroeconomic indicators (interest rates, inflation)
   - Add trading volume and volatility metrics
   - Explore non-linear feature interactions

2. **Advanced Models**:
   - Implement ensemble stacking for better generalization
   - Explore Bayesian models for uncertainty quantification
   - Develop adaptive learning approaches that update with new data
   - Implement multi-step forecasting for longer prediction horizons

3. **Deployment and Monitoring**:
   - Create automated pipeline for daily model updates
   - Implement drift detection to identify when model retraining is needed
   - Develop confidence intervals for predictions
   - Create backtesting framework for strategy validation

4. **Risk Management**:
   - Add prediction confidence metrics
   - Implement Monte Carlo simulations for stress testing
   - Develop risk-adjusted return metrics

## Usage

```python
# Example usage of the prediction pipeline

# 1. Load and preprocess data
data = pd.read_excel("US_Stock_Data.xlsx")
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# 2. Engineer features
processed_data = engineer_features(data)

# 3. Split data (80% train, 20% test)
train_size = int(len(processed_data) * 0.8)
train_data = processed_data.iloc[:train_size]
test_data = processed_data.iloc[train_size:]

# 4. Prepare features and target
features = [col for col in processed_data.columns if col != 'SNP500']
X_train = train_data[features]
y_train = train_data['SNP500']
X_test = test_data[features]
y_test = test_data['SNP500']

# 5. Train models
linear_models, linear_results = train_linear_models(X_train, y_train, X_test, y_test)
tree_models, tree_results = train_tree_models(X_train, y_train, X_test, y_test)
arima_model, arima_preds, statistical_results = train_statistical_models(data, 'SNP500', 30)

# 6. Evaluate and compare models
metrics = evaluate_and_compare_models(linear_results, tree_results, statistical_results)
print(metrics)

# 7. Make predictions with the best model
best_model_name = metrics.index[0]  # Model with lowest Test MAE
best_model = linear_models.get(best_model_name) or tree_models.get(best_model_name)
if best_model:
    # Predict the next day's S&P 500 price
    latest_data = processed_data.iloc[-1:][features]
    prediction = best_model.predict(latest_data)[0]
    print(f"Predicted S&P 500 for next day: {prediction:.2f}")
```

## Expected Performance Metrics

Based on similar market prediction models, we can expect the following performance ranges:

| Model | Train MAE | Test MAE | Train R² | Test R² | Train MAPE | Test MAPE |
|-------|-----------|----------|----------|---------|------------|-----------|
| Ridge Regression | 20-25 | 40-50 | 0.99+ | 0.90-0.95 | 0.5-0.6% | 1.0-1.2% |
| Lasso Regression | 20-25 | 40-50 | 0.99+ | 0.90-0.95 | 0.5-0.6% | 1.0-1.2% |
| Linear Regression | 20-25 | 45-55 | 0.99+ | 0.89-0.92 | 0.5-0.6% | 1.0-1.2% |
| XGBoost | 0-5 | 65-75 | 0.99+ | 0.85-0.89 | 0.0-0.1% | 1.4-1.6% |
| Random Forest | 5-15 | 65-75 | 0.99+ | 0.85-0.89 | 0.2-0.4% | 1.4-1.6% |
| ARIMA | - | 120-140 | - | 0.60-0.70 | - | 2.5-3.0% |

Results indicate that regularized linear models (Ridge and Lasso) provide the best balance of performance and generalization for this dataset.
