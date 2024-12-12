import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import logging
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    r2_score, 
    f1_score
)

# Imputation method using Random Forest
def label_encoder(df, column):
    le = LabelEncoder()
    non_null_values = df[column].dropna()
    # Fit and transform only the non-null values
    df.loc[non_null_values.index, column] = le.fit_transform(non_null_values)
    return df

def impute_missing_values(df, target_column, feature_columns, model_type="classification"):
    """
    Fills missing values in the target_column based on related feature_columns.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - target_column (str): The column with missing values to fill.
    - feature_columns (list): List of columns to use as predictors for filling missing values.
    - model_type (str): Type of model to use, "classification" for discrete values (e.g., month)
                        or "regression" for continuous values (e.g., years).

    Returns:
    - DataFrame with missing values in target_column filled.
    """
    # Separate data into rows with and without missing target_column values
    known_data = df.dropna(subset=[target_column])
    unknown_data = df[df[target_column].isnull()]

    if known_data.empty or unknown_data.empty:
        print(f"No missing values in {target_column} or no non-missing data for training.")
        return df

    # Set up predictors and target variable
    X = known_data[feature_columns]
    y = known_data[target_column]

    # Train/test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model based on type
    if model_type == "classification":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "regression":
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Optional: Model Evaluation
    if model_type == "classification":
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Model f1 score for {target_column} imputation: {f1:.2f}")
    else:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squr = r2_score(y_test,y_pred)
        print(f"R2 for {target_column} imputation: {r_squr:.2f}")

    # Impute missing values in the target column
    df.loc[unknown_data.index, target_column] = model.predict(unknown_data[feature_columns])

    return df

def impute_values(df, target, feature_columns, isna=False):
    # Copy 1: For encoding and ML processing
    df_encoded = df.copy()
    # Copy 2: For storing imputed values
    df_imputed = df.copy()

    if isna:
        df_encoded = df_encoded.dropna(subset=['pizza_name_id'])
    for col in feature_columns:
        df_encoded = label_encoder(df_encoded, col)

    df_encoded = impute_missing_values( df_encoded, target_column=target, feature_columns=feature_columns,
        model_type='classification')
    df_imputed[target] = df_imputed[target].combine_first(df_encoded[target])
    df = df_imputed

    # Deleting the encoded and imputed copies
    del df_encoded, df_imputed

    return df


# Method for future sales prediction using best_models dict
def generate_future_sales_predictions(best_models, historical_data, prediction_period=7):
    """
    Generate future sales predictions with robust feature handling
    
    Parameters:
    - best_models: Dictionary of best models for each pizza type
    - historical_data: Full historical dataset used for training
    - prediction_period: Number of days to forecast (default 7)
    
    Returns:
    - DataFrame with future sales predictions
    """
    import pandas as pd
    import numpy as np
    from prophet import Prophet
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from xgboost import XGBRegressor
    
    # Prepare future dates
    last_date = historical_data['order_date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=prediction_period
    )
    
    # Initialize prediction results
    future_predictions = []
    failed_pizzas = []
    
    def create_standard_features(dates):
        """
        Create standard time-based features
        """
        features = pd.DataFrame(index=dates)
        features['year'] = dates.year
        features['month'] = dates.month
        features['day'] = dates.day
        features['dayofweek'] = dates.dayofweek
        features['quarter'] = dates.quarter
        return features
    
    def get_lagged_features(historical_data, pizza_name, future_features):
        """
        Get lagged features for a specific pizza type
        """
        # Find all lagged feature columns for this pizza
        lagged_cols = [
            col for col in historical_data.columns 
            if col.startswith(f'{pizza_name}_lag_')
        ]
        
        # If no lagged features, return the existing features
        if not lagged_cols:
            return future_features
        
        # Use the last known values for lagged features
        last_data_point = historical_data.iloc[-1]
        for col in lagged_cols:
            future_features[col] = last_data_point[col]
        
        return future_features
    
    # Iterate through each pizza type with its best model
    for pizza_name, model_info in best_models.items():
        # Prepare historical data for the specific pizza type
        pizza_history = historical_data[['order_date', pizza_name]].copy()
        pizza_history['order_date'] = pd.to_datetime(pizza_history['order_date'])
        
        # Predict based on model type
        if model_info['model_type'] == 'prophet':
            # Prophet prediction (unchanged from previous version)
            prophet_data = pizza_history.rename(
                columns={'order_date': 'ds', pizza_name: 'y'}
            )
            
            model = Prophet(
                changepoint_prior_scale=model_info['config']['changepoint_prior_scale'],
                seasonality_prior_scale=model_info['config']['seasonality_prior_scale'],
                seasonality_mode='additive',
                weekly_seasonality=True,
                daily_seasonality=False,
                yearly_seasonality=True
            )
            model.fit(prophet_data)
            
            future = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future)
            
            daily_predictions = forecast[['ds', 'yhat']].copy()
            daily_predictions['yhat'] = np.maximum(0, np.round(daily_predictions['yhat']))
            daily_predictions.columns = ['order_date', 'quantity']
            daily_predictions['pizza_name'] = pizza_name
        
        elif model_info['model_type'] == 'sarima':
            # SARIMA prediction (unchanged from previous version)
            pizza_series = pizza_history.set_index('order_date')[pizza_name]
            pizza_series = pizza_series.resample('D').sum().fillna(0)
            
            model = SARIMAX(
                pizza_series,
                order=model_info['config']['order'],
                seasonal_order=model_info['config']['seasonal_order'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit()
            
            forecast = results.get_forecast(steps=prediction_period)
            forecast_mean = pd.Series(
                np.maximum(0, np.round(forecast.predicted_mean.values)),
                index=future_dates
            )
            
            daily_predictions = pd.DataFrame({
                'order_date': future_dates,
                'quantity': forecast_mean.values,
                'pizza_name': pizza_name
            })
        
        elif model_info['model_type'] == 'xgboost':
            # Robust XGBoost prediction with feature handling
            # Prepare feature columns
            feature_columns = [col for col in historical_data.columns 
                                if col not in ['order_date', pizza_name]]
            
            # Create standard features
            future_features = create_standard_features(future_dates)
            
            # Add lagged features
            future_features = get_lagged_features(
                historical_data, 
                pizza_name, 
                future_features
            )
            
            # Ensure all original training features are present
            for col in feature_columns:
                if col not in future_features.columns:
                    # Fill missing columns with last known value or 0
                    last_value = historical_data[col].iloc[-1] if len(historical_data[col]) > 0 else 0
                    future_features[col] = last_value
            
            # Ensure feature order matches training data
            future_features = future_features[feature_columns]
            
            # Predict
            predictions = model_info['model'].predict(future_features)
            
            daily_predictions = pd.DataFrame({
                'order_date': future_dates,
                'quantity': np.maximum(0, np.round(predictions)),
                'pizza_name': pizza_name
            })
        
        # Collect predictions
        future_predictions.append(daily_predictions)
    
    # Combine predictions
    if future_predictions:
        final_predictions = pd.concat(future_predictions, ignore_index=True)
        
        # Sort and reset index
        final_predictions = final_predictions.sort_values(['order_date', 'pizza_name']).reset_index(drop=True)
        
        # Print failed pizzas for debugging
        if failed_pizzas:
            print("\nFailed Pizzas:")
            for pizza, error in failed_pizzas:
                print(f"{pizza}: {error}")
        
        return final_predictions
    
    return pd.DataFrame(columns=['order_date', 'pizza_name', 'quantity'])

def prepare_pizza_sales_data(df):
    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Aggregate sales by pizza type and date with observed=True
    daily_pizza_sales = df.groupby(['order_date', 'pizza_name_id'], observed=True)['quantity'].sum().reset_index()
    
    # Create comprehensive time series features
    def create_time_features(df):
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['dayofweek'] = df['order_date'].dt.dayofweek
        df['quarter'] = df['order_date'].dt.quarter
        return df
    
    daily_pizza_sales = create_time_features(daily_pizza_sales)
    
    # Encode pizza names
    le = LabelEncoder()
    daily_pizza_sales['pizza_name_id_encoded'] = le.fit_transform(daily_pizza_sales['pizza_name_id'])
    
    # Create pivot table with observed=True
    pizza_pivot = daily_pizza_sales.pivot_table(
        index='order_date', 
        columns='pizza_name_id', 
        values='quantity', 
        fill_value=0,
        observed=True
    ).reset_index()
    pizza_pivot = pizza_pivot.apply(lambda column: column.astype('int64') if column.dtype == 'float64' else column)
    return pizza_pivot

def create_lagged_features(df, lag_days=[1, 7, 14, 30]):
    """
    Create lagged features for each pizza type with improved performance
    """
    df_lagged = df.copy()
    pizza_columns = df.columns[df.columns != 'order_date']
    
    # Create all lagged features at once
    lagged_features = {}
    for pizza in pizza_columns:
        for lag in lag_days:
            lagged_features[f'{pizza}_lag_{lag}'] = df_lagged[pizza].shift(lag)
    
    # Concatenate all features at once
    df_lagged = pd.concat([df_lagged, pd.DataFrame(lagged_features)], axis=1)
    
    # Remove rows with NaN (first lag periods)
    df_lagged = df_lagged.dropna()
    
    return df_lagged

def split_time_series(df, train_ratio=0.80):
    """
    Split time series data into train and test sets
    
    Parameters:
    - df: DataFrame with time series data
    - train_ratio: Proportion of data to use for training
    
    Returns:
    - train_df, test_df
    """
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    return train_df, test_df


# Class for predicting quantity with different models - SARIMA, PROPHET, XGBoost
class PizzaSalesModelValidator:
    def __init__(self, data):
        self.data = data
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.predictions = {}
        self.errors = {}
    
    def train_test_split(self, test_size=0.20, time_column='order_date'):
        """
        Split time series data into train and test sets
        
        Parameters:
        - test_size: Proportion of data to use for testing
        - time_column: Column to use for ordering
        
        Returns:
        - train and test dataframes
        """
        # Sort data by time
        sorted_data = self.data.sort_values(by=time_column)
        
        # Calculate split index
        split_index = int(len(sorted_data) * (1 - test_size))
        
        # Split data
        self.train_data = sorted_data.iloc[:split_index]
        self.test_data = sorted_data.iloc[split_index:]
        
        return self.train_data, self.test_data
    
    def validate_prophet_model(self, pizza_name_id):
        """
        Validate Prophet model for a specific pizza with integer predictions
        """
        from prophet import Prophet
        
        # Prepare data for Prophet
        train_df = self.train_data[['order_date', pizza_name_id]].rename(
            columns={'order_date': 'ds', pizza_name_id: 'y'}
        )
        test_df = self.test_data[['order_date', pizza_name_id]].rename(
            columns={'order_date': 'ds', pizza_name_id: 'y'}
        )
        
        # Fit Prophet model
        model = Prophet(
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=10,
            seasonality_mode='additive',
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(train_df)
        
        # Create future dataframe for exact test period
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=test_df['ds'].min(),
                end=test_df['ds'].max(),
                freq='D'
            )
        })
        
        # Make predictions
        forecast = model.predict(future)
        
        # Round predictions to nearest integer and ensure non-negative
        forecast['yhat'] = np.maximum(0, np.round(forecast['yhat']))
        forecast['yhat_lower'] = np.maximum(0, np.round(forecast['yhat_lower']))
        forecast['yhat_upper'] = np.maximum(0, np.round(forecast['yhat_upper']))
        
        # Merge with test data
        merged = pd.merge(
            test_df,
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='left'
        )
        
        # Filter out zero values for error calculation
        non_zero_index = merged[merged['y'] != 0].index
        valid_index = non_zero_index.intersection(merged.index)
        filtered_actual = merged['y'].loc[valid_index]
        filtered_predictions = merged['yhat'].loc[valid_index]
        
        # Calculate error metrics
        errors = {
            'MAE': mean_absolute_error(filtered_actual, filtered_predictions),
            'MSE': mean_squared_error(filtered_actual, filtered_predictions),
            'RMSE': np.sqrt(mean_squared_error(filtered_actual, filtered_predictions)),
            'MAPE': mean_absolute_percentage_error(filtered_actual, filtered_predictions),
            'R2': r2_score(filtered_actual, filtered_predictions)
        }
        
        # Store results
        self.models[f'prophet_{pizza_name_id}'] = model
        self.predictions[f'prophet_{pizza_name_id}'] = merged
        self.errors[f'prophet_{pizza_name_id}'] = errors
        
        return errors
    
    def validate_sarima_model(self, pizza_name_id, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Validate SARIMA model for a specific pizza with integer predictions
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # Convert to datetime and ensure consistent index
        self.train_data['order_date'] = pd.to_datetime(self.train_data['order_date'])
        self.test_data['order_date'] = pd.to_datetime(self.test_data['order_date'])
        
        # Set index and prepare series
        train_series = self.train_data.set_index('order_date')[pizza_name_id]
        test_series = self.test_data.set_index('order_date')[pizza_name_id]
        
        # Resample to handle potential missing dates
        train_series = train_series.resample('D').sum()
        test_series = test_series.resample('D').sum()
        
        # Fill NaN values
        train_series = train_series.fillna(0)
        test_series = test_series.fillna(0)
        
        try:
            # Fit SARIMA model
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit()
            
            # Forecast for test period
            forecast = results.get_forecast(steps=len(test_series))
            forecast_mean = pd.Series(
                # Round predictions to nearest integer and ensure non-negative
                np.maximum(0, np.round(forecast.predicted_mean.values)),
                index=test_series.index
            )
            
            # Filter out zero values
            non_zero_index = test_series[test_series != 0].index
            valid_index = non_zero_index.intersection(forecast_mean.index)
            
            # Filter the series to only include valid indices
            filtered_test_series = test_series.loc[valid_index]
            filtered_forecast_mean = forecast_mean.loc[valid_index]
            
            # Calculate metrics
            errors = {
                'MAE': mean_absolute_error(filtered_test_series, filtered_forecast_mean),
                'MSE': mean_squared_error(filtered_test_series, filtered_forecast_mean),
                'RMSE': np.sqrt(mean_squared_error(filtered_test_series, filtered_forecast_mean)),
                'MAPE': mean_absolute_percentage_error(filtered_test_series, filtered_forecast_mean),
                'R2': r2_score(filtered_test_series, filtered_forecast_mean)
            }
            
            # Store results
            self.models[f'sarima_{pizza_name_id}'] = results
            self.predictions[f'sarima_{pizza_name_id}'] = pd.DataFrame({
                'actual': test_series,
                'predicted': forecast_mean
            })
            self.errors[f'sarima_{pizza_name_id}'] = errors
            
            return errors
        
        except Exception as e:
            print(f"Error processing {pizza_name_id}: {str(e)}")
            return None
    
    def validate_xgboost_model(self, pizza_name_id):
        """
        Validate XGBoost model for a specific pizza
        
        Parameters:
        - pizza_name: Name of pizza to forecast
        
        Returns:
        - Dictionary of error metrics
        """
        from xgboost import XGBRegressor
        
        # Prepare features and target
        feature_columns = [col for col in self.train_data.columns 
                       if col not in ['order_date', pizza_name_id]]
    
        X_train = self.train_data[feature_columns]
        y_train = self.train_data[pizza_name_id]
        
        X_test = self.test_data[feature_columns]
        y_test = self.test_data[pizza_name_id]
        
        # Fit XGBoost model
        model = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        # Round predictions
        predictions = np.round(predictions)
        # Convert predictions to a pandas Series to have an index
        predictions = pd.Series(predictions, index=y_test.index)
        # Filter out zero values in the actual and predicted values
        non_zero_index = y_test[y_test != 0].index
        # Align the predictions with valid non-zero actual values
        valid_index = non_zero_index.intersection(predictions.index)

        # Filter the actual values and predictions to only include valid indices
        filtered_y_test = y_test.loc[valid_index]
        filtered_predictions = predictions.loc[valid_index]

        # Calculate error metrics for the filtered data
        errors = {
            'MAE': mean_absolute_error(filtered_y_test, filtered_predictions),
            'MSE': mean_squared_error(filtered_y_test, filtered_predictions),
            'RMSE': np.sqrt(mean_squared_error(filtered_y_test, filtered_predictions)),
            'MAPE': mean_absolute_percentage_error(filtered_y_test, filtered_predictions),
            'R2': r2_score(filtered_y_test, filtered_predictions)
        }
        
        # Store results
        self.models[f'xgboost_{pizza_name_id}'] = model
        self.predictions[f'xgboost_{pizza_name_id}'] = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        })
        self.errors[f'xgboost_{pizza_name_id}'] = errors
        
        return errors
    
    def plot_model_comparison(self, pizza_name_id):
        """
        Create comparison plots for different models with proper date formatting
        """
        plt.figure(figsize=(15, 10))
        
        # Actual vs Predicted for different models
        models = ['sarima', 'prophet', 'xgboost']
        
        for i, model_name in enumerate(models, 1):
            plt.subplot(2, 2, i)
            
            # Get predictions
            predictions = self.predictions.get(f'{model_name}_{pizza_name_id}')
            
            if predictions is not None:
                if model_name == 'prophet':
                    plt.plot(predictions['ds'], predictions['y'], label='Actual')
                    plt.plot(predictions['ds'], predictions['yhat'], label='Predicted', color='red')
                else:
                    # Ensure the index is datetime for XGBoost and SARIMA
                    if model_name == 'xgboost':
                        predictions.index = self.test_data['order_date']
                    plt.plot(predictions.index, predictions['actual'], label='Actual')
                    plt.plot(predictions.index, predictions['predicted'], label='Predicted', color='red')
                
                plt.title(f'{model_name.upper()} - {pizza_name_id}')
                plt.xlabel('Date')
                plt.ylabel('Sales Quantity')
                plt.legend()
                plt.xticks(rotation=45)
            
        # Error Comparison Bar Plot
        plt.subplot(2, 2, 4)
        error_data = {model: self.errors.get(f'{model}_{pizza_name_id}', {}).get('MAPE', 0) 
                    for model in models}
        plt.bar(error_data.keys(), error_data.values())
        plt.title('MAPE (Mean Absolute Percentage Error)')
        plt.ylabel('Error (%)')
        
        plt.tight_layout()
        plt.show()
    
    def compare_model_errors(self, pizza_names):
        """
        Compare errors across different models and pizzas
        
        Parameters:
        - pizza_names: List of pizza names to compare
        
        Returns:
        - DataFrame with error comparisons
        """
        error_comparison = {}
        
        for pizza_name in pizza_names:
            error_comparison[pizza_name] = {
                f'{model}_MAPE': self.errors.get(f'{model}_{pizza_name}', {}).get('MAPE', np.nan)
                for model in ['prophet', 'sarima', 'xgboost']
            }
        
        return pd.DataFrame.from_dict(error_comparison, orient='index')



# Class for selecting best models for
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedPizzaSalesModelSelector:
    def __init__(self, data):
        self.data = data
        self.models = {
            'prophet': self._validate_prophet,
            'sarima': self._validate_sarima,
            'xgboost': self._validate_xgboost
        }
        self.best_models = {}

    # Configure warning filter
        warnings.filterwarnings(
            "ignore", 
            message=".*Optimization terminated abnormally.*"
        )
        warnings.filterwarnings(
            "ignore", 
            message=".*Maximum Likelihood optimization failed.*"
        )
    
    def _validate_prophet(self, train_data, test_data, pizza_name_id):
        """
        Wrapper for Prophet model validation with multiple hyperparameter configurations
        """
        configs = [
            {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1},
            # Add more configurations here as needed
        ]

        best_errors = float('inf')
        best_config = None

        for config in configs:
            try:
                train_df = train_data[['order_date', pizza_name_id]].rename(
                    columns={'order_date': 'ds', pizza_name_id: 'y'}
                )
                test_df = test_data[['order_date', pizza_name_id]].rename(
                    columns={'order_date': 'ds', pizza_name_id: 'y'}
                )

                model = Prophet(
                    changepoint_prior_scale=config['changepoint_prior_scale'],
                    seasonality_prior_scale=config['seasonality_prior_scale'],
                seasonality_mode='additive',
                weekly_seasonality=True,
                daily_seasonality=False,
                yearly_seasonality=True
                )
                model.fit(train_df)

                future = pd.DataFrame({
                    'ds': pd.date_range(
                        start=test_df['ds'].min(),
                        end=test_df['ds'].max(),
                        freq='D'
                    )
                })

                forecast = model.predict(future)
                forecast['yhat'] = np.maximum(0, np.round(forecast['yhat']))

                # Calculate error (example: MAPE)
                from sklearn.metrics import mean_absolute_percentage_error
                mape = mean_absolute_percentage_error(
                    test_df['y'], 
                    forecast.set_index('ds').loc[test_df['ds']]['yhat']
                )

                if mape < best_errors:
                    best_errors = mape
                    best_config = {
                        'model': model,
                        'config': config,
                        'mape': mape
                    }

            except Exception as e:
                self._log_warning(f"Prophet model failed for {pizza_name_id}: {str(e)}")

        return best_config
    
    def _validate_sarima(self, train_data, test_data, pizza_name_id):
        """
        Wrapper for SARIMA model validation with multiple order configurations
        """        
        # Different order configurations
        order_configs = [
            {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
            {'order': (0, 1, 1), 'seasonal_order': (1, 1, 0, 12)},
            {'order': (1, 0, 1), 'seasonal_order': (0, 1, 1, 12)},
            {'order': (2, 1, 2), 'seasonal_order': (1, 1, 1, 12)},
            {'order': (0, 1, 2), 'seasonal_order': (0, 1, 1, 12)},
            {'order': (1, 1, 0), 'seasonal_order': (1, 0, 1, 12)},
            {'order': (2, 1, 1), 'seasonal_order': (0, 1, 0, 12)},
            {'order': (1, 1, 2), 'seasonal_order': (1, 1, 0, 12)}
        ]
        
        best_errors = float('inf')
        best_config = None
        
        for config in order_configs:
            try:
                # Prepare time series
                train_series = train_data.set_index('order_date')[pizza_name_id]
                test_series = test_data.set_index('order_date')[pizza_name_id]
                
                train_series = train_series.resample('D').sum().fillna(0)
                test_series = test_series.resample('D').sum().fillna(0)
                
                model = SARIMAX(
                    train_series,
                    order=config['order'],
                    seasonal_order=config['seasonal_order'],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit()
                
                # Forecast
                forecast = results.get_forecast(steps=len(test_series))
                forecast_mean = pd.Series(
                    np.maximum(0, np.round(forecast.predicted_mean.values)),
                    index=test_series.index
                )
                
                # Filter out zero values
                non_zero_index = test_series[test_series != 0].index
                valid_index = non_zero_index.intersection(forecast_mean.index)
                
                filtered_test_series = test_series.loc[valid_index]
                filtered_forecast_mean = forecast_mean.loc[valid_index]
                
                # Calculate MAPE
                from sklearn.metrics import mean_absolute_percentage_error
                mape = mean_absolute_percentage_error(
                    filtered_test_series, 
                    filtered_forecast_mean
                )
                
                if mape < best_errors:
                    best_errors = mape
                    best_config = {
                        'model': results,
                        'config': config,
                        'mape': mape
                    }
            
            except Exception as e:
                warnings.warn(f"SARIMA model failed for {pizza_name_id}: {str(e)}")
        
        return best_config
    
    def _validate_xgboost(self, train_data, test_data, pizza_name_id):
        """
        Wrapper for XGBoost model validation with multiple hyperparameter configurations
        """
        
        # Different hyperparameter configurations
        configs = [
            {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8},
            {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.9},
            {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 6, 'subsample': 1.0},
            {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8},
            {'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 7, 'subsample': 0.9},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.7},
            {'n_estimators': 50, 'learning_rate': 0.15, 'max_depth': 4, 'subsample': 0.8},
            {'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 6, 'subsample': 1.0}
        ]
        
        # Prepare features
        feature_columns = [col for col in train_data.columns 
                           if col not in ['order_date', pizza_name_id]]
        
        best_errors = float('inf')
        best_config = None
        
        for config in configs:
            try:
                X_train = train_data[feature_columns]
                y_train = train_data[pizza_name_id]
                
                X_test = test_data[feature_columns]
                y_test = test_data[pizza_name_id]
                
                model = XGBRegressor(
                    n_estimators=config['n_estimators'], 
                    learning_rate=config['learning_rate'], 
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Predict
                predictions = model.predict(X_test)
                predictions = pd.Series(predictions, index=y_test.index)
                
                # Filter out zero values
                non_zero_index = y_test[y_test != 0].index
                valid_index = non_zero_index.intersection(predictions.index)
                
                filtered_y_test = y_test.loc[valid_index]
                filtered_predictions = predictions.loc[valid_index]
                
                # Calculate MAPE
                from sklearn.metrics import mean_absolute_percentage_error
                mape = mean_absolute_percentage_error(
                    filtered_y_test, 
                    filtered_predictions
                )
                
                if mape < best_errors:
                    best_errors = mape
                    best_config = {
                        'model': model,
                        'config': config,
                        'mape': mape
                    }
            
            except Exception as e:
                warnings.warn(f"XGBoost model failed for {pizza_name_id}: {str(e)}")
        
        return best_config
    
    def select_best_model(self, train_data, test_data, pizza_name_id):
        """
        Select the best model for a specific pizza type
        """
        model_results = {}
        
        # Try each model
        for model_name, model_func in self.models.items():
            try:
                result = model_func(train_data, test_data, pizza_name_id)
                if result:
                    model_results[model_name] = result
            except Exception as e:
                warnings.warn(f"Model selection failed for {model_name}: {str(e)}")
        
        # Select model with lowest MAPE
        if model_results:
            best_model = min(
                model_results.items(), 
                key=lambda x: x[1]['mape']
            )
            
            self.best_models[pizza_name_id] = {
                'model_type': best_model[0],
                'model': best_model[1]['model'],
                'config': best_model[1]['config'],
                'mape': best_model[1]['mape']
            }
            
            return self.best_models[pizza_name_id]
        
        return None
    
    def select_models_for_all_pizzas(self):
        """Select best models for all pizza types."""      

        train_data, test_data = train_test_split(self.data, test_size=0.2, shuffle=False)
        pizza_columns = [col for col in self.data.columns if col != 'order_date']

        return {
            pizza_name: self.select_best_model(train_data, test_data, pizza_name)
            for pizza_name in pizza_columns
        }