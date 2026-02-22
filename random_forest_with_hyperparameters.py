import pandas as pd
import xarray as xr
import numpy as np
import tabulate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

# Output directory for results
OUTPUT_DIR = 'results/Illinois/ml_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load NC file
NC_FILE_PATH = 'Illinois/merged_data_Illinois_Bureau_2014_2024.nc'
ds = xr.open_dataset(NC_FILE_PATH)
# Convert xarray dataset to pandas DataFrame
df = ds.to_dataframe().reset_index()
print("="*60)
print("DATA OVERVIEW")
print("="*60)
print(tabulate.tabulate(df.head(), headers='keys', tablefmt='plain'))
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

#print event plots for each year
event_peaks = df[df['event_number_eaglei'] > 0].groupby('event_number_eaglei')['customers_out'].max()
top_5_events = event_peaks.nlargest(10).index.tolist()

print(f"\nTop 10 events: {top_5_events}")
print(f"Peak outages: {event_peaks.nlargest(10).values}")

# Plot each of the top 10 events
fig, axes = plt.subplots(10, 1, figsize=(15, 12), sharex=False)

for i, event_id in enumerate(top_5_events):
    event_df = df[df['event_number_eaglei'] == event_id]

    axes[i].plot(event_df['time'], event_df['customers_out'], linewidth=1.5)
    axes[i].set_title(f'Event {int(event_id)} - Peak: {event_df["customers_out"].max():,.0f} customers')
    axes[i].set_ylabel('Customers Out')
    axes[i].grid(True, alpha=0.3)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top_10_events.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# RANDOM FOREST FOR TIME SERIES PREDICTION
# ============================================================
print("\n" + "="*60)
print("RANDOM FOREST TIME SERIES MODEL FOR OUTAGE PREDICTION")
print("="*60)

def create_time_series_features(event_df, n_lags=5):
    """
    Create lagged features for time series prediction within an event.

    Parameters:
    - event_df: DataFrame for a single event
    - n_lags: Number of lag features to create

    Returns:
    - DataFrame with lag features
    """
    event_df = event_df.sort_values('time').copy()

    # Create lag features (previous timestep values)
    for lag in range(1, n_lags + 1):
        event_df[f'customers_out_lag_{lag}'] = event_df['customers_out'].shift(lag)

    # Create time-based features
    event_df['hour'] = event_df['time'].dt.hour
    event_df['day_of_week'] = event_df['time'].dt.dayofweek
    event_df['month'] = event_df['time'].dt.month

    # Time since event start (in hours)
    event_start = event_df['time'].min()
    event_df['hours_since_start'] = (event_df['time'] - event_start).dt.total_seconds() / 3600

    # Rolling statistics
    event_df['rolling_mean_3'] = event_df['customers_out'].shift(1).rolling(window=3, min_periods=1).mean()
    event_df['rolling_std_3'] = event_df['customers_out'].shift(1).rolling(window=3, min_periods=1).std()
    event_df['rolling_max_3'] = event_df['customers_out'].shift(1).rolling(window=3, min_periods=1).max()

    # Rate of change
    event_df['rate_of_change'] = event_df['customers_out'].shift(1).diff()

    # ============================================================
    # WEATHER FEATURES
    # ============================================================
    # Current weather conditions
    weather_cols = ['tmpf', 'sknt', 'gust', 'p01i']

    for col in weather_cols:
        if col in event_df.columns:
            # Current value (already available)
            # Create lag features for weather
            event_df[f'{col}_lag_1'] = event_df[col].shift(1)

            # Rolling weather statistics (3-hour window)
            event_df[f'{col}_rolling_mean_3'] = event_df[col].shift(1).rolling(window=3, min_periods=1).mean()
            event_df[f'{col}_rolling_max_3'] = event_df[col].shift(1).rolling(window=3, min_periods=1).max()

    # Use gust if available, otherwise use wind speed
    if 'sknt' in event_df.columns:
        if 'gust' in event_df.columns:
            # Use gust when available
            event_df['gust_wind'] = event_df['gust'].fillna(event_df['sknt'])  # fallback to wind speed if gust is NaN
        else:
            # Use wind speed if gust not available
            event_df['gust_wind'] = event_df['sknt']

        event_df['gust_speed_ratio'] = event_df['gust_wind'] / (event_df['sknt'] + 0.1)  # avoid division by zero

    # Precipitation occurrence (binary)
    if 'poccurence' in event_df.columns:
        event_df['precip_occurring'] = event_df['poccurence']

    return event_df

def prepare_event_data_for_ml(df, n_lags=5):
    """
    Prepare all events data for machine learning.
    """
    # Filter only event periods (where event_number_eaglei > 0)
    event_df = df[df['event_number_eaglei'] > 0].copy()

    all_events_processed = []

    for event_id in event_df['event_number_eaglei'].unique():
        single_event = event_df[event_df['event_number_eaglei'] == event_id].copy()

        if len(single_event) > n_lags + 2:  # Need enough data points
            processed_event = create_time_series_features(single_event, n_lags)
            processed_event['event_id'] = event_id
            all_events_processed.append(processed_event)

    if all_events_processed:
        combined_df = pd.concat(all_events_processed, ignore_index=True)
        # Drop rows with NaN (from lag creation)
        combined_df = combined_df.dropna()
        return combined_df
    return None

# Prepare data
print("\nPreparing event data for machine learning...")
n_lags = 5
ml_data = prepare_event_data_for_ml(df, n_lags=n_lags)

if ml_data is not None and len(ml_data) > 0:
    print(f"Total samples for training: {len(ml_data)}")
    print(f"Number of events used: {ml_data['event_id'].nunique()}")

    # Define features and target
    lag_features = [f'customers_out_lag_{i}' for i in range(1, n_lags + 1)]
    time_features = ['hour', 'day_of_week', 'month', 'hours_since_start']
    rolling_features = ['rolling_mean_3', 'rolling_std_3', 'rolling_max_3', 'rate_of_change']

    # Weather features
    weather_base_cols = ['tmpf', 'sknt', 'gust', 'p01i']
    weather_features = []
    for col in weather_base_cols:
        weather_features.append(col)  # Current value
        weather_features.append(f'{col}_lag_1')  # Lagged value
        weather_features.append(f'{col}_rolling_mean_3')  # Rolling mean
        weather_features.append(f'{col}_rolling_max_3')  # Rolling max

    # Additional derived weather features
    weather_features.extend(['gust_wind', 'gust_speed_ratio', 'precip_occurring'])

    # Filter to only include features that exist in the data
    weather_features = [f for f in weather_features if f in ml_data.columns]

    print(f"\nWeather features used: {weather_features}")

    feature_columns = lag_features + time_features + rolling_features + weather_features
    target_column = 'customers_out'


    X = ml_data[feature_columns].fillna(0)
    y = ml_data[target_column]

    # Time-series aware train-test split (use last 20% of data as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ============================================================
    # HYPERPARAMETER TUNING
    # ============================================================
    print("\n" + "-"*40)
    print("HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("-"*40)

    # Define the hyperparameter search space
    param_dist = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [5, 10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False]
    }

    # Base model for tuning
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Use TimeSeriesSplit for cross-validation during hyperparameter search
    tscv_tuning = TimeSeriesSplit(n_splits=3)

    # RandomizedSearchCV for faster hyperparameter search
    print("\nSearching for optimal hyperparameters...")
    print("This may take a few minutes...")

    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=30,                    # Number of random combinations to try
        cv=tscv_tuning,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    # Get best parameters
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("\n" + "-"*40)
    print("BEST HYPERPARAMETERS FOUND")
    print("-"*40)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV R² Score: {best_score:.4f}")

    # Save best parameters
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(f'{OUTPUT_DIR}/best_hyperparameters.csv', index=False)
    print(f"\nBest parameters saved to: {OUTPUT_DIR}/best_hyperparameters.csv")

    # ============================================================
    # TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
    # ============================================================
    print("\n" + "-"*40)
    print("Training Final Random Forest with Best Hyperparameters...")
    print("-"*40)

    # Use the best estimator from RandomizedSearchCV
    rf_model = random_search.best_estimator_

    # Alternatively, you can create a new model with best params:
    # rf_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    # rf_model.fit(X_train, y_train)

    # Predictions (convert to integers since customers_out should be whole numbers)
    y_pred_train = rf_model.predict(X_train).astype(int)
    y_pred_test = rf_model.predict(X_test).astype(int)

    # ============================================================
    # MODEL EVALUATION
    # ============================================================
    print("\n" + "-"*40)
    print("MODEL EVALUATION")
    print("-"*40)

    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    # Testing metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\nTraining Performance:")
    print(f"  RMSE: {train_rmse:,.2f} customers")
    print(f"  MAE:  {train_mae:,.2f} customers")
    print(f"  R²:   {train_r2:.4f}")

    print("\nTesting Performance:")
    print(f"  RMSE: {test_rmse:,.2f} customers")
    print(f"  MAE:  {test_mae:,.2f} customers")
    print(f"  R²:   {test_r2:.4f}")

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "-"*40)
    print("FEATURE IMPORTANCE")
    print("-"*40)

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(tabulate.tabulate(feature_importance.head(10), headers='keys', tablefmt='plain', floatfmt='.4f'))

    # ============================================================
    # VISUALIZATIONS
    # ============================================================

    # 1. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest - Top 10 Feature Importances')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rf_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Actual vs Predicted Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training
    axes[0].scatter(y_train, y_pred_train, alpha=0.5, s=10)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Customers Out')
    axes[0].set_ylabel('Predicted Customers Out')
    axes[0].set_title(f'Training Set (R² = {train_r2:.4f})')
    axes[0].grid(True, alpha=0.3)

    # Testing
    axes[1].scatter(y_test, y_pred_test, alpha=0.5, s=10, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Customers Out')
    axes[1].set_ylabel('Predicted Customers Out')
    axes[1].set_title(f'Testing Set (R² = {test_r2:.4f})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rf_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Prediction on Sample Events with Weather Variables
    print("\n" + "-"*40)
    print("PREDICTION EXAMPLES ON TEST EVENTS WITH WEATHER VARIABLES")
    print("-"*40)

    # Get 3 random test events instead of first 3
    test_event_ids = ml_data.iloc[split_idx:]['event_id'].unique()
    np.random.seed(42)  # For reproducible results
    if len(test_event_ids) >= 3:
        random_test_events = np.random.choice(test_event_ids, size=3, replace=False)
    else:
        random_test_events = test_event_ids[:3]

    print(f"Plotting 3 random test events: {random_test_events}")


    # ============================================================
    # CORRELATION ANALYSIS BETWEEN WEATHER AND OUTAGES
    # ============================================================
    print("\n" + "-"*40)
    print("WEATHER-OUTAGE CORRELATION ANALYSIS")
    print("-"*40)

    # Calculate correlations for the test events
    correlations_data = []

    for event_id in random_test_events:
        test_mask = (ml_data['event_id'] == event_id) & (ml_data.index >= ml_data.index[split_idx])
        event_test = ml_data[test_mask]

        if len(event_test) > 3:
            correlations = {}
            correlations['event_id'] = event_id

            for weather_var in weather_vars:
                if weather_var in event_test.columns:
                    corr = event_test['customers_out'].corr(event_test[weather_var])
                    correlations[f'{weather_var}_correlation'] = corr
                else:
                    correlations[f'{weather_var}_correlation'] = np.nan

            correlations_data.append(correlations)

    if correlations_data:
        corr_df = pd.DataFrame(correlations_data)
        print("\nWeather-Outage Correlations for Test Events:")
        print(tabulate.tabulate(corr_df, headers='keys', tablefmt='plain', floatfmt='.3f'))

        # Save correlations
        corr_df.to_csv(f'{OUTPUT_DIR}/weather_outage_correlations.csv', index=False)
        print(f"\nCorrelations saved to: {OUTPUT_DIR}/weather_outage_correlations.csv")

    # ============================================================
    # BEST PREDICTION EVENTS (PERFECT/NEAR-PERFECT MATCHES)
    # ============================================================
    print("\n" + "-"*40)
    print("FINDING BEST PREDICTION EVENTS (PERFECT MATCHES)")
    print("-"*40)

    # Calculate R² score for each event in the dataset
    event_scores = []
    all_event_ids = ml_data['event_id'].unique()

    for event_id in all_event_ids:
        event_mask = ml_data['event_id'] == event_id
        event_data = ml_data[event_mask]

        if len(event_data) >= 5:  # Need minimum samples
            X_event = event_data[feature_columns].fillna(0)
            y_event_actual = event_data['customers_out']
            y_event_pred = rf_model.predict(X_event).astype(int)

            # Calculate metrics for this event
            event_r2 = r2_score(y_event_actual, y_event_pred)
            event_rmse = np.sqrt(mean_squared_error(y_event_actual, y_event_pred))
            event_mae = mean_absolute_error(y_event_actual, y_event_pred)
            peak_outage = y_event_actual.max()
            mape = np.mean(np.abs((y_event_actual - y_event_pred) / (y_event_actual + 1))) * 100  # MAPE

            event_scores.append({
                'event_id': event_id,
                'r2': event_r2,
                'rmse': event_rmse,
                'mae': event_mae,
                'mape': mape,
                'peak_outage': peak_outage,
                'n_samples': len(event_data)
            })

    event_scores_df = pd.DataFrame(event_scores)
    event_scores_df = event_scores_df.sort_values('r2', ascending=False)

    print("\nTop 10 Best Predicted Events (by R²):")
    print(tabulate.tabulate(event_scores_df.head(10), headers='keys', tablefmt='plain', floatfmt='.4f'))

    # Save event scores
    event_scores_df.to_csv(f'{OUTPUT_DIR}/event_prediction_scores.csv', index=False)
    print(f"\nEvent scores saved to: {OUTPUT_DIR}/event_prediction_scores.csv")

    # ============================================================
    # PLOT TOP 6 BEST PREDICTED EVENTS
    # ============================================================
    print("\n" + "-"*40)
    print("PLOTTING BEST PREDICTED EVENTS")
    print("-"*40)

    # Get top 6 events with best R² scores (and significant outages)
    best_events = event_scores_df[event_scores_df['peak_outage'] > 100].head(6)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (_, row) in enumerate(best_events.iterrows()):
        event_id = row['event_id']
        event_r2 = row['r2']
        event_mae = row['mae']

        event_mask = ml_data['event_id'] == event_id
        event_data = ml_data[event_mask]

        X_event = event_data[feature_columns].fillna(0)
        y_event_actual = event_data['customers_out']
        y_event_pred = rf_model.predict(X_event).astype(int)

        # Plot
        axes[i].plot(range(len(y_event_actual)), y_event_actual.values,
                    'b-', linewidth=2, label='Actual', marker='o', markersize=4, alpha=0.8)
        axes[i].plot(range(len(y_event_pred)), y_event_pred,
                    'r--', linewidth=2, label='Predicted', marker='s', markersize=4, alpha=0.8)

        # Fill between to show match quality
        axes[i].fill_between(range(len(y_event_actual)),
                            y_event_actual.values, y_event_pred,
                            alpha=0.2, color='green')

        axes[i].set_xlabel('Time Step (15-min intervals)')
        axes[i].set_ylabel('Customers Out')
        axes[i].set_title(f'Event {int(event_id)} | R² = {event_r2:.4f} | MAE = {event_mae:.1f}',
                         fontsize=11, fontweight='bold')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

        # Add peak annotation
        peak_idx = np.argmax(y_event_actual.values)
        axes[i].annotate(f'Peak: {y_event_actual.max():,.0f}',
                        xy=(peak_idx, y_event_actual.max()),
                        xytext=(peak_idx + 2, y_event_actual.max() * 1.05),
                        fontsize=9, color='blue')

    plt.suptitle('🎯 BEST PREDICTED EVENTS (Near-Perfect Matches)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rf_best_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ============================================================
    # PLOT EVENTS WITH DIFFERENT PREDICTION QUALITY LEVELS
    # ============================================================
    print("\n" + "-"*40)
    print("PREDICTION QUALITY COMPARISON")
    print("-"*40)

    # Get events from different quality tiers
    excellent_events = event_scores_df[(event_scores_df['r2'] >= 0.99) & (event_scores_df['peak_outage'] > 100)].head(2)
    good_events = event_scores_df[(event_scores_df['r2'] >= 0.95) & (event_scores_df['r2'] < 0.99) & (event_scores_df['peak_outage'] > 100)].head(2)
    moderate_events = event_scores_df[(event_scores_df['r2'] >= 0.85) & (event_scores_df['r2'] < 0.95) & (event_scores_df['peak_outage'] > 100)].head(2)

    quality_events = pd.concat([excellent_events, good_events, moderate_events])

    if len(quality_events) >= 4:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        quality_labels = ['🌟 EXCELLENT', '🌟 EXCELLENT', '✅ GOOD', '✅ GOOD', '📊 MODERATE', '📊 MODERATE']

        for i, (_, row) in enumerate(quality_events.iterrows()):
            if i >= 6:
                break

            event_id = row['event_id']
            event_r2 = row['r2']
            event_mae = row['mae']

            event_mask = ml_data['event_id'] == event_id
            event_data = ml_data[event_mask]

            X_event = event_data[feature_columns].fillna(0)
            y_event_actual = event_data['customers_out']
            y_event_pred = rf_model.predict(X_event).astype(int)

            # Different colors for different quality
            if i < 2:
                color_actual, color_pred = 'darkgreen', 'limegreen'
            elif i < 4:
                color_actual, color_pred = 'darkblue', 'dodgerblue'
            else:
                color_actual, color_pred = 'darkorange', 'orange'

            axes[i].plot(range(len(y_event_actual)), y_event_actual.values,
                        '-', color=color_actual, linewidth=2, label='Actual', marker='o', markersize=4)
            axes[i].plot(range(len(y_event_pred)), y_event_pred,
                        '--', color=color_pred, linewidth=2, label='Predicted', marker='x', markersize=4)

            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Customers Out')
            axes[i].set_title(f'{quality_labels[i]} | Event {int(event_id)} | R² = {event_r2:.4f}',
                             fontsize=11, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('Prediction Quality Comparison: Excellent vs Good vs Moderate', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/rf_quality_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Print summary statistics
    print("\n📊 PREDICTION QUALITY SUMMARY:")
    print(f"  Events with R² ≥ 0.99 (Excellent): {len(event_scores_df[event_scores_df['r2'] >= 0.99])}")
    print(f"  Events with R² ≥ 0.95 (Good):      {len(event_scores_df[event_scores_df['r2'] >= 0.95])}")
    print(f"  Events with R² ≥ 0.85 (Moderate):  {len(event_scores_df[event_scores_df['r2'] >= 0.85])}")
    print(f"  Events with R² < 0.85 (Poor):      {len(event_scores_df[event_scores_df['r2'] < 0.85])}")

    # ============================================================
    # TIME SERIES CROSS-VALIDATION
    # ============================================================
    print("\n" + "-"*40)
    print("TIME SERIES CROSS-VALIDATION")
    print("-"*40)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='r2')

    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # ============================================================
    # PREDICT NEXT STEP FUNCTION
    # ============================================================
    def predict_next_customers_out(rf_model, recent_history, hour, day_of_week, month, hours_since_start,
                                   weather_current=None, weather_history=None):
        """
        Predict the next customers_out value given recent history and weather.

        Parameters:
        - rf_model: Trained Random Forest model
        - recent_history: List of recent customers_out values (at least n_lags values)
        - hour, day_of_week, month: Time features
        - hours_since_start: Hours since event started
        - weather_current: Dict with current weather {'tmpf': val, 'sknt': val, 'gust': val, 'p01i': val}
        - weather_history: List of dicts with recent weather history (for rolling features)

        Returns:
        - Predicted customers_out value
        """
        # Create lag features
        features = {}
        for i, lag in enumerate(range(1, n_lags + 1)):
            if i < len(recent_history):
                features[f'customers_out_lag_{lag}'] = recent_history[-(i+1)]
            else:
                features[f'customers_out_lag_{lag}'] = 0

        # Time features
        features['hour'] = hour
        features['day_of_week'] = day_of_week
        features['month'] = month
        features['hours_since_start'] = hours_since_start

        # Rolling features
        recent_vals = recent_history[-3:] if len(recent_history) >= 3 else recent_history
        features['rolling_mean_3'] = np.mean(recent_vals)
        features['rolling_std_3'] = np.std(recent_vals) if len(recent_vals) > 1 else 0
        features['rolling_max_3'] = np.max(recent_vals)
        features['rate_of_change'] = recent_history[-1] - recent_history[-2] if len(recent_history) >= 2 else 0

        # Weather features
        weather_cols = ['tmpf', 'sknt', 'gust', 'p01i']
        if weather_current is not None:
            for col in weather_cols:
                features[col] = weather_current.get(col, 0)

            # Weather lag features (use current as proxy if no history)
            if weather_history and len(weather_history) >= 1:
                for col in weather_cols:
                    features[f'{col}_lag_1'] = weather_history[-1].get(col, 0)
            else:
                for col in weather_cols:
                    features[f'{col}_lag_1'] = weather_current.get(col, 0)

            # Weather rolling features
            if weather_history and len(weather_history) >= 3:
                recent_weather = weather_history[-3:]
                for col in weather_cols:
                    vals = [w.get(col, 0) for w in recent_weather]
                    features[f'{col}_rolling_mean_3'] = np.mean(vals)
                    features[f'{col}_rolling_max_3'] = np.max(vals)
            else:
                for col in weather_cols:
                    features[f'{col}_rolling_mean_3'] = weather_current.get(col, 0)
                    features[f'{col}_rolling_max_3'] = weather_current.get(col, 0)

            # Derived features
            gust = weather_current.get('gust', 0)
            sknt = weather_current.get('sknt', 0.1)

            # Create gust_wind feature (use gust if available, otherwise wind speed)
            if gust > 0:
                features['gust_wind'] = gust
            else:
                features['gust_wind'] = sknt

            features['gust_speed_ratio'] = features['gust_wind'] / (sknt + 0.1)
            features['precip_occurring'] = 1 if weather_current.get('p01i', 0) > 0 else 0
        else:
            # Default weather features when no weather data is provided
            for col in weather_cols:
                features[col] = 0
                features[f'{col}_lag_1'] = 0
                features[f'{col}_rolling_mean_3'] = 0
                features[f'{col}_rolling_max_3'] = 0
            features['gust_wind'] = 0
            features['gust_speed_ratio'] = 0
            features['precip_occurring'] = 0

        # Create feature vector
        feature_vector = pd.DataFrame([features])[feature_columns].fillna(0)

        return int(rf_model.predict(feature_vector)[0])

    # Demo prediction
    print("\n" + "-"*40)
    print("DEMO: NEXT STEP PREDICTION (WITH WEATHER)")
    print("-"*40)

    sample_history = y_test.iloc[:10].tolist()

    # Sample weather conditions (e.g., stormy weather)
    sample_weather = {
        'tmpf': 65.0,   # Temperature in Fahrenheit
        'sknt': 25.0,   # Wind speed in knots
        'gust': 45.0,   # Wind gust in knots
        'p01i': 0.5     # Precipitation in inches
    }

    # Weather history (last 3 observations)
    sample_weather_history = [
        {'tmpf': 68.0, 'sknt': 20.0, 'gust': 35.0, 'p01i': 0.2},
        {'tmpf': 66.0, 'sknt': 22.0, 'gust': 40.0, 'p01i': 0.3},
        {'tmpf': 65.0, 'sknt': 24.0, 'gust': 42.0, 'p01i': 0.4},
    ]

    next_pred = predict_next_customers_out(
        rf_model,
        sample_history,
        hour=14,
        day_of_week=2,
        month=7,
        hours_since_start=5.0,
        weather_current=sample_weather,
        weather_history=sample_weather_history
    )
    print(f"\nGiven recent outage history: {[f'{x:,.0f}' for x in sample_history[-5:]]}")
    print(f"Current weather: Temp={sample_weather['tmpf']}°F, Wind={sample_weather['sknt']}kts, Gust={sample_weather['gust']}kts, Precip={sample_weather['p01i']}in")
    print(f"Predicted next customers_out: {next_pred:,.0f}")

    # ============================================================
    # SAVE MODEL AND RESULTS
    # ============================================================

    # Save the trained model
    model_path = f'{OUTPUT_DIR}/random_forest_model.joblib'
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save feature importance to CSV
    feature_importance.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
    print(f"Feature importance saved to: {OUTPUT_DIR}/feature_importance.csv")

    # Save evaluation metrics
    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'cv_mean_r2': cv_scores.mean(),
        'cv_std_r2': cv_scores.std(),
        'n_samples': len(X),
        'n_events': ml_data['event_id'].nunique(),
        'n_features': len(feature_columns),
        'best_cv_r2_tuning': best_score
    }
    # Add best hyperparameters to metrics
    metrics.update({f'best_{k}': v for k, v in best_params.items()})

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{OUTPUT_DIR}/model_metrics.csv', index=False)
    print(f"Metrics saved to: {OUTPUT_DIR}/model_metrics.csv")

    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING COMPLETE!")
    print("="*60)

else:
    print("ERROR: Not enough event data for machine learning. Check your dataset.")
