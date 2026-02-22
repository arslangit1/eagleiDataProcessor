import pandas as pd
import xarray as xr
import numpy as np
import tabulate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Output directory for results
OUTPUT_DIR = 'results/Illinois/ml_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load NC file
NC_FILE_PATH = 'Illinois/merged_data_Illinois_Cook_2014_2024.nc'
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

    # Wind gust to speed ratio (indicates gustiness/variability)
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

    # ============================================================
    # DISPLAY ALL FEATURES USED FOR TRAINING
    # ============================================================
    print("\n" + "="*80)
    print("ALL FEATURES USED FOR MODEL TRAINING")
    print("="*80)

    print(f"\n📊 TOTAL FEATURES: {len(feature_columns)}")
    print(f"🎯 TARGET VARIABLE: {target_column}")

    print(f"\n1️⃣  LAG FEATURES ({len(lag_features)} features):")
    print("   These capture recent outage history patterns:")
    for i, feature in enumerate(lag_features, 1):
        print(f"   • {feature:<25} - Customers out {i} timestep(s) ago")

    print(f"\n2️⃣  TIME FEATURES ({len(time_features)} features):")
    print("   These capture temporal patterns and event progression:")
    time_descriptions = {
        'hour': 'Hour of day (0-23)',
        'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
        'month': 'Month of year (1-12)',
        'hours_since_start': 'Hours since event started'
    }
    for feature in time_features:
        print(f"   • {feature:<25} - {time_descriptions[feature]}")

    print(f"\n3️⃣  ROLLING STATISTICS FEATURES ({len(rolling_features)} features):")
    print("   These capture short-term trends and volatility:")
    rolling_descriptions = {
        'rolling_mean_3': 'Average of last 3 outage values',
        'rolling_std_3': 'Standard deviation of last 3 values',
        'rolling_max_3': 'Maximum of last 3 outage values',
        'rate_of_change': 'Change from previous timestep'
    }
    for feature in rolling_features:
        print(f"   • {feature:<25} - {rolling_descriptions[feature]}")

    print(f"\n4️⃣  WEATHER FEATURES ({len(weather_features)} features):")
    print("   These capture current and historical weather conditions:")

    # Group weather features by type
    current_weather = [f for f in weather_features if not ('_lag_' in f or '_rolling_' in f) and f not in ['gust_speed_ratio', 'precip_occurring']]
    lag_weather = [f for f in weather_features if '_lag_1' in f]
    rolling_weather = [f for f in weather_features if '_rolling_' in f]
    derived_weather = [f for f in weather_features if f in ['gust_wind', 'gust_speed_ratio', 'precip_occurring']]

    if current_weather:
        print(f"   \n   📡 Current Weather ({len(current_weather)} features):")
        weather_descriptions = {
            'tmpf': 'Temperature (°F)',
            'sknt': 'Wind speed (knots)',
            'gust': 'Wind gust speed (knots)',
            'p01i': 'Precipitation (inches)'
        }
        for feature in current_weather:
            desc = weather_descriptions.get(feature, 'Weather measurement')
            print(f"   • {feature:<25} - {desc}")

    if lag_weather:
        print(f"   \n   🔄 Lagged Weather ({len(lag_weather)} features):")
        for feature in lag_weather:
            base_feature = feature.replace('_lag_1', '')
            desc = weather_descriptions.get(base_feature, 'Weather measurement')
            print(f"   • {feature:<25} - {desc} (1 timestep ago)")

    if rolling_weather:
        print(f"   \n   📈 Rolling Weather Stats ({len(rolling_weather)} features):")
        for feature in rolling_weather:
            if '_rolling_mean_3' in feature:
                base_feature = feature.replace('_rolling_mean_3', '')
                desc = weather_descriptions.get(base_feature, 'Weather measurement')
                print(f"   • {feature:<25} - {desc} (3-period average)")
            elif '_rolling_max_3' in feature:
                base_feature = feature.replace('_rolling_max_3', '')
                desc = weather_descriptions.get(base_feature, 'Weather measurement')
                print(f"   • {feature:<25} - {desc} (3-period maximum)")

    if derived_weather:
        print(f"   \n   🧮 Derived Weather Features ({len(derived_weather)} features):")
        derived_descriptions = {
            'gust_wind': 'Effective wind speed (gust when available, otherwise wind speed)',
            'gust_speed_ratio': 'Wind gust to speed ratio (gustiness indicator)',
            'precip_occurring': 'Binary indicator of precipitation occurrence'
        }
        for feature in derived_weather:
            desc = derived_descriptions.get(feature, 'Derived weather measurement')
            print(f"   • {feature:<25} - {desc}")

    print(f"\n" + "="*80)
    print("FEATURE SUMMARY BY CATEGORY")
    print("="*80)
    print(f"• Historical Outage Data:  {len(lag_features + rolling_features)} features")
    print(f"• Temporal Information:    {len(time_features)} features")
    print(f"• Weather Information:     {len(weather_features)} features")
    print(f"• TOTAL MODEL FEATURES:    {len(feature_columns)} features")
    print("="*80)


    X = ml_data[feature_columns].fillna(0)
    y = ml_data[target_column]

    # Time-series aware train-test split (use last 20% of data as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ============================================================
    # TRAIN RANDOM FOREST MODEL
    # ============================================================
    print("\n" + "-"*40)
    print("Training Random Forest Regressor...")
    print("-"*40)

    # Initialize and train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,          # Number of trees
        max_depth=15,              # Maximum depth of trees
        min_samples_split=5,       # Minimum samples to split
        min_samples_leaf=2,        # Minimum samples in leaf
        max_features='sqrt',       # Features to consider at each split
        random_state=42,
        n_jobs=-1                  # Use all CPU cores
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # ============================================================
    # MODEL EVALUATION
    # ============================================================
    print("\n" + "-"*40)
    print("MODEL EVALUATION")
    print("-"*40)

    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100  # Convert to percentage
    train_r2 = r2_score(y_train, y_pred_train)

    # Testing metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100  # Convert to percentage
    test_r2 = r2_score(y_test, y_pred_test)

    print("\nTraining Performance:")
    print(f"  RMSE: {train_rmse:,.2f} customers")
    print(f"  MAE:  {train_mae:,.2f} customers")
    print(f"  MAPE: {train_mape:.2f}%")
    print(f"  R²:   {train_r2:.4f}")

    print("\nTesting Performance:")
    print(f"  RMSE: {test_rmse:,.2f} customers")
    print(f"  MAE:  {test_mae:,.2f} customers")
    print(f"  MAPE: {test_mape:.2f}%")
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

    # Set font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12

    # Training
    axes[0].scatter(y_train, y_pred_train, alpha=0.5, s=10)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Customers Out', fontsize=label_fontsize)
    axes[0].set_ylabel('Predicted Customers Out', fontsize=label_fontsize)
    axes[0].set_title(f'Training Set (R² = {train_r2:.4f})', fontsize=title_fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[0].grid(True, alpha=0.3)

    # Testing
    axes[1].scatter(y_test, y_pred_test, alpha=0.5, s=10, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Customers Out', fontsize=label_fontsize)
    axes[1].set_ylabel('Predicted Customers Out', fontsize=label_fontsize)
    axes[1].set_title(f'Testing Set (R² = {test_r2:.4f})', fontsize=title_fontsize)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rf_actual_vs_predicted_cook.pdf', dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Prediction on Sample Events
    print("\n" + "-"*40)
    print("PREDICTION EXAMPLES ON TEST EVENTS")
    print("-"*40)

    test_event_ids = ml_data.iloc[split_idx:]['event_id'].unique()[:3]

    fig, axes = plt.subplots(len(test_event_ids), 1, figsize=(14, 4 * len(test_event_ids)))
    if len(test_event_ids) == 1:
        axes = [axes]

    # Set font sizes
    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 14
    legend_fontsize = 16

    for i, event_id in enumerate(test_event_ids):
        # Get event data from test set
        test_mask = (ml_data['event_id'] == event_id) & (ml_data.index >= ml_data.index[split_idx])
        event_test = ml_data[test_mask]

        if len(event_test) > 0:
            X_event = event_test[feature_columns].fillna(0)
            y_event_actual = event_test['customers_out']
            y_event_pred = rf_model.predict(X_event)

            axes[i].plot(range(len(y_event_actual)), y_event_actual.values,
                         'b-', linewidth=2, label='Actual', marker='o', markersize=6)
            axes[i].plot(range(len(y_event_pred)), y_event_pred,
                         'r--', linewidth=2, label='Predicted', marker='x', markersize=6)
            axes[i].set_xlabel('Time Step', fontsize=label_fontsize)
            axes[i].set_ylabel('Customers Out', fontsize=label_fontsize)
            axes[i].set_title(f'Event {int(event_id)}', fontsize=title_fontsize)
            axes[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            axes[i].legend(fontsize=legend_fontsize)
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rf_event_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

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
            gust = weather_current.get('gust', None)
            sknt = weather_current.get('sknt', 0.1)

            # Use gust if available, otherwise use wind speed
            if gust is not None:
                gust_wind = gust
            else:
                gust_wind = sknt

            features['gust_wind'] = gust_wind
            features['gust_speed_ratio'] = gust_wind / (sknt + 0.1)
            features['precip_occurring'] = 1 if weather_current.get('p01i', 0) > 0 else 0

        # Create feature vector
        feature_vector = pd.DataFrame([features])[feature_columns].fillna(0)

        return rf_model.predict(feature_vector)[0]

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
    import joblib

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
        'train_mape': train_mape,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'cv_mean_r2': cv_scores.mean(),
        'cv_std_r2': cv_scores.std(),
        'n_samples': len(X),
        'n_events': ml_data['event_id'].nunique(),
        'n_features': len(feature_columns)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{OUTPUT_DIR}/model_metrics.csv', index=False)
    print(f"Metrics saved to: {OUTPUT_DIR}/model_metrics.csv")

    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING COMPLETE!")
    print("="*60)

else:
    print("ERROR: Not enough event data for machine learning. Check your dataset.")
