import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('merged_gw.csv', on_bad_lines='skip')

# sory by name then gameweek
df = df.sort_values(by=['name', 'GW'])

# calculate rolling 5 gameweek form
form_series = df.groupby('name')['total_points'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
form_series = form_series.reset_index(level=0, drop=True)
df['form_5_gw'] = form_series


# HOME VS AWAY FEATURE
df['is_home'] = df['was_home'].astype(int) # converting boolean True/False to 1/0

# GETTING TEAM ATTACKING STRENGTH
# calculating average points scored by each team
team_attack = df.groupby('team')['total_points'].mean().to_dict()
# mapping that strength to each player's row
df['team_strength'] = df['team'].map(team_attack)

# CALCULATING ROLLING 5 GAMEWEEK XG (EXPECTED GOALS)
xg_series = df.groupby('name')['expected_goals'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
xg_series = xg_series.reset_index(level=0, drop=True)
df['rolling_xg'] = xg_series

# CALCULATING ROLLING 5 GAMEWEEK XA (EXPECTED ASSISTS)
xa_series = df.groupby('name')['expected_assists'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
xa_series = xa_series.reset_index(level=0, drop=True)
df['rolling_xa'] = xa_series


imp_players_df = df.copy()

# GETTING FIXTURE DIFFICULTY
team_strength = imp_players_df.groupby('opponent_team')['total_points'].mean().to_dict() # calculating average points conceded by each team when they were the opponents.

# mapping strength rating to each row based on opponent
imp_players_df['fixture_difficulty'] = imp_players_df['opponent_team'].map(team_strength) # higher number = defensively weaker

print("\nVerifying 'form' and 'fixture_difficulty' for a player:")
print(imp_players_df[imp_players_df['name'] == 'Son Heung-min'][['name', 'GW', 'opponent_team', 'total_points', 'form_5_gw', 'fixture_difficulty']].head(7))

# DATA PREPARATION FOR MACHINE LEARNING
features = [
    'form_5_gw',
    'fixture_difficulty',
    'is_home',
    'team_strength',
    'rolling_xg',
    'rolling_xa'
]
target = 'total_points'

# any missing values are filled with 0
imp_players_df = imp_players_df.fillna(0)

X = imp_players_df[features]
y = imp_players_df[target]

# split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData prepared for ML:")
print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows")


# TRAINING XGBOOST MODEL
# initializing XGBoost Regressor model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    objective='reg:squarederror' # Explicitly set the objective
)

print("\nTraining the XGBoost model...")
model.fit(X_train, y_train)
print("Model training complete.")


# MODEL EVALUATION
predictions = model.predict(X_test)

# calculating rmse (Root mean square error)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"\nModel Performance on Test Set:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")

print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# saving to joblib file
print("\nSaving the trained model to a file...")
joblib.dump(model, 'fpl_model.joblib')
print("Model saved as fpl_model.joblib")