import pandas as pd
import numpy as np
import xgboost as xgb
import pulp

print("Starting backtest simulation...")

# --- 1. LOAD AND PREPARE ALL HISTORICAL DATA ---
df_22_23 = pd.read_csv('data/2022-23/merged_gw.csv', on_bad_lines='skip')
df_22_23['season'] = '2022-23'
df_23_24 = pd.read_csv('data/2023-24/merged_gw.csv', on_bad_lines='skip')
df_23_24['season'] = '2023-24'
df_24_25 = pd.read_csv('data/2024-25/merged_gw.csv', on_bad_lines='skip')
df_24_25['season'] = '2024-25'
df = pd.concat([df_22_23, df_23_24, df_24_25])
df = df.sort_values(by=['name', 'season', 'GW'])
print("Data loaded and sorted.")

# --- 2. FEATURE ENGINEERING FOR THE ENTIRE DATASET ---

# *** FIX 1: CREATE THE CORRECT 'cost' COLUMN ***
df['cost'] = df['value'] / 10

# (The rest of the feature engineering is the same)
form_series = df.groupby('name')['total_points'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df['form_5_gw'] = form_series.reset_index(level=0, drop=True)
df['is_home'] = df['was_home'].astype(int)
team_attack = df.groupby('team')['total_points'].mean().to_dict()
df['team_strength'] = df['team'].map(team_attack)
team_defense = df.groupby('opponent_team')['total_points'].mean().to_dict()
df['fixture_difficulty'] = df['opponent_team'].map(team_defense)
xg_series = df.groupby('name')['expected_goals'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df['rolling_xg'] = xg_series.reset_index(level=0, drop=True)
xa_series = df.groupby('name')['expected_assists'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df['rolling_xa'] = xa_series.reset_index(level=0, drop=True)
df = df.fillna(0)
print("Feature engineering complete.")

# --- 3. TRAIN THE MODEL ON THE 2022-23 AND 2023-24 SEASONS ---
train_df = df[df['season'].isin(['2022-23', '2023-24'])]
features = [
    'form_5_gw', 'fixture_difficulty', 'is_home',
    'team_strength', 'rolling_xg', 'rolling_xa'
]
target = 'total_points'
X_train = train_df[features]
y_train = train_df[target]
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    objective='reg:squarederror'
)
print("\nTraining model on 2022-23 and 2023-24 season data...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. RUN THE SIMULATION ON THE 2024-25 SEASON ---
test_df = df[df['season'] == '2024-25'].copy()
total_season_score = 0
print("\nStarting simulation for the 2024-25 season...")

for gw in range(1, 39):
    gw_df = test_df[test_df['GW'] == gw].copy()
    if gw_df.empty:
        print(f"Gameweek {gw}: No data available, skipping.")
        continue
    
    X_predict = gw_df[features]
    gw_df['predicted_points'] = model.predict(X_predict)
    
    prob = pulp.LpProblem(f"FPL_GW_{gw}", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("player", gw_df.index, cat=pulp.LpBinary)
    
    prob += pulp.lpSum(gw_df.loc[i, 'predicted_points'] * player_vars[i] for i in gw_df.index)
    
    # *** FIX 2: UPDATE THE BUDGET CONSTRAINT TO USE THE CORRECT 'cost' COLUMN ***
    prob += pulp.lpSum(gw_df.loc[i, 'cost'] * player_vars[i] for i in gw_df.index) <= 100.0
    
    prob += pulp.lpSum(player_vars[i] for i in gw_df.index) == 15
    prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'GKP') == 2
    prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'DEF') == 5
    prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'MID') == 5
    prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'FWD') == 3
    for team_name in gw_df['team'].unique():
        prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'team'] == team_name) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    gw_score = 0
    selected_player_indices = []
    for i in gw_df.index:
        if player_vars[i].varValue == 1:
            selected_player_indices.append(i)
            
    gw_score = gw_df.loc[selected_player_indices, 'total_points'].sum()
    total_season_score += gw_score
    
    print(f"Gameweek {gw}: Team selected, Actual Score = {gw_score}, Total Score = {total_season_score}")

# --- 5. FINAL RESULTS ---
print("\n--- Backtest Simulation Complete ---")
print(f"Total points scored over the 2024-25 season: {total_season_score}")