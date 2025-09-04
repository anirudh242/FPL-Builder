import requests
import pandas as pd
import numpy as np
import pulp
import xgboost as xgb

# TRAINING MODEL ON PAST SEASONS
print("Loading historical data from past seasons to train model...")
df_23_24 = pd.read_csv('data/2023-24/merged_gw.csv', on_bad_lines='skip')
df_23_24['season'] = '2023-24'
df_24_25 = pd.read_csv('data/2024-25/merged_gw.csv', on_bad_lines='skip')
df_24_25['season'] = '2024-25'
train_df = pd.concat([df_23_24, df_24_25])
train_df = train_df.sort_values(by=['name', 'season', 'GW'])

# feature engineering for historical data
team_attack = train_df.groupby('team')['total_points'].mean().to_dict()
team_defense = train_df.groupby('opponent_team')['total_points'].mean().to_dict()
train_df['cost'] = train_df['value'] / 10
form_series = train_df.groupby('name')['total_points'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
train_df['form_5_gw'] = form_series.reset_index(level=0, drop=True)
train_df['is_home'] = train_df['was_home'].astype(int)
train_df['team_strength'] = train_df['team'].map(team_attack)
train_df['fixture_difficulty'] = train_df['opponent_team'].map(team_defense)
xg_series = train_df.groupby('name')['expected_goals'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
train_df['rolling_xg'] = xg_series.reset_index(level=0, drop=True)
xa_series = train_df.groupby('name')['expected_assists'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
train_df['rolling_xa'] = xa_series.reset_index(level=0, drop=True)
train_df = train_df.fillna(0)

# training the model
features = ['form_5_gw', 'fixture_difficulty', 'is_home', 'team_strength', 'rolling_xg', 'rolling_xa']
target = 'total_points'
X_train = train_df[features]
y_train = train_df[target]
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, objective='reg:squarederror')
print("Training ML model on all past seasons...")
model.fit(X_train, y_train)
print("Model training complete.")


FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(FPL_API_URL)
print("\nRESPONSE STATUS: " + str(response.status_code))
data = response.json()
players_df = pd.DataFrame(data['elements'])

# feature engineering for live data
teams_map = {team['id']: team['name'] for team in data['teams']}
positions_map = {pos['id']: pos['singular_name_short'] for pos in data['element_types']}
players_df['team'] = players_df['team'].map(teams_map)
players_df['position'] = players_df['element_type'].map(positions_map)
players_df['cost'] = players_df['now_cost'] / 10
players_df['form_5_gw'] = players_df['form'].astype(float)
players_df['rolling_xg'] = players_df['expected_goals_per_90'].astype(float)
players_df['rolling_xa'] = players_df['expected_assists_per_90'].astype(float)
players_df['team_strength'] = players_df['team'].map(team_attack)
players_df['fixture_difficulty'] = np.mean(list(team_defense.values()))
players_df['is_home'] = 1
players_df = players_df.fillna(0)
print("Live data features engineered robustly.")


# OPTIMIZATION + PREDICtIONS
X_predict = players_df[features]
players_df['predicted_points'] = model.predict(X_predict)
print("Point predictions made for all players.")

final_df = players_df.copy()

prob = pulp.LpProblem("FPL_Team_Selection_ML", pulp.LpMaximize)
player_vars = pulp.LpVariable.dicts("player", final_df.index, cat=pulp.LpBinary)

prob += pulp.lpSum(final_df.loc[i, 'predicted_points'] * player_vars[i] for i in final_df.index)

# CONSTRAINTS
prob += pulp.lpSum(final_df.loc[i, 'cost'] * player_vars[i] for i in final_df.index) <= 100.0
prob += pulp.lpSum(player_vars[i] for i in final_df.index) == 15
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'GKP') == 2
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'DEF') == 5
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'MID') == 5
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'FWD') == 3
for team_name in final_df['team'].unique():
    prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'team'] == team_name) <= 3

prob.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\nSolver status: {pulp.LpStatus[prob.status]}")

#DISPLAY
print(f'\n   Optimal FPL Team (ML Predictions)   ')
total_cost = 0
total_predicted_points = 0
positions_order = ['GKP', 'DEF', 'MID', 'FWD']
for position in positions_order:
    for i in final_df.index:
        if player_vars[i].varValue == 1 and final_df.loc[i, 'position'] == position:
            player_info = final_df.loc[i]
            print(f"{player_info['position']:<4} {player_info['web_name']:<15} {player_info['team']:<15} £{player_info['cost']:.1f}m \t Predicted Points: {player_info['predicted_points']:.2f}")
            total_cost += player_info['cost']
            total_predicted_points += player_info['predicted_points']
print(f"\nTotal Cost: £{total_cost:.1f}m")
print(f"Total Predicted Points: {total_predicted_points:.2f}")