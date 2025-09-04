import requests
import pandas as pd
import numpy as np
import pulp
import joblib 

# loading the saved machine learning model
model = joblib.load('fpl_model.joblib')

# loading historical data to calculate team strengths
historical_df = pd.read_csv('merged_gw.csv', on_bad_lines='skip')

FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'

response = requests.get(FPL_API_URL)
print("RESPONSE STATUS: " + str(response.status_code))

data = response.json()

# dataframe for players
players_df = pd.DataFrame(data['elements'])


# important columns to keep from original
imp_columns = ['web_name', 'team', 'element_type', 'now_cost', 'total_points', 'form', 'expected_goals_per_90', 'expected_assists_per_90']
imp_players_df = players_df[imp_columns]


teams_map = {team['id']: team['name'] for team in data['teams']}
positions_map = {pos['id']: pos['singular_name_short'] for pos in data['element_types']}

# replacing numbers with actual names
imp_players_df['team'] = imp_players_df['team'].map(teams_map)
imp_players_df['element_type'] = imp_players_df['element_type'].map(positions_map)

imp_players_df = imp_players_df.rename(columns={'element_type': 'position'})

imp_players_df['cost'] = imp_players_df['now_cost'] / 10

# creating features that match the trained model
# calculating team attacking strength from historical data
team_attack = historical_df.groupby('team')['total_points'].mean().to_dict()
imp_players_df['team_strength'] = imp_players_df['team'].map(team_attack)

# calculating fixture difficulty from historical data
team_defense = historical_df.groupby('opponent_team')['total_points'].mean().to_dict()
imp_players_df['fixture_difficulty'] = np.mean(list(team_defense.values()))

# creating other features using live api data as proxies
imp_players_df['form_5_gw'] = imp_players_df['form'].astype(float)
imp_players_df['is_home'] = 1 # assuming a home game for simplicity
imp_players_df['rolling_xg'] = imp_players_df['expected_goals_per_90'].astype(float)
imp_players_df['rolling_xa'] = imp_players_df['expected_assists_per_90'].astype(float)

# making sure all feature columns have no missing values
imp_players_df = imp_players_df.fillna(0)


# --- MAKING PREDICTIONS WITH THE MODEL ---
features = [
    'form_5_gw',
    'fixture_difficulty',
    'is_home',
    'team_strength',
    'rolling_xg',
    'rolling_xa'
]
X_predict = imp_players_df[features]

# using the model to predict points for all current players
imp_players_df['predicted_points'] = model.predict(X_predict)

final_df = imp_players_df.copy()

# pulp linear programming problem
prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

player_vars = pulp.LpVariable.dicts("player", final_df.index, cat=pulp.LpBinary)

# maximise predicted_points of selected players
prob += pulp.lpSum(final_df.loc[i, 'predicted_points'] * player_vars[i] for i in final_df.index)

# CONSTRAINTS
prob += pulp.lpSum(final_df.loc[i, 'cost'] * player_vars[i] for i in final_df.index) <= 100.0 # budget constraint
prob += pulp.lpSum(player_vars[i] for i in final_df.index) == 15 # squad size constraint
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'GKP') == 2 # goalkeeper constraint
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'DEF') == 5 # defender constraint
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'MID') == 5 # midfielder constraint
prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'position'] == 'FWD') == 3 # attacker constraint

for team_name in final_df['team'].unique():
    prob += pulp.lpSum(player_vars[i] for i in final_df.index if final_df.loc[i, 'team'] == team_name) <= 3 # maximum 3 players from one team

prob.solve()
print(f"\nSolver status: {pulp.LpStatus[prob.status]}")

# display result
print(f'\n   Optimal FPL Team (ML Predictions)   ')
total_cost = 0
total_predicted_points = 0

# to print out in order
positions_order = ['GKP', 'DEF', 'MID', 'FWD']

for position in positions_order:
    for i in final_df.index:
        if player_vars[i].varValue == 1 and final_df.loc[i, 'position'] == position: # if player was chosen the value is 1 else 0
            player_info = final_df.loc[i]
            print(f"{player_info['position']:<4} {player_info['web_name']:<15} {player_info['team']:<15} £{player_info['cost']:.1f}m \t Predicted Points: {player_info['predicted_points']:.2f}")
            total_cost += player_info['cost']
            total_predicted_points += player_info['predicted_points']

print(f"\nTotal Cost: £{total_cost:.1f}m")
print(f"Total Predicted Points: {total_predicted_points:.2f}")