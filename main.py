import requests
import pandas as pd
import numpy as np
import pulp

FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'

response = requests.get(FPL_API_URL)
print("RESPONSE STATUS: " + str(response.status_code))

data = response.json()

# dataframe for players
players_df = pd.DataFrame(data['elements'])

# important columns to keep from original
imp_columns = ['web_name', 'team', 'element_type', 'now_cost', 'total_points']
imp_players_df = players_df[imp_columns]


teams_map = {team['id']: team['name'] for team in data['teams']}
positions_map = {pos['id']: pos['singular_name_short'] for pos in data['element_types']}

# replacing numbers with actual names
imp_players_df['team'] = imp_players_df['team'].map(teams_map)
imp_players_df['element_type'] = imp_players_df['element_type'].map(positions_map)

imp_players_df = imp_players_df.rename(columns={'element_type': 'position'})

imp_players_df['cost'] = imp_players_df['now_cost'] / 10
imp_players_df['ppm'] = imp_players_df['total_points'] / imp_players_df['cost']
imp_players_df['ppm'] = imp_players_df['ppm'].replace([np.inf, -np.inf], 0).fillna(0)

# making sure no infinite or NaN values
imp_players_df['ppm'] = imp_players_df['ppm'].replace([np.inf, -np.inf], 0).fillna(0)

final_df = imp_players_df.copy()

# pulp linear programming problem
prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

player_vars = pulp.LpVariable.dicts("player", final_df.index, cat=pulp.LpBinary)

# maximise ppm of selected players
prob += pulp.lpSum(final_df.loc[i, 'ppm'] * player_vars[i] for i in final_df.index)

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
print(f'\n   Optimal FPL Team   ')
total_cost = 0
total_ppm = 0

# to print out in order
positions_order = ['GKP', 'DEF', 'MID', 'FWD']

for position in positions_order:
    for i in final_df.index:
        if player_vars[i].varValue == 1 and final_df.loc[i, 'position'] == position: # if player was chosen the value is 1 else 0
            player_info = final_df.loc[i]
            print(f"{player_info['position']:<4} {player_info['web_name']:<15} {player_info['team']:<15} £{player_info['cost']:.1f}m \t PPM: {player_info['ppm']:.2f}")
            total_cost += player_info['cost']
            total_ppm += player_info['ppm']

print(f"\nTotal Cost: £{total_cost:.1f}m")
print(f"Total PPM Score: {total_ppm:.2f}")

# print("\nDataFrame with readable names:")
# print(imp_players_df.head())
