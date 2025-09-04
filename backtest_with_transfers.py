import pandas as pd
import numpy as np
import xgboost as xgb
import pulp

print("Starting realistic backtest simulation with weekly transfers...")

# GETTING HISTORICAL DATA
df_22_23 = pd.read_csv('data/2022-23/merged_gw.csv', on_bad_lines='skip')
df_22_23['season'] = '2022-23'
df_23_24 = pd.read_csv('data/2023-24/merged_gw.csv', on_bad_lines='skip')
df_23_24['season'] = '2023-24'
df = pd.concat([df_22_23, df_23_24])
df = df.sort_values(by=['name', 'season', 'GW'])
print("Data loaded.")

df['cost'] = df['value'] / 10
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

# MODEL TRAINING
train_df = df[df['season'] == '2022-23']
features = ['form_5_gw', 'fixture_difficulty', 'is_home', 'team_strength', 'rolling_xg', 'rolling_xa']
target = 'total_points'
X_train = train_df[features]
y_train = train_df[target]
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, objective='reg:squarederror')
print("\nTraining model on 2022-23 season data...")
model.fit(X_train, y_train)
print("Model training complete.")

def pick_starting_xi(team_df):
    team_df = team_df.sort_values('predicted_points', ascending=False)
    starters = pd.DataFrame()
    formation = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    captain_id = team_df.iloc[0]['name']
    for i, player in team_df.iterrows():
        pos = player['position']
        if len(starters) < 11:
            if pos == 'GKP' and formation['GKP'] < 1:
                starters = pd.concat([starters, player.to_frame().T])
                formation['GKP'] += 1
            elif pos == 'DEF' and formation['DEF'] < 5:
                starters = pd.concat([starters, player.to_frame().T])
                formation['DEF'] += 1
            elif pos == 'MID' and formation['MID'] < 5:
                starters = pd.concat([starters, player.to_frame().T])
                formation['MID'] += 1
            elif pos == 'FWD' and formation['FWD'] < 3:
                starters = pd.concat([starters, player.to_frame().T])
                formation['FWD'] += 1
    return starters, captain_id

# BACKTESTING SIMULATION
test_df = df[df['season'] == '2023-24'].copy()
my_team_players = pd.DataFrame()
money_in_bank = 100.0
total_season_score = 0
TRANSFER_THRESHOLD = 0.5 # set a  minimum threshold in points increase for making transfers
print(f"\nStarting simulation for the 2023-24 season with 1 free transfer/week (Threshold: {TRANSFER_THRESHOLD} pts)...")

for gw in range(1, 39):
    gw_df = test_df[test_df['GW'] == gw].copy()
    if gw_df.empty: continue
    
    gw_df['predicted_points'] = model.predict(gw_df[features])
    
    print(f"--- Gameweek {gw} ---")

    if gw == 1:
        prob = pulp.LpProblem(f"FPL_GW_{gw}", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("player", gw_df.index, cat=pulp.LpBinary)
        prob += pulp.lpSum(gw_df.loc[i, 'predicted_points'] * player_vars[i] for i in gw_df.index)
        prob += pulp.lpSum(gw_df.loc[i, 'cost'] * player_vars[i] for i in gw_df.index) <= 100.0
        prob += pulp.lpSum(player_vars[i] for i in gw_df.index) == 15
        prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'GKP') == 2
        prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'DEF') == 5
        prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'MID') == 5
        prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'position'] == 'FWD') == 3
        for team_name in gw_df['team'].unique():
            prob += pulp.lpSum(player_vars[i] for i in gw_df.index if gw_df.loc[i, 'team'] == team_name) <= 3
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected_indices = [i for i in gw_df.index if player_vars[i].varValue == 1]
        my_team_players = gw_df.loc[selected_indices].copy()
        money_in_bank = 100.0 - my_team_players['cost'].sum()
        print("Initial team selected.")

    else:
        current_team_predicted_points = my_team_players.merge(gw_df[['name', 'predicted_points']], on='name', suffixes=('', '_new'))
        player_to_sell = current_team_predicted_points.sort_values('predicted_points_new').iloc[0]
        budget = money_in_bank + player_to_sell['cost']
        candidates = gw_df[~gw_df['name'].isin(my_team_players['name'])]
        possible_replacements = candidates[(candidates['position'] == player_to_sell['position']) & (candidates['cost'] <= budget)]
        
        if not possible_replacements.empty:
            player_to_buy = possible_replacements.sort_values('predicted_points', ascending=False).iloc[0]
            
            # checking if upgrade meets the threshold
            if (player_to_buy['predicted_points'] - player_to_sell['predicted_points_new']) > TRANSFER_THRESHOLD:
                print(f"   Transfer OUT: {player_to_sell['name']} ({player_to_sell['predicted_points_new']:.2f} pts)")
                print(f"   Transfer IN:  {player_to_buy['name']} ({player_to_buy['predicted_points']:.2f} pts)")
                my_team_players = my_team_players[my_team_players['name'] != player_to_sell['name']]
                my_team_players = pd.concat([my_team_players, player_to_buy.to_frame().T])
                money_in_bank += (player_to_sell['cost'] - player_to_buy['cost'])
            else:
                print(f"   Best transfer option not good enough. Transfer Rolled.")

    team_for_gw = my_team_players.merge(gw_df[['name', 'predicted_points']], on='name', suffixes=('', '_new'))
    starters, captain_name = pick_starting_xi(team_for_gw)
    starters_with_actual_points = starters.merge(gw_df[['name', 'total_points']], on='name', suffixes=('_team', '_actual'))
    gw_score = starters_with_actual_points['total_points_actual'].sum()
    captain_actual_points = starters_with_actual_points[starters_with_actual_points['name'] == captain_name]['total_points_actual'].iloc[0]
    gw_score += captain_actual_points
    total_season_score += gw_score
    print(f"   Score for GW: {gw_score}, Total Score: {total_season_score}\n")

# RESULTs
print("\n--- Realistic Backtest Simulation Complete ---")
print(f"Total points scored over the 2023-24 season with 1 transfer/week: {total_season_score}")