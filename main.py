import requests
import pandas as pd

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

print("\nDataFrame with readable names:")
print(imp_players_df.head())