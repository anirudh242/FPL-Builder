import requests
import pandas as pd

FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'

response = requests.get(FPL_API_URL)
print("RESPONSE STATUS: " + str(response.status_code))

data = response.json()

# dataframe for players
players_df = pd.DataFrame(data['elements'])

imp_columns = ['web_name', 'team', 'element_type', 'now_cost', 'total_points']
imp_players_df = players_df[imp_columns]

print(imp_players_df.head())
