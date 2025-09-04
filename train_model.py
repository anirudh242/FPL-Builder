import pandas as pd

df = pd.read_csv('merged_gw.csv', on_bad_lines='skip')

# sory by name then gameweek
df = df.sort_values(by=['name', 'GW'])



# calculate rolling 5 gameweek form
form_series = df.groupby('name')['total_points'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)

form_series = form_series.reset_index(level=0, drop=True)
df['form_5_gw'] = form_series

# GETTING FIXTURE DIFFICULTY
team_strength = df.groupby('opponent_team')['total_points'].mean().to_dict() # calculating average points conceded by each team when they were the opponents.

# mapping strength rating to each row based on opponent
df['fixture_difficulty'] = df['opponent_team'].map(team_strength) # higher number = defensively weaker


print("\nVerifying 'form' and 'fixture_difficulty' for Son:")
print(df[df['name'] == 'Son Heung-min'][['name', 'GW', 'opponent_team', 'total_points', 'form_5_gw', 'fixture_difficulty']].head(7))
