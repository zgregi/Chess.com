import pandas as pd

# 1. Charger ton backup actuel
df = pd.read_csv('players_mars_2026.csv')

# 2. Identifier les GMs en trop
# On garde tous ceux qui ne sont pas Maîtres OU qui ne sont pas GMs
df_fix = df[~((df['class'] == 'Maitre') & (df['title'] == 'GM'))]

# 3. On ne reprend que les 10 premiers GMs
df_gms_limited = df[(df['class'] == 'Maitre') & (df['title'] == 'GM')].head(10)

# 4. On fusionne et on écrase le backup
df_final = pd.concat([df_fix, df_gms_limited])
df_final.to_csv('players_mars_2026.csv', index=False)

print(f"Fait ! Il ne reste plus que {len(df_gms_limited)} GMs dans ton fichier.")