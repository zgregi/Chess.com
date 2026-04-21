import pandas as pd
import requests
import chess.pgn
import io
import re
import time

# --- Configuration ---
CSV_INPUT = 'data/players_mars_2026.csv'
PARQUET_OUTPUT = 'data/stat_parties.parquet'
HEADERS = {'User-Agent': 'ChessMLProject - gregoire.weber@eleve.ensai.fr'}


def analyze_player_march(username):
    url = f"https://api.chess.com/pub/player/{username}/games/2026/03"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None
        
        games = response.json().get('games', [])
        if not games:
            return None

        game_data = []
        
        for game in games:
            if game.get('time_class') != 'rapid':
                continue
                
            pgn_text = game.get('pgn', '')
            if not pgn_text:
                continue

            pgn_io = io.StringIO(pgn_text)
            parsed_game = chess.pgn.read_game(pgn_io)
            if not parsed_game:
                continue

            # 1. Ouverture et Résultat
            eco = parsed_game.headers.get("ECO", "Unknown")
            res = parsed_game.headers.get("Result", "*")
            white_user = parsed_game.headers.get("White", "").lower()
            is_white = white_user == username.lower()
            
            # Déterminer la victoire
            win = 1 if (is_white and res == "1-0") or (not is_white and res == "0-1") else 0
            
            # 2. Temps moyen par coup (Initialisation par défaut)
            avg_time = 0
            times = []
            node = parsed_game
            moves_count = 0
            
            while node.variations and moves_count < 20:
                next_node = node.variation(0)
                is_player_turn = (moves_count % 2 == 0) if is_white else (moves_count % 2 != 0)
                
                if is_player_turn:
                    match = re.search(r"\[%clk (\d+):(\d+):(\d+\.?\d*)\]", next_node.comment)
                    if match:
                        h, m, s = map(float, match.groups())
                        times.append(h * 3600 + m * 60 + s)
                node = next_node
                moves_count += 1
            
            if len(times) > 1:
                diffs = [times[i-1] - times[i] for i in range(1, len(times)) if times[i-1] >= times[i]]
                if diffs:
                    avg_time = sum(diffs) / len(diffs)

            game_data.append({'eco': eco, 'avg_time': avg_time, 'win': win})

        if not game_data:
            return None

        # --- Calcul des stats pour le Parquet ---
        df_temp = pd.DataFrame(game_data)
        
        # Top 5 Utilisées
        top_5_used = df_temp['eco'].value_counts().head(5).index.tolist()
        
        # Top 5 Gagnantes (min 1 partie jouée)
        top_5_win = df_temp.groupby('eco')['win'].mean().sort_values(ascending=False).head(5).index.tolist()
        
        return {
            'avg_time': df_temp['avg_time'].mean(),
            'top_used': ", ".join(top_5_used),
            'top_win': ", ".join(top_5_win),
            'nb_games': len(df_temp)
        }

    except Exception as e:
        print(f"\nErreur pour {username}: {e}")
        return None


# --- Exécution ---
df_players = pd.read_csv(CSV_INPUT)
results = []

for index, row in df_players.iterrows():
    print(f"[{index+1}/{len(df_players)}] Analyse de {row['username']}...", end="\r")
    stats = analyze_player_march(row['username'])
    
    if stats:
        results.append({
            'username': row['username'],
            'class': row['class'],
            'rating': row['rating'],
            'avg_opening_move_time': round(stats['avg_time'], 2),
            'top_5_openings_used': stats['top_used'],
            'top_5_openings_win': stats['top_win'],
            'nb_rapid_games': stats['nb_games']
        })
    time.sleep(0.2)

pd.DataFrame(results).to_parquet(PARQUET_OUTPUT, index=False)
print(f"\n✅ Terminé ! Fichier {PARQUET_OUTPUT} créé.")