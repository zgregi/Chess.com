import requests
import pandas as pd
import time
import random
import json
from tqdm import tqdm
from datetime import datetime


class ChessDataCollector:
    def __init__(self, email, backup_file='backup_progress.json'):
        self.collected_players = set()
        self.players_by_class = {
            'Debutant': [],      # < 800
            'Intermediaire': [],  # 800-1200
            'Avance': [],        # 1200-1600
            'Expert': [],        # 1600-2200
            'Maitre': []         # 2200+
        }
        # Très important : Chess.com demande un User-Agent avec ton email
        self.headers = {
            'User-Agent': f'ChessProject-Analysis-Student (Contact: {email})'
        }
        self.target_month = (2026, 3)  # ⚠️ Mars 2026 n'existe pas encore, changé en 2026
        self.backup_file = backup_file
        
        # Charger la sauvegarde si elle existe
        self.load_backup()

    def classify_elo(self, rating):
        if rating < 800: return 'Debutant'
        elif rating < 1200: return 'Intermediaire'
        elif rating < 1600: return 'Avance'
        elif rating < 2200: return 'Expert'
        else: return 'Maitre'

    def get_player_stats(self, username):
        """Récupère le rating rapid actuel"""
        try:
            url = f"https://api.chess.com/pub/player/{username}/stats"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                stats = response.json()
                if 'chess_rapid' in stats:
                    return stats['chess_rapid'].get('last', {}).get('rating', None)
            return None
        except Exception as e:
            return None

    def count_rapid_games(self, username):
        """Vérifie si le joueur a assez de parties en Mars 2026"""
        try:
            year, month = self.target_month
            url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                games = response.json().get('games', [])
                # ✅ CORRECTION: games au lieu de all_games
                rapid_games = [g for g in games 
                    if g.get('time_class') == 'rapid'
                ]
                return len(rapid_games)
            return 0
        except Exception as e:
            return 0

    def save_backup(self):
        """Sauvegarde intermédiaire pour ne pas perdre les données"""
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'collected_players': list(self.collected_players),
            'players_by_class': self.players_by_class
        }
        
        with open(self.backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2)

    def load_backup(self):
        """Charge la sauvegarde précédente si elle existe"""
        try:
            with open(self.backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
                self.collected_players = set(backup_data['collected_players'])
                self.players_by_class = backup_data['players_by_class']
                print(f"✅ Sauvegarde chargée ({len(self.collected_players)} joueurs déjà collectés)")
        except FileNotFoundError:
            print("ℹ️  Pas de sauvegarde trouvée, démarrage à zéro")
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de la sauvegarde: {e}")

    def collect_by_country(self, country_code='FR', target_per_class=40):
        """Méthode Massive Sampling : pioche dans la liste nationale"""
        print(f"\n🌍 Récupération des joueurs du pays : {country_code}...")
        url = f"https://api.chess.com/pub/country/{country_code}/players"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code != 200:
                print(f"❌ Erreur HTTP {response.status_code} lors de la récupération de la liste pays.")
                return
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            return

        all_usernames = response.json().get('players', [])
        random.shuffle(all_usernames)
        
        print(f"✅ {len(all_usernames)} noms trouvés. Analyse en cours...")

        # Classes à remplir (pas Maitre)
        classes_to_check = ['Debutant', 'Intermediaire', 'Avance', 'Expert']
        
        # Calculer le total nécessaire
        total_needed = sum(max(0, target_per_class - len(self.players_by_class[c])) for c in classes_to_check)
        
        if total_needed == 0:
            print("✅ Toutes les classes sont déjà complètes !")
            return
        
        # Créer la barre de progression
        pbar = tqdm(
            total=total_needed,
            desc="Collecte pays",
            unit="joueur"
        )
        
        players_checked = 0
        players_added = 0

        for username in all_usernames:
            # Arrêt si tout est plein (sauf Maitre, traité séparément)
            if all(len(self.players_by_class[c]) >= target_per_class for c in classes_to_check):
                print("\n✅ Classes Débutant à Expert complètes !")
                break

            if username in self.collected_players:
                continue

            players_checked += 1
            
            rating = self.get_player_stats(username)
            if not rating:
                continue

            p_class = self.classify_elo(rating)
            
            # On skip les Maîtres ici, ils seront traités par collect_titled_players
            if p_class == 'Maitre':
                continue

            # Si on a encore besoin de cette classe
            if len(self.players_by_class[p_class]) < target_per_class:
                nb_games = self.count_rapid_games(username)
                
                if nb_games >= 20:
                    self.players_by_class[p_class].append({
                        'username': username,
                        'rating': rating,
                        'nb_games': nb_games
                    })
                    self.collected_players.add(username)
                    players_added += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'ajoutés': players_added,
                        'classe': p_class[:3]
                    })
                    
                    # Sauvegarde tous les 5 joueurs
                    if players_added % 5 == 0:
                        self.save_backup()
            
            # Pause pour respecter l'API
            time.sleep(0.15)  # Augmenté à 0.15s pour éviter rate limiting
        
        pbar.close()
        
        # Sauvegarde finale de cette étape
        self.save_backup()
        print(f"\n📊 {players_added} joueurs ajoutés (sur {players_checked} analysés)")

    def collect_titled_players(self, target_maitre=40):
            """
            Collecte les joueurs titrés (2200+ Elo) avec une limite par titre
            pour garantir la diversité (évite d'avoir 40 joueurs du même titre).
            """
            # Liste des titres ciblés (ordre inversé pour varier les profils)
            titles = ['IM', 'FM', 'NM', 'GM']
            limit_per_title = 10  # On s'arrête à 10 par titre
            
            print(f"\n👑 Collecte diversifiée des Maîtres (Objectif total : {target_maitre})")
            
            # Calculer combien de Maîtres il reste à collecter au total
            remaining_total = target_maitre - len(self.players_by_class['Maitre'])
            
            if remaining_total <= 0:
                print("✅ Classe Maitre déjà complète !")
                return
            
            # Barre de progression globale
            pbar = tqdm(total=remaining_total, desc="Progression totale Maîtres", unit="joueur")
            players_added = 0
            
            for title in titles:
                # Si l'objectif total de 40 est atteint, on arrête tout
                if len(self.players_by_class['Maitre']) >= target_maitre:
                    break
                
                # Compteur spécifique pour le titre en cours
                count_for_this_title = 0
                
                print(f"\n🏆 Recherche des {title} (Quota max : {limit_per_title})...")
                url = f"https://api.chess.com/pub/titled/{title}"
                
                try:
                    response = requests.get(url, headers=self.headers, timeout=30)
                    if response.status_code != 200:
                        print(f"  ⚠️ Erreur HTTP {response.status_code} pour les {title}")
                        continue
                    
                    titled_players = response.json().get('players', [])
                    random.shuffle(titled_players)
                    
                    # Sous-barre pour analyser les joueurs de ce titre
                    title_pbar = tqdm(titled_players, desc=f"  Analyse {title}", leave=False)
                    
                    for username in title_pbar:
                        # On s'arrête si :
                        # 1. Le quota total (40) est atteint
                        # 2. OU le quota pour ce titre (10) est atteint
                        if len(self.players_by_class['Maitre']) >= target_maitre or count_for_this_title >= limit_per_title:
                            break
                        
                        if username in self.collected_players:
                            continue
                        
                        # 1. Vérification du Rating
                        rating = self.get_player_stats(username)
                        if not rating or rating < 2200:
                            continue
                        
                        # 2. Vérification de l'activité (Mars 2025)
                        nb_games = self.count_rapid_games(username)
                        
                        if nb_games >= 20: 
                            # Ajout du joueur
                            self.players_by_class['Maitre'].append({
                                'username': username,
                                'rating': rating,
                                'nb_games': nb_games,
                                'title': title,
                                'class': 'Maitre'
                            })
                            self.collected_players.add(username)
                            
                            players_added += 1
                            count_for_this_title += 1 # On incrémente le compteur du titre
                            
                            pbar.update(1)
                            pbar.set_postfix({'titre': title, 'count': f"{count_for_this_title}/{limit_per_title}"})
                            
                            # Sauvegarde régulière (tous les 3 joueurs)
                            if players_added % 3 == 0:
                                self.save_backup()
                        
                        # Respect du rate-limiting de l'API
                        time.sleep(0.15)
                    
                    title_pbar.close()
                    time.sleep(0.5) # Pause entre deux catégories de titres
                    
                except Exception as e:
                    print(f"  ❌ Erreur critique sur {title}: {e}")
                    continue
            
            pbar.close()
            self.save_backup() # Sauvegarde finale
            print(f"\n📊 Fin de collecte : {players_added} nouveaux maîtres ajoutés.")

    def print_summary(self):
        """Affiche un résumé de la collecte"""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA COLLECTE")
        print("="*60)
        total = 0
        
        # Créer un DataFrame pour l'affichage
        summary_data = []
        
        for class_name, players in self.players_by_class.items():
            count = len(players)
            total += count
            
            if count > 0:
                ratings = [p['rating'] for p in players]
                avg_rating = sum(ratings) / len(ratings)
                min_rating = min(ratings)
                max_rating = max(ratings)
                
                summary_data.append({
                    'Classe': class_name,
                    'Joueurs': count,
                    'Elo moyen': f"{avg_rating:.0f}",
                    'Elo min': min_rating,
                    'Elo max': max_rating
                })
            else:
                summary_data.append({
                    'Classe': class_name,
                    'Joueurs': count,
                    'Elo moyen': '-',
                    'Elo min': '-',
                    'Elo max': '-'
                })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        print("\n" + "="*60)
        print(f"TOTAL: {total} joueurs collectés")
        print("="*60)

    def save_to_csv(self, filename='players_dataset.csv'):
        data = []
        for c_name, players in self.players_by_class.items():
            for p in players:
                row = {
                    'username': p['username'], 
                    'rating': p['rating'], 
                    'class': c_name,
                    'nb_games': p['nb_games']
                }
                # Ajouter le titre si disponible (uniquement pour Maîtres)
                if 'title' in p:
                    row['title'] = p['title']
                else:
                    row['title'] = None
                    
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n💾 Sauvegardé dans {filename}")
        print(f"📈 Total: {len(df)} joueurs")
        
        # Supprimer le fichier de backup après sauvegarde finale
        try:
            import os
            if os.path.exists(self.backup_file):
                os.remove(self.backup_file)
                print(f"🗑️  Backup supprimé ({self.backup_file})")
        except:
            pass
        
        return df


# 🚀 EXECUTION
if __name__ == "__main__":
    try:
        # Remplace par ton vrai mail
        collector = ChessDataCollector(
            email="gregoire.weber@eleve.ensai.fr",
            backup_file="backup_chess_collection.json"
        )
        
        print("🎯 ÉTAPE 1 : Collecte des joueurs amateurs (< 2200 Elo)")
        print("-" * 60)
        # 1. On commence par le pays pour remplir les classes Débutant à Expert
        collector.collect_by_country(country_code='FR', target_per_class=40)
        
        print("\n\n🎯 ÉTAPE 2 : Collecte des joueurs titrés (2200+ Elo)")
        print("-" * 60)
        # 2. Ensuite on cible les joueurs titrés pour la classe Maitre
        collector.collect_titled_players(target_maitre=40)
        
        # 3. Afficher le résumé
        collector.print_summary()
        
        # 4. Sauvegarder
        df = collector.save_to_csv('players_mars_2026.csv')
        
        print("\n✅ Collecte terminée avec succès !")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption détectée (Ctrl+C)")
        print("💾 Sauvegarde en cours...")
        collector.save_backup()
        collector.print_summary()
        print("✅ Progrès sauvegardé ! Relancez le script pour reprendre.")
    
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        print("💾 Tentative de sauvegarde d'urgence...")
        try:
            collector.save_backup()
            print("✅ Sauvegarde d'urgence réussie")
        except:
            print("❌ Impossible de sauvegarder")
