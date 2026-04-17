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
            'Expert': [],        # 1600-2000
            'Maitre': []         # 2000+
        }
        # Très important : Chess.com demande un User-Agent avec ton email
        self.headers = {
            'User-Agent': f'ChessProject-Analysis-Student (Contact: {email})'
        }
        self.target_month = (2026, 3)
        self.backup_file = backup_file
        
        # Charger la sauvegarde si elle existe
        self.load_backup()

    def classify_elo(self, rating):
        if rating < 800: return 'Debutant'
        elif rating < 1200: return 'Intermediaire'
        elif rating < 1600: return 'Avance'
        elif rating < 2000: return 'Expert'
        else: return 'Maitre'

    def get_player_stats(self, username):
        """Récupère le rating rapid actuel"""
        try:
            url = f"https://api.chess.com/pub/player/{username}/stats"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                stats = response.json()
                if 'chess_rapid' in stats:
                    return stats['chess_rapid'].get('last', {}).get('rating', None)
            return None
        except:
            return None

    def count_rapid_games(self, username):
        """Vérifie si le joueur a assez de parties en Mars 2026"""
        try:
            year, month = self.target_month
            url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                games = response.json().get('games', [])
                # On filtre rapidement par time_class si l'API le permet, sinon manuellement
                rapid_games = [g for g in games if g.get('time_class') == 'rapid']
                return len(rapid_games)
            return 0
        except:
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

    def collect_by_country(self, country_code='FR', target_per_class=50):
        """Méthode Massive Sampling : pioche dans la liste nationale"""
        print(f"\n🌍 Récupération des joueurs du pays : {country_code}...")
        url = f"https://api.chess.com/pub/country/{country_code}/players"
        
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print("❌ Erreur lors de la récupération de la liste pays.")
            return

        all_usernames = response.json().get('players', [])
        random.shuffle(all_usernames)
        
        print(f"✅ {len(all_usernames)} noms trouvés. Analyse en cours...")

        # Classes à remplir (pas Maitre)
        classes_to_check = ['Debutant', 'Intermediaire', 'Avance', 'Expert']
        
        # Créer la barre de progression
        pbar = tqdm(
            total=sum(target_per_class - len(self.players_by_class[c]) for c in classes_to_check),
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
                        'dernière classe': p_class
                    })
                    
                    # Sauvegarde tous les 5 joueurs
                    if players_added % 5 == 0:
                        self.save_backup()
            
            # Pause pour respecter l'API
            time.sleep(0.1)
        
        pbar.close()
        
        # Sauvegarde finale de cette étape
        self.save_backup()
        print(f"\n📊 {players_added} joueurs ajoutés (sur {players_checked} analysés)")

    def collect_titled_players(self, target_maitre=50):
        """
        Collecte les joueurs titrés (2000+ Elo) via les endpoints spécifiques
        Titres FIDE : GM, WGM, IM, WIM, FM, WFM, NM, WNM, CM, WCM
        """
        # Liste des titres par ordre décroissant de force
        titles = ['GM', 'WGM', 'IM', 'WIM', 'FM', 'WFM', 'NM', 'WNM', 'CM', 'WCM']
        
        print(f"\n👑 Collecte des joueurs titrés pour la classe Maitre (2000+ Elo)...")
        
        # Calculer combien de Maîtres il reste à collecter
        remaining = target_maitre - len(self.players_by_class['Maitre'])
        
        if remaining <= 0:
            print("✅ Classe Maitre déjà complète !")
            return
        
        # Barre de progression pour les Maîtres
        pbar = tqdm(total=remaining, desc="Collecte titrés", unit="maître")
        
        players_added = 0
        
        for title in titles:
            # Arrêt si on a assez de Maîtres
            if len(self.players_by_class['Maitre']) >= target_maitre:
                print(f"\n✅ Classe Maitre complète ({target_maitre} joueurs) !")
                break
            
            print(f"\n🏆 Recherche des {title}...")
            url = f"https://api.chess.com/pub/titled/{title}"
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    print(f"  ⚠️  Impossible de récupérer la liste des {title}")
                    time.sleep(1)
                    continue
                
                titled_players = response.json().get('players', [])
                random.shuffle(titled_players)
                
                print(f"  📋 {len(titled_players)} {title} trouvés")
                
                # Sous-barre pour chaque titre
                title_pbar = tqdm(titled_players, desc=f"  Analyse {title}", leave=False)
                
                for username in title_pbar:
                    # Arrêt si classe complète
                    if len(self.players_by_class['Maitre']) >= target_maitre:
                        break
                    
                    if username in self.collected_players:
                        continue
                    
                    # Récupérer le rating rapid
                    rating = self.get_player_stats(username)
                    
                    # Vérifier que c'est bien un Maître (2000+)
                    if not rating or rating < 2000:
                        continue
                    
                    # Vérifier le nombre de parties en mars 2026
                    nb_games = self.count_rapid_games(username)
                    
                    if nb_games >= 20:
                        self.players_by_class['Maitre'].append({
                            'username': username,
                            'rating': rating,
                            'nb_games': nb_games,
                            'title': title
                        })
                        self.collected_players.add(username)
                        players_added += 1
                        pbar.update(1)
                        pbar.set_postfix({'titre': title, 'elo': rating})
                        
                        # Sauvegarde tous les 3 maîtres
                        if players_added % 3 == 0:
                            self.save_backup()
                    
                    # Pause pour l'API
                    time.sleep(0.1)
                
                title_pbar.close()
                
                # Pause entre chaque titre
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Erreur lors de la collecte des {title}: {e}")
                continue
        
        pbar.close()
        
        # Sauvegarde finale
        self.save_backup()
        print(f"\n📊 {players_added} maîtres ajoutés")

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
        
        print("🎯 ÉTAPE 1 : Collecte des joueurs amateurs (< 2000 Elo)")
        print("-" * 60)
        # 1. On commence par le pays pour remplir les classes Débutant à Expert
        collector.collect_by_country(country_code='FR', target_per_class=25)
        
        print("\n\n🎯 ÉTAPE 2 : Collecte des joueurs titrés (2000+ Elo)")
        print("-" * 60)
        # 2. Ensuite on cible les joueurs titrés pour la classe Maitre
        collector.collect_titled_players(target_maitre=25)
        
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