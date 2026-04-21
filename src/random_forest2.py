import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json


def map_eco_to_name(eco):
        if not isinstance(eco, str) or len(eco) < 3:
            return "Unknown"
        
        # Extraction de la lettre et du nombre
        letter = eco[0].upper()
        try:
            number = int(eco[1:3])
        except ValueError:
            return "Unknown"

        # LOGIQUE DE REGROUPEMENT
        if letter == 'A':
            if number <= 3: return "Flank_Openings" # Larsen, Bird, Reti
            if 10 <= number <= 39: return "English_Opening"
            if 40 <= number <= 44: return "Pion_Dame_Indienne_Ancienne"
            if 45 <= number <= 49: return "London_Torre_Trompowsky"
            if 51 <= number <= 52: return "Budapest_Gambit"
            if 53 <= number <= 55: return "Old_Indian"
            if 57 <= number <= 59: return "Benko_Gambit"
            if 56 or 60 <= number <= 79: return "Benoni_Defense"
            if 80 <= number <= 99: return "Dutch_Defense"
            return "Other_A_Openings"

        elif letter == 'B':
            if number == 0: return "Nimzowitsch_Defense"
            if number == 1: return "Scandinavian_Defense"
            if 2 <= number <= 5: return "Alekhine_Defense"
            if 7 <= number <= 9: return "Pirc_Robatsch"
            if 10 <= number <= 19: return "Caro_Kann"
            if 20 <= number <= 99: return "Sicilian_Defense"
            return "Other_B_Openings"

        elif letter == 'C':
            if number <= 19: return "French_Defense"
            if 20 <= number <= 22: return "Center_Game_Alapin"
            if 23 <= number <= 24: return "Bishop_Opening"
            if 25 <= number <= 29: return "Vienna_Game"
            if 30 <= number <= 39: return "Kings_Gambit"
            if number == 41: return "Philidor_Defense"
            if 42 <= number <= 43: return "Petrov_Defense"
            if number == 45: return "Scotch_Game"
            if 47 <= number <= 49: return "Four_Knights"
            if 50 <= number <= 54: return "Italian_Game"
            if 55 <= number <= 59: return "Two_Knights"
            if 60 <= number <= 99: return "Ruy_Lopez_Spanish"
            return "Other_C_Openings"

        elif letter == 'D':
            if number <= 5: return "Closed_Games_London_Colle"
            if number == 7: return "Chigorin_Defense"
            if 8 <= number <= 9: return "Albin_Counter_Gambit"
            if 10 <= number <= 19: return "Slav_Defense"
            if 20 <= number <= 29: return "Queens_Gambit_Accepted"
            if 30 <= number <= 69: return "Queens_Gambit_Declined"
            if 70 <= number <= 99: return "Grunfeld_Defense"
            return "Other_D_Openings"

        elif letter == 'E':
            if number <= 9: return "Catalan_Opening"
            if 10 <= number <= 19: return "Bogo_West_Indian"
            if 20 <= number <= 59: return "Nimzo_Indian"
            if 60 <= number <= 99: return "Kings_Indian"
            return "Other_E_Openings"

        return "Other"
    

class ChessClassifierPipeline2:
    """
    Pipeline complet pour prédire la classe Elo via Random Forest
    """
    
    def __init__(self, data_path='data/stat_parties.parquet'):                
        self.df = pd.read_parquet(data_path)
        self.best_model = None
        self.feature_names = None
                
        # On vérifie que toutes les classes sont bien présentes dans notre liste
        self.custom_order = ['Debutant', 'Intermediaire', 'Avance', 'Expert', 'Maitre']
        self.seed = 42

        print(f"Ordre des classes : {self.custom_order}")        
        print(f"✅ {len(self.df)} joueurs chargés")
        print(f"Colonnes: {list(self.df.columns)}")
        print(f"\nDistribution des classes:")
        print(self.df['class'].value_counts())
    
    def prepare_features(self):
        """
        Transforme les features complexes (top 5 ouvertures, etc.) en features numériques
        """
        print("\n🔧 PRÉPARATION DES FEATURES")
        print("="*70)
        
        feature_list = []
        
        # 1. Features de base
              
        if 'avg_opening_move_time' in self.df.columns:
            feature_list.append(self.df[['avg_opening_move_time']])
        
        # 2. Top 5 ouvertures jouées (fréquence)
        print("📊 Extraction des fréquences d'ouvertures...")
        top5_played = self._extract_opening_frequencies(
            self.df['top_5_openings_used'], 
            prefix='freq'
        )
        feature_list.append(top5_played)
        
        # 3. Top 5 ouvertures gagnantes (win rate)
        print("🏆 Extraction des win rates par ouverture...")
        top5_wins = self._extract_opening_winrates(
            self.df['top_5_openings_win'],
            prefix='winrate'
        )
        feature_list.append(top5_wins)
        
        # 4. Features agrégées
        print("🧮 Calcul de features agrégées...")
        
        # Diversité du répertoire (nombre d'ouvertures uniques)
        diversity = self.df['top_5_openings_used'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0).rename('opening_diversity')
        feature_list.append(diversity)
        
        # Concentration (% de parties avec l'ouverture la plus jouée)
        concentration = self.df['top_5_openings_used'].apply(
            self._calculate_concentration
        ).rename('opening_concentration')
        feature_list.append(concentration)
        
        # Winrate moyen sur les top ouvertures
        avg_winrate = self.df['top_5_openings_win'].apply(
            self._calculate_avg_winrate
        ).rename('avg_winrate_top5')
        feature_list.append(avg_winrate)
        
        # Combiner toutes les features
        X = pd.concat(feature_list, axis=1)
        
        # Remplir les NaN
        X = X.fillna(0)
        
        # Label (classe à prédire)
        y = self.df['class'].map({name: i for i, name in enumerate(self.custom_order)})        
        self.feature_names = X.columns.tolist()
        
        print(f"\n✅ Features créées: {X.shape[1]} colonnes")
        print(f"   Classes (ordre logique): {self.custom_order}")
        
        return X, y
    
    def _extract_opening_frequencies(self, series, prefix='freq', top_n=12):
        """Version avec regroupement par familles d'ouvertures"""
        all_families = []
        
        # 1. On transforme chaque ECO en nom de famille
        for item in series:
            if isinstance(item, str):
                ecos = [e.strip() for e in item.split(',')]
                families = [map_eco_to_name(e) for e in ecos]
                all_families.extend(families)
        
        # 2. On prend les familles les plus courantes
        most_common_families = [fam for fam, count in Counter(all_families).most_common(top_n) if fam != "Unknown"]
        
        # 3. Création des features
        features = {}
        for fam in most_common_families:
            col_name = f'{prefix}_{fam}'
            # On met 1 si l'une des ouvertures du joueur appartient à cette famille
            features[col_name] = series.apply(
                lambda x: 1 if isinstance(x, str) and any(map_eco_to_name(e.strip()) == fam for e in x.split(',')) else 0
            )
        
        return pd.DataFrame(features)
    
    def _extract_opening_winrates(self, series, prefix='winrate', top_n=10):
        """Version avec regroupement par familles pour les winrates"""
        all_families = []
        
        # 1. On identifie les familles présentes dans les colonnes de victoires
        for item in series:
            if isinstance(item, str):
                ecos = [e.strip() for e in item.split(',')]
                families = [map_eco_to_name(e) for e in ecos]
                all_families.extend(families)
        
        # 2. On prend les familles les plus représentées parmi les victoires
        most_common_win_families = [fam for fam, count in Counter(all_families).most_common(top_n) if fam != "Unknown"]
        
        features = {}
        for fam in most_common_win_families:
            col_name = f'{prefix}_{fam}'
            # On met 1 si le joueur a gagné avec une ouverture de cette famille
            features[col_name] = series.apply(
                lambda x: 1 if isinstance(x, str) and any(map_eco_to_name(e.strip()) == fam for e in x.split(',')) else 0
            )
        return pd.DataFrame(features)
    
    def _calculate_concentration(self, top5_str):
        """Mesure la diversité du répertoire via le texte"""
        if not isinstance(top5_str, str) or not top5_str.strip():
            return 0
        # On compte combien d'ouvertures il y a. 
        # Plus il y en a, moins le joueur est 'concentré' sur une seule.
        openings = [e.strip() for e in top5_str.split(',')]
        return 1.0 / len(openings) if len(openings) > 0 else 0
    
    def _calculate_avg_winrate(self, top5_str):
        """Compte simplement le nombre d'ouvertures 'fortes'"""
        if not isinstance(top5_str, str) or not top5_str.strip():
            return 0
        openings = [e.strip() for e in top5_str.split(',')]
        return float(len(openings))


    def train_with_cross_validation(self, X, y, cv_folds=5):
        """
        Entraînement avec cross-validation stratifiée
        """
        print("\n🎯 CROSS-VALIDATION")
        print("="*70)
        
        # Stratified K-Fold pour équilibrer les classes
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Modèle de base
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Cross-validation
        print(f"🔄 Cross-validation avec {cv_folds} folds...")
        cv_scores = cross_val_score(
            rf_base, X, y, 
            cv=skf, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"\n📊 Résultats CV:")
        print(f"   Accuracy par fold: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"   Moyenne: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # F1-score (mieux pour classes déséquilibrées)
        cv_f1_scores = cross_val_score(
            rf_base, X, y,
            cv=skf,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        print(f"   F1-Score macro: {cv_f1_scores.mean():.3f} (+/- {cv_f1_scores.std():.3f})")
        
        return cv_scores, cv_f1_scores
    
    def hyperparameter_tuning(self, X, y, cv_folds=5):
        """
        Recherche des meilleurs hyperparamètres avec GridSearchCV
        """
        print("\n🔍 RECHERCHE D'HYPERPARAMÈTRES")
        print("="*70)
        
        # Grille de paramètres à tester
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        print(f"🎲 Combinaisons à tester: {np.prod([len(v) for v in param_grid.values()])}")
        
        # GridSearchCV avec Stratified K-Fold
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        print("⏳ Recherche en cours (peut prendre plusieurs minutes)...")
        grid_search.fit(X, y)
        
        print(f"\n✅ Meilleurs paramètres:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 Meilleur score (F1 macro): {grid_search.best_score_:.3f}")
        
        self.best_model = grid_search.best_estimator_
        
        return grid_search
    
    def train_final_model(self, X_train, X_test, y_train, y_test, params=None):
        """
        Entraînement du modèle final sur train/test split
        """
        print("\n🎓 ENTRAÎNEMENT DU MODÈLE FINAL")
        print("="*70)
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 20,
                'random_state': self.seed, # 4. FIXER LE HASARD DANS LE MODÈLE
                'n_jobs': -1
            }
        else:
            params['random_state'] = self.seed
        
        print(f"Paramètres utilisés:")
        for k, v in params.items():
            if k != 'n_jobs':
                print(f"   {k}: {v}")
        
        # Entraîner
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        # Prédictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        
        # Métriques
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='macro')
        
        print(f"\n📊 Performances:")
        print(f"   Accuracy Train: {train_acc:.3f}")
        print(f"   Accuracy Test:  {test_acc:.3f}")
        print(f"   F1-Score Test:  {test_f1:.3f}")
        
        if train_acc - test_acc > 0.1:
            print(f"   ⚠️  Possible overfitting (diff: {train_acc - test_acc:.3f})")
        
        self.best_model = rf
        
        return rf, y_pred_test
    
    def plot_confusion_matrix(self, y_true, y_pred):
        # On force l'ordre des labels dans la matrice
        labels_indices = range(len(self.custom_order))
        cm = confusion_matrix(y_true, y_pred, labels=labels_indices)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.custom_order, # On utilise notre ordre
            yticklabels=self.custom_order
        )
        plt.title('Matrice de Confusion', fontsize=16)
        plt.ylabel('Vraie Classe', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Matrice sauvegardée: confusion_matrix2.png")
    
    def plot_feature_importance(self, top_n=20):
        """Affiche l'importance des features"""
        if self.best_model is None:
            print("❌ Aucun modèle entraîné")
            return
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Features les plus importantes', fontsize=16)
        plt.barh(
            range(top_n),
            importances[indices],
            align='center'
        )
        plt.yticks(
            range(top_n),
            [self.feature_names[i] for i in indices]
        )
        plt.xlabel('Importance', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Graphique sauvegardé: feature_importance2.png")
        
        # Afficher les valeurs
        print(f"\n📊 Top {top_n} features:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i:2d}. {self.feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    def print_classification_report(self, y_true, y_pred):
        report = classification_report(
            y_true, 
            y_pred,
            target_names=self.custom_order, # On utilise notre ordre
            digits=3
        )
        print(report)
    
    def save_model(self, filepath='models/random_forest_chess2.pkl'):
        """Sauvegarde le modèle"""
        import pickle
        import os
        
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'custom_order': self.custom_order, # On remplace label_encoder par custom_order
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modèle sauvegardé: {filepath}")


# 🚀 SCRIPT D'EXÉCUTION COMPLET
if __name__ == "__main__":
    
    # 1. Charger et préparer les données
    print("="*70)
    print("🎯 PIPELINE RANDOM FOREST - CLASSIFICATION ELO")
    print("="*70)
    
    pipeline = ChessClassifierPipeline2('data/stat_parties.parquet')
    
    # 2. Préparer les features
    X, y = pipeline.prepare_features()
    
    # 3. Split train/test stratifié
    print("\n📊 SPLIT TRAIN/TEST")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Important: préserve la distribution des classes
    )
    
    print(f"Train: {X_train.shape[0]} exemples")
    print(f"Test:  {X_test.shape[0]} exemples")
    
    # Vérifier la distribution
    print(f"\nDistribution Train:")
    train_classes = pd.Series(y_train).value_counts()
    for class_idx, count in train_classes.items():
        class_name = pipeline.custom_order[class_idx]  # Utilise custom_order
        print(f"   {class_name:15s}: {count}")
    
    # 4. Cross-validation baseline
    cv_acc, cv_f1 = pipeline.train_with_cross_validation(X_train, y_train, cv_folds=5)
    
    # 5. Recherche d'hyperparamètres (optionnel, prend du temps)
    print("\n⚠️  Voulez-vous lancer la recherche d'hyperparamètres ?")
    print("   (Peut prendre 10-30 minutes)")
    do_grid_search = input("   [y/N]: ").lower() == 'y'
    
    if do_grid_search:
        grid_search = pipeline.hyperparameter_tuning(X_train, y_train, cv_folds=5)
        best_params = grid_search.best_params_
    else:
        # Paramètres par défaut
        best_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    
    # 6. Entraînement final
    rf_model, y_pred = pipeline.train_final_model(
        X_train, X_test, y_train, y_test,
        params=best_params
    )
    
    # 7. Évaluation détaillée
    pipeline.print_classification_report(y_test, y_pred)
    
    # 8. Visualisations
    print("\n📊 GÉNÉRATION DES GRAPHIQUES")
    print("="*70)
    pipeline.plot_confusion_matrix(y_test, y_pred)
    pipeline.plot_feature_importance(top_n=20)
    
    # 9. Sauvegarder le modèle
    pipeline.save_model('models/random_forest_chess2.pkl')
    
    print("\n" + "="*70)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("="*70)
    print("\n📁 Fichiers générés:")
    print("   - models/random_forest_chess2.pkl")
    print("   - confusion_matrix2.png")
    print("   - feature_importance2.png")
