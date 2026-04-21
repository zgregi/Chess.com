"""
Pipeline complet pour la classification Elo par Random Forest
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os


# ========================================================================
# PREPROCESSING
# ========================================================================

def map_eco_to_name(eco):
    """Mappe un code ECO vers une famille d'ouvertures"""
    if not isinstance(eco, str) or len(eco) < 3:
        return "Unknown"
    
    letter = eco[0].upper()
    try:
        number = int(eco[1:3])
    except ValueError:
        return "Unknown"

    if letter == 'A':
        if number <= 3: return "Flank_Openings"
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


def prepare_features(df):
    """Prépare les features à partir du DataFrame"""
    feature_list = []
    
    # Temps moyen
    if 'avg_opening_move_time' in df.columns:
        feature_list.append(df[['avg_opening_move_time']])
    
    # Fréquences d'ouvertures
    all_families = []
    for item in df['top_5_openings_used']:
        if isinstance(item, str):
            ecos = [e.strip() for e in item.split(',')]
            families = [map_eco_to_name(e) for e in ecos]
            all_families.extend(families)
    
    most_common = [f for f, c in Counter(all_families).most_common(12) if f != "Unknown"]
    
    freq_features = {}
    for fam in most_common:
        freq_features[f'freq_{fam}'] = df['top_5_openings_used'].apply(
            lambda x: 1 if isinstance(x, str) and any(
                map_eco_to_name(e.strip()) == fam for e in x.split(',')
            ) else 0
        )
    feature_list.append(pd.DataFrame(freq_features))
    
    # Win rates
    all_win_families = []
    for item in df['top_5_openings_win']:
        if isinstance(item, str):
            ecos = [e.strip() for e in item.split(',')]
            families = [map_eco_to_name(e) for e in ecos]
            all_win_families.extend(families)
    
    most_common_wins = [f for f, c in Counter(all_win_families).most_common(10) if f != "Unknown"]
    
    win_features = {}
    for fam in most_common_wins:
        win_features[f'winrate_{fam}'] = df['top_5_openings_win'].apply(
            lambda x: 1 if isinstance(x, str) and any(
                map_eco_to_name(e.strip()) == fam for e in x.split(',')
            ) else 0
        )
    feature_list.append(pd.DataFrame(win_features))
    
    # Features agrégées
    diversity = df['top_5_openings_used'].apply(
        lambda x: len(x.split(',')) if isinstance(x, str) else 0
    ).rename('opening_diversity')
    
    concentration = df['top_5_openings_used'].apply(
        lambda x: 1.0 / len(x.split(',')) if isinstance(x, str) and x.strip() else 0
    ).rename('opening_concentration')
    
    avg_winrate = df['top_5_openings_win'].apply(
        lambda x: float(len(x.split(','))) if isinstance(x, str) and x.strip() else 0
    ).rename('avg_winrate_top5')
    
    feature_list.extend([diversity, concentration, avg_winrate])
    
    X = pd.concat(feature_list, axis=1).fillna(0)
    
    class_order = ['Debutant', 'Intermediaire', 'Avance', 'Expert', 'Maitre']
    y = df['class'].map({name: i for i, name in enumerate(class_order)})
    
    return X, y, X.columns.tolist(), class_order


# ========================================================================
# PIPELINE PRINCIPAL
# ========================================================================

class ChessClassifier:
    """Classe principale pour le pipeline de classification"""
    
    def __init__(self, data_path, random_state=42):
        self.df = pd.read_parquet(data_path)
        self.random_state = random_state
        self.X, self.y, self.feature_names, self.class_order = prepare_features(self.df)
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=random_state, stratify=self.y
        )
        
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.metrics = None
    
    def get_data_summary(self):
        """Imprime et renvoie un résumé complet des données"""
        summary = {
            'n_joueurs': len(self.df),
            'n_features': len(self.feature_names),
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'class_distribution': self.df['class'].value_counts().to_dict()
        }
        print(f"Nombre total de joueurs : {summary['n_joueurs']}")
        print(f"Nombre de caractéristiques : {summary['n_features']}")
        print(f"Répartition Train/Test  : {summary['n_train']} / {summary['n_test']}")

        print("Distribution des classes :")
        for classe, count in summary['class_distribution'].items():
            percentage = (count / summary['n_joueurs']) * 100
            print(f"  - {classe:13}: {count} ({percentage:.1f}%)")

        # On garde le return au cas où tu aurais besoin des chiffres pour un calcul
        return summary
        
    
    def cross_validate(self, cv_folds=5):
        """Effectue la cross-validation"""
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_acc = cross_val_score(rf, self.X_train, self.y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_f1 = cross_val_score(rf, self.X_train, self.y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
        
        self.cv_results = {
            'accuracy_mean': cv_acc.mean(),
            'accuracy_std': cv_acc.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std()
        }
        
        return self.cv_results
    
    def tune_hyperparameters(self, cv_folds=5, verbose=0):
        """Recherche des meilleurs hyperparamètres"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_macro',
            n_jobs=-1,
            verbose=verbose
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        self.best_params['random_state'] = self.random_state
        self.best_params['n_jobs'] = -1
        
        return self.best_params, grid_search.best_score_
    
    def train_final_model(self, params=None):
        """Entraîne le modèle final et affiche les performances"""
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        if params is None:
            params = self.best_params if self.best_params else {
                'n_estimators': 200,
                'max_depth': 20,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        # Entraînement
        self.model = RandomForestClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        
        # Prédictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        self.y_pred = y_pred_test
        
        # Calcul des métriques
        self.metrics = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'test_f1': f1_score(self.y_test, y_pred_test, average='macro'),
            'overfitting': accuracy_score(self.y_train, y_pred_train) - accuracy_score(self.y_test, y_pred_test)
        }
      
        print(f"Accuracy Train : {self.metrics['train_accuracy']:.3f}")
        print(f"Accuracy Test  : {self.metrics['test_accuracy']:.3f}")
        print(f"F1-Score Test  : {self.metrics['test_f1']:.3f}")
        print(f"Overfitting    : {self.metrics['overfitting']:.3f}")
        print("📝 RAPPORT DE CLASSIFICATION DÉTAILLÉ :")
        # On utilise class_order pour que le rapport soit dans le bon sens (Débutant -> Maître)
        print(classification_report(self.y_test, y_pred_test, target_names=self.class_order))
  
        
        return self.metrics
    
    def get_classification_report(self):
        """Retourne le rapport de classification"""
        return classification_report(
            self.y_test, self.y_pred, 
            target_names=self.class_order, 
            digits=3
        )
    
    def plot_hyperparameter_results(self, params_to_display=None):
        """Affiche les résultats de l'optimisation des hyperparamètres"""
        if self.best_params is None:
            print("⚠️ Aucune optimisation effectuée")
            return
        
        if params_to_display is None:
            params_to_display = {k: v for k, v in self.best_params.items() 
                               if k not in ['random_state', 'n_jobs']}
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        params_str = '\n'.join([f'{k}: {v}' for k, v in params_to_display.items()])
        
        ax.text(0.5, 0.5, f'Meilleurs hyperparamètres\n\n{params_str}', 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        return params_to_display
    
    def plot_confusion_matrix(self, figsize=(10, 8), save_path=None):
        """Affiche la matrice de confusion"""
        labels_indices = range(len(self.class_order))
        cm = confusion_matrix(self.y_test, self.y_pred, labels=labels_indices)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_order, yticklabels=self.class_order,
                   cbar_kws={'label': 'Nombre de prédictions'})
        plt.title('Matrice de Confusion', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Vraie Classe', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, top_n=15, figsize=(12, 8), save_path=None):
        """Affiche l'importance des features"""
        if self.model is None:
            print("⚠️ Aucun modèle entraîné")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=figsize)
        plt.title(f'Top {top_n} Features les plus importantes', 
                 fontsize=16, fontweight='bold', pad=20)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        plt.barh(range(top_n), importances[indices], align='center', color=colors)
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return [(self.feature_names[i], importances[i]) for i in indices]
    
    def save_model(self, filepath='models/random_forest_final.pkl'):
        """Sauvegarde le modèle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'class_order': self.class_order,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'cv_results': self.cv_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    from sklearn.tree import plot_tree

    def visualize_one_tree(self, tree_index=0, max_depth=3):
        """Affiche un des arbres de la forêt pour comprendre la logique"""
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        # 1. On utilise self.model car c'est lui qui contient les arbres
        if self.model is None:
            print("❌ Modèle non entraîné")
            return
            
        # 2. On extrait l'arbre depuis le modèle, pas depuis les paramètres
        individual_tree = self.model.estimators_[tree_index]
        
        plt.figure(figsize=(20, 10))
        plot_tree(
            individual_tree,
            feature_names=self.feature_names,
            # Assure-toi que c'est bien self.class_order le nom dans ta classe
            class_names=self.class_order, 
            filled=True,          
            rounded=True,         
            max_depth=max_depth,  
            fontsize=10
        )
        plt.title(f"Logique de décision (Arbre n°{tree_index})", fontsize=15)
        plt.show()