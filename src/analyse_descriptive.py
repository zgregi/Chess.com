import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

def plot_rating(df_joueurs):
    sns.histplot(df_joueurs['rating'], bins=20)
    plt.title("Distribution du rating")
    plt.show()
def add_activity_profile(df_joueurs):

    def categoriser_joueur(nb):
        if nb < 50:
            return "Occasionnel"
        elif nb < 100:
            return "Régulier"
        else:
            return "Très actif"

    df_joueurs['profil_activite'] = df_joueurs['nb_games'].apply(categoriser_joueur)

    return df_joueurs
def plot_activity(df_joueurs):
    # 1. Calculer la proportion (fréquence relative)
    # value_counts(normalize=True) donne des valeurs entre 0 et 1
    counts = df_joueurs['profil_activite'].value_counts(normalize=True).reset_index()
    counts.columns = ['profil_activite', 'proportion']
    
    # Conversion en pourcentage pour plus de lisibilité (0.25 -> 25%)
    counts['proportion'] = counts['proportion'] * 100

    # 2. Créer le graphique
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='profil_activite', 
        y='proportion', 
        data=counts,
        order=["Occasionnel", "Régulier", "Très actif"]
    )

    # 3. Ajouter les étiquettes de pourcentage au-dessus des barres
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.title("Répartition des profils d'activité (en % du total)")
    plt.ylabel("Part des joueurs (%)")
    plt.xlabel("Profil")
    plt.ylim(0, 45) # Fixer le max à 100% pour bien voir la proportion
    plt.show()
def plot_activity_vs_rating(df_joueurs):
    sns.boxplot(x='profil_activite', y='rating', data=df_joueurs,
                order=["Occasionnel", "Régulier", "Très actif"])
    plt.title("Rating selon le niveau d'activité")
    plt.show()
def plot_time_distribution(df_parties):
    plt.figure(figsize=(10, 6))
    
    # stat='percent' transforme l'axe Y en pourcentage du total
    sns.histplot(
        data=df_parties, 
        x='avg_opening_move_time', 
        bins=20, 
        stat='percent',  # Ajoute une courbe de densité pour mieux voir la tendance
        color='skyblue'
    )
    
    plt.title("Distribution du temps de réflexion")
    plt.xlabel("Temps moyen par coup (secondes)")
    plt.ylabel("Part des joueurs (%)")
    
    # Optionnel : ajouter une grille pour mieux lire les pourcentages
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
def plot_opening_distribution_by_class(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # 1. On utilise la colonne identifiée dans ton erreur
    col_classe = 'class' 
    col_ouverture = 'top_5_openings_used'
    
    if col_classe not in df.columns or col_ouverture not in df.columns:
        print(f"❌ Colonnes manquantes. Colonnes dispo : {df.columns.tolist()}")
        return

    df_plot = df.copy()

    # 2. Extraction de la FAMILLE d'ouverture
    # On prend la première de la liste, et on s'arrête au premier ":" ou " " pour avoir la famille
    def extraire_famille(txt):
        if pd.isna(txt): return "Inconnu"
        # On nettoie les crochets/guillemets si c'est une chaîne de liste
        txt = str(txt).replace('[', '').replace(']', '').replace("'", "").split(',')[0]
        # On coupe au ":" (ex: "Sicilian Defense: Dragon" -> "Sicilian Defense")
        return txt.split(':')[0].strip()

    df_plot['opening_family'] = df_plot[col_ouverture].apply(extraire_famille)

    # 3. On garde le Top 8 des familles les plus jouées
    top_families = df_plot['opening_family'].value_counts().nlargest(8).index
    df_top = df_plot[df_plot['opening_family'].isin(top_families)].copy()

    # 4. Tableau croisé en pourcentages
    dist = pd.crosstab(df_top[col_classe], df_top['opening_family'], normalize='index') * 100
    
    # 5. Ordre des niveaux
    ordre = ['Debutant', 'Intermediaire', 'Avance', 'Expert', 'Maitre']
    ordre_existant = [c for c in ordre if c in dist.index]
    dist = dist.reindex(ordre_existant)

    # 6. Graphique
    plt.figure(figsize=(14, 8))
    # 'Set3' ou 'Paired' offrent des couleurs distinctes pour les familles
    dist.plot(kind='barh', stacked=True, figsize=(13, 7), colormap='Paired')

    plt.title("Répartition des ouvertures par niveau", fontsize=15, pad=20)
    plt.xlabel("Part de l'utilisation (%)", fontsize=12)
    plt.ylabel("Niveau du joueur", fontsize=12)
    plt.legend(title="Ouvertures", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_time_vs_speed(df_parties):
    sns.scatterplot(
        x='avg_opening_move_time',
        y='rating',
        data=df_parties
    )
    plt.title("Relation entre style de jeu et vitesse moyenne des 10 premiers coups")
    plt.show()
def add_profile(df_parties):

    def profil_joueur(row):
        if row['avg_opening_move_time'] < 5 and row['nb_rapid_games'] > 50:
            return "Rapide & actif"
        elif row['avg_opening_move_time'] > 10:
            return "Réfléchi"
        else:
            return "Standard"

    df_parties['profil'] = df_parties.apply(profil_joueur, axis=1)
    return df_parties
def plot_profile(df_parties):
    sns.countplot(x='profil', data=df_parties)
    plt.title("Profils de joueurs")
    plt.show()
def plot_clean_scatter(df_parties):
    df_clean = df_parties.dropna(subset=['avg_opening_move_time', 'rating'])

    df_clean['avg_opening_move_time'] = pd.to_numeric(df_clean['avg_opening_move_time'], errors='coerce')
    df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')

    df_clean = df_clean.dropna()

    sns.scatterplot(
        x='avg_opening_move_time',
        y='rating',
        data=df_clean
    )

    plt.title("Vitesse vs niveau")
    plt.show()
