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
    sns.countplot(x='profil_activite', data=df_joueurs,
                  order=["Occasionnel", "Régulier", "Très actif"])
    plt.title("Profil d'activité des joueurs")
    plt.show()
def plot_activity_vs_rating(df_joueurs):
    sns.boxplot(x='profil_activite', y='rating', data=df_joueurs,
                order=["Occasionnel", "Régulier", "Très actif"])
    plt.title("Rating selon le niveau d'activité")
    plt.show()
def plot_time_distribution(df_parties):
    sns.histplot(df_parties['avg_opening_move_time'], bins=20)
    plt.title("Distribution du temps de réflexion à l'ouverture")
    plt.show()
def plot_time_vs_rating(df_parties):
    sns.scatterplot(
        x='avg_opening_move_time',
        y='rating',
        data=df_parties
    )
    plt.title("Relation entre style de jeu et niveau")
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
