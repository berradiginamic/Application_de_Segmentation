import streamlit as st
import pandas as pd

def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/description.jpg", use_column_width=True)
def main():
    show_homepage()
    """
    Fonction principale pour la page de description des données.
    Affiche les statistiques descriptives, les types de données,
    les valeurs manquantes et les valeurs uniques des colonnes catégorielles.
    """
    st.title("Page de Description des Données")
    st.write("Ceci est la page de description des données.")

    if 'final_dataframe' in st.session_state:
        df = st.session_state['final_dataframe']

        # Afficher les statistiques descriptives de base
        st.header("Statistiques descriptives")
        st.write(df.describe(include='all'))

        # Afficher les types de données
        st.header("Types de Données")
        st.write(df.dtypes)

        # Afficher les valeurs manquantes
        st.header("Valeurs Manquantes")
        st.write(df.isnull().sum())

        # Afficher les valeurs uniques pour les colonnes catégorielles
        st.header("Valeurs Uniques dans les Colonnes Catégorielles")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            st.subheader(f"Colonne: {col}")
            st.write(df[col].value_counts())

    else:
        st.write("Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la "
                 "page de connexion aux données.")


if __name__ == "__main__":
    main()