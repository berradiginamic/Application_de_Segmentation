import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import datetime
import io
from scipy import stats

def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/donnees.jpg", use_column_width=True)

def transformation_page():
    """
    Fonction principale pour la page de Transformation des Données.
    Permet de gérer les données manquantes, effectuer des transformations, encoder les variables catégorielles,
    gérer les données de date et heure, détecter et traiter les valeurs aberrantes, gérer les types de données,
    sauvegarder et réinitialiser les données traitées.
    """
    st.title("Page de Transformation des Données")
    st.write("Cette page fournit des fonctionnalités de traitements des données.")

    # Vérifier si le dataframe est disponible dans l'état de session
    if 'final_dataframe' in st.session_state:
        df = st.session_state['final_dataframe']
        st.dataframe(df)

        # Gestion des Données Manquantes
        st.header("Gestion des Données Manquantes")
        if st.checkbox("Afficher les Données Manquantes"):
            st.write(df.isnull().sum())

        missing_data_method = st.selectbox("Gérer les Données Manquantes",
                                           ["Aucune", "Supprimer les lignes", "Supprimer les colonnes",
                                            "Remplir avec la Moyenne",
                                            "Remplir avec la Médiane", "Remplir avec le Mode",
                                            "Remplir avec une Valeur Spécifique"])

        if missing_data_method == "Supprimer les lignes":
            df = df.dropna()
        elif missing_data_method == "Supprimer les colonnes":
            df = df.dropna(axis=1)
        elif missing_data_method == "Remplir avec la Moyenne":
            df = df.fillna(df.mean())
        elif missing_data_method == "Remplir avec la Médiane":
            df = df.fillna(df.median())
        elif missing_data_method == "Remplir avec le Mode":
            df = df.fillna(df.mode().iloc[0])
        elif missing_data_method == "Remplir avec une Valeur Spécifique":
            fill_value = st.text_input("Entrer la valeur pour remplir les données manquantes")
            if fill_value:
                try:
                    fill_value = float(fill_value) if fill_value.replace('.', '', 1).isdigit() else fill_value
                    df = df.fillna(fill_value)
                except ValueError:
                    st.error("Veuillez entrer une valeur numérique valide.")

        st.dataframe(df)

        # Transformation des Données
        st.header("Transformation des Données")
        transformation_method = st.selectbox("Sélectionner la Méthode de Transformation",
                                             ["Aucune", "Normalisation (Min-Max)", "Standardisation (Z-score)"])

        if transformation_method == "Normalisation (Min-Max)":
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))
        elif transformation_method == "Standardisation (Z-score)":
            scaler = StandardScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))

        st.dataframe(df)

        # Encodage des Variables Catégorielles
        st.header("Encodage des Variables Catégorielles")
        encoding_method = st.selectbox("Sélectionner la Méthode d'Encodage",
                                       ["Aucun", "Encodage One-Hot", "Encodage Label"])

        if encoding_method == "Encodage One-Hot":
            df = pd.get_dummies(df)
        elif encoding_method == "Encodage Label":
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = le.fit_transform(df[col])

        st.dataframe(df)

        # Filtrage et Sélection des Données
        st.header("Filtrage et Sélection des Données")
        if st.checkbox("Supprimer les Doublons"):
            df = df.drop_duplicates()

        st.dataframe(df)

        columns_to_drop = st.multiselect("Sélectionner les Colonnes à Supprimer", df.columns)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        st.dataframe(df)

        # Gestion des Données de Date et Heure
        st.header("Gestion des Données de Date et Heure")
        date_columns = st.multiselect("Sélectionner les Colonnes de Date", df.columns[df.dtypes == 'object'])

        for date_col in date_columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f'{date_col}_jour'] = df[date_col].dt.day
            df[f'{date_col}_mois'] = df[date_col].dt.month
            df[f'{date_col}_année'] = df[date_col].dt.year

        st.dataframe(df)

        # Gestion des Valeurs Aberrantes
        st.header("Gestion des Valeurs Aberrantes")
        outlier_method = st.selectbox("Sélectionner la Méthode de Détection des Valeurs Aberrantes",
                                      ["Aucune", "Z-score", "IQR"])

        if outlier_method == "Z-score":
            numeric_df = df.select_dtypes(include=[np.number])
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
            df = df[(z_scores < 3).all(axis=1)]
        elif outlier_method == "IQR":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.dataframe(df)

        # Gestion des Types de Données
        st.header("Gestion des Types de Données")
        for col in df.columns:
            col_type = st.selectbox(f"Sélectionner le type pour la colonne {col}", ["Aucun", "int", "float", "str"],
                                    index=0)
            if col_type != "Aucun":
                try:
                    df[col] = df[col].astype(col_type)
                except ValueError:
                    st.error(f"Impossible de convertir la colonne {col} en type {col_type}")

        st.dataframe(df)

        # Sauvegarde des Données Traitées / Réinitialisation du DataFrame
        st.header("Sauvegarde/Réinitialisation des Données Traitées")
        if st.button("Sauvegarder les Données Traitées"):
            st.session_state['final_dataframe'] = df
            st.success("Données traitées sauvegardées et mises à jour!")

        if st.button("Réinitialiser le DataFrame"):
            st.session_state['final_dataframe'] = st.session_state.get('original_dataframe', df)
            st.success("DataFrame réinitialisé à l'état d'origine.")

    else:
        st.write(
            "Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la page de connexion aux données.")


def calculate_age(birthdate):
    """
    Fonction pour calculer l'âge à partir d'une date de naissance.
    """
    today = datetime.date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


def clean_data(df, remove_na, remove_inf, remove_duplicates):
    """
    Fonction pour nettoyer les données en supprimant les valeurs infinies, les NaNs, et les lignes dupliquées.
    """
    df_clean = df.copy()

    # Remplacer les valeurs infinies par NaN
    if remove_inf:
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # Supprimer les lignes contenant des NaN
    if remove_na:
        df_clean = df_clean.dropna()

    # Supprimer les lignes dupliquées
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()

    return df_clean


def analysis_page():
    """
    Fonction principale pour la page d'Analyse des Données.
    Réalise l'Analyse Exploratoire des Données (AED) et les analyses statistiques sur un jeu de données.
    """
    st.title("Analyse de la table choisie")
    st.write(
        "Cette partie fournit des insights d'Analyse Exploratoire des Données (AED) et des analyses statistiques pour la table choisie.")

    # Vérifier si le dataframe est disponible dans l'état de session
    if 'final_dataframe' in st.session_state and st.session_state['final_dataframe'] is not None:
        df = st.session_state['final_dataframe']

        # Vérifier si la colonne 'birth_day' existe
        if 'birth_day' in df.columns:
            # Convertir 'birth_day' en datetime
            df['birth_day'] = pd.to_datetime(df['birth_day'], errors='coerce')

            # Ajouter une colonne d'âge
            df['age'] = df['birth_day'].apply(lambda x: calculate_age(x) if pd.notnull(x) else np.nan)

            # Mettre à jour le DataFrame final dans l'état de session
            st.session_state['final_dataframe'] = df

            st.write("Tableau avec la colonne d'âge ajoutée :")
            st.dataframe(df)

        else:
            st.write("La colonne 'birth_day' n'existe pas dans le DataFrame final.")

        # Afficher les données avant nettoyage
        st.header("Données Initiales")
        st.dataframe(df)

        # Options de nettoyage
        st.sidebar.header("Options de Nettoyage des Données")
        remove_na = st.sidebar.checkbox("Supprimer les lignes avec des valeurs manquantes")
        remove_inf = st.sidebar.checkbox("Remplacer les valeurs infinies par NaN")
        remove_duplicates = st.sidebar.checkbox("Supprimer les lignes dupliquées")

        # Nettoyer les données en fonction des options sélectionnées
        if st.sidebar.button("Appliquer le Nettoyage"):
            df_clean = clean_data(df, remove_na, remove_inf, remove_duplicates)
            st.write("Tableau nettoyé :")
            st.dataframe(df_clean)

            # Mettre à jour le DataFrame final dans l'état de session après nettoyage
            st.session_state['final_dataframe'] = df_clean
            df = df_clean

            # Boutons de téléchargement
            st.download_button(
                label="Télécharger le tableau nettoyé en CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='tableau_nettoye.csv',
                mime='text/csv'
            )

            towrite = io.BytesIO()
            df.to_excel(towrite, index=False, engine='xlsxwriter')
            towrite.seek(0)
            st.download_button(
                label="Télécharger le tableau nettoyé en Excel",
                data=towrite,
                file_name='tableau_nettoye.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        st.header("Statistiques Sommaires")
        st.write("Statistiques de base du jeu de données :")
        st.write(df.describe())
        st.write("""
        **Explication** : Les statistiques sommaires fournissent un aperçu rapide des caractéristiques numériques du jeu de données, y compris des mesures telles que la moyenne, l'écart-type et les quartiles.
        Elles aident à comprendre la tendance centrale, la dispersion et la plage de valeurs pour chaque colonne numérique.
        """)

        st.header("Décompte des Valeurs")
        st.write("Décompte des valeurs pour les variables catégorielles :")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            st.subheader(col)
            st.write(df[col].value_counts())
            st.write(f"""
            **Explication** : Les décomptes des valeurs montrent la fréquence de chaque catégorie dans les variables catégorielles. 
            Ils aident à comprendre la distribution et le déséquilibre des catégories, ce qui est important pour l'ingénierie des caractéristiques et les décisions de modélisation.
            """)

        st.header("Analyse Statistique")

        # Exemple : Test d'Hypothèse (exemple de test t)
        st.subheader("Test d'Hypothèse (Exemple : test t)")
        available_cols = df.columns.tolist()

        col1 = st.selectbox("Sélectionner la première colonne pour le test t", available_cols, index=0)
        col2 = st.selectbox("Sélectionner la deuxième colonne pour le test t", available_cols, index=1)

        if st.button("Exécuter le test t"):
            try:
                # Convert categorical variables to numeric if needed
                if df[col1].dtype == 'object':
                    df[col1] = pd.to_numeric(df[col1].astype('category').cat.codes, errors='coerce')
                if df[col2].dtype == 'object':
                    df[col2] = pd.to_numeric(df[col2].astype('category').cat.codes, errors='coerce')

                # Ensure both columns are numeric before performing the test
                col1_numeric = pd.to_numeric(df[col1], errors='coerce')
                col2_numeric = pd.to_numeric(df[col2], errors='coerce')

                if col1_numeric.isnull().all() or col2_numeric.isnull().all():
                    st.error("Les colonnes sélectionnées ne contiennent pas de données numériques valides.")
                else:
                    t_stat, p_value = stats.ttest_ind(col1_numeric.dropna(), col2_numeric.dropna())
                    st.write(f"Résultat du test t entre {col1} et {col2} :")
                    st.write(f"Statistique t : {t_stat}")
                    st.write(f"Valeur de p : {p_value}")
                    st.write("""
                    **Explication** : Le test t compare les moyennes de deux groupes de données numériques pour déterminer s'ils sont significativement différents l'un de l'autre.
                    La statistique t mesure la différence par rapport à la variation des données, tandis que la valeur de p indique la signification de la différence.
                    """)

                    if p_value < 0.05:
                        st.write("Il y a une différence significative entre les groupes.")
                    else:
                        st.write("Il n'y a pas de différence significative entre les groupes.")
                    st.write("""
                    **Interprétation** : Si la valeur de p est inférieure à 0,05, cela suggère que la différence observée entre les groupes n'est probablement pas due au hasard (c'est-à-dire statistiquement significative).
                    Cela pourrait impliquer une différence significative entre les variables comparées.
                    """)
            except Exception as e:
                st.error(f"Erreur lors de l'exécution du test t : {e}")

        st.subheader("Analyse de Corrélation")
        st.write("Matrice de corrélation :")

        # Filtrer uniquement les colonnes numériques pour la corrélation
        numeric_df = df.select_dtypes(include=['number'])

        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            st.write(corr_matrix)
            st.write("""
            **Explication** : La matrice de corrélation montre les corrélations entre toutes les colonnes numériques dans le jeu de données. 
            Les coefficients de corrélation vont de -1 à +1, où +1 indique une forte corrélation positive, -1 indique une forte corrélation négative, et 0 indique aucune corrélation.
            Cela aide à identifier les relations entre les variables, ce qui est crucial pour la sélection des caractéristiques et la compréhension des dépendances dans les données.
            """)
        else:
            st.write("Aucune colonne numérique disponible pour calculer la matrice de corrélation.")
            st.write("""
            **Explication** : La matrice de corrélation nécessite des colonnes numériques pour calculer les relations entre les variables. 
            Veuillez vérifier votre jeu de données pour vous assurer qu'il contient des colonnes numériques.
            """)

    else:
        st.write(
            "Aucune donnée disponible. Veuillez télécharger ou vous connecter à une source de données sur la page de connexion aux données.")


if __name__ == "__main__":
    show_homepage()
    transformation_page()
    # Appel de la fonction d'analyse après la transformation
    analysis_page()
