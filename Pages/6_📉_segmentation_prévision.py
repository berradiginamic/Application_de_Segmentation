import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import logging
import psycopg2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition d'un modèle Keras dans une classe pour l'utiliser avec Scikit-Learn
# Définir la classe KerasClassifierWrapper en premier
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, **kwargs):
        self.build_fn = build_fn
        self.model = None
        self.kwargs = kwargs

    def fit(self, X, y, **fit_kwargs):
        self.model = self.build_fn()
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/prediction.jpg", use_column_width=True)


# Fonction de nettoyage et de prétraitement des données
def preprocess_data(df, num_features, cat_features, target_column, cols_to_merge=None, merged_col_name=None):
    try:
        # Nettoyage des données
        df = df.replace(['?', '', 'NaN', 'None'], np.nan)  # Remplace les valeurs manquantes par NaN

        # Fusion des colonnes sélectionnées par l'utilisateur
        if cols_to_merge and merged_col_name:
            df[merged_col_name] = df[cols_to_merge].bfill(axis=1).iloc[:, 0]
            df.drop(columns=cols_to_merge, inplace=True)
            df[merged_col_name].fillna('missing', inplace=True)

        # Vérification de la présence de la colonne 'date_of_birth'
        if 'date_of_birth' in df.columns:
            # Conversion des dates
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], format='%Y-%m-%d', errors='coerce')
            invalid_dates = df['date_of_birth'].isna()

            if invalid_dates.any():
                st.warning("Certaines valeurs dans la colonne 'date_of_birth' sont invalides et ont été converties en NaT.")
                st.write(df[invalid_dates])

            # Calcul de l'âge
            today = pd.Timestamp.now()
            df['age'] = today.year - df['date_of_birth'].dt.year
            df['age'].fillna(-1, inplace=True)  # Remplir les âges manquants avec une valeur par défaut si nécessaire

        # Conversion des colonnes catégorielles en chaînes de caractères
        for col in cat_features:
            df[col] = df[col].astype(str)
            df[col].fillna('missing', inplace=True)

        # Suppression des lignes avec des valeurs manquantes dans les colonnes essentielles
        df.dropna(subset=num_features + cat_features + [target_column], inplace=True)

        # Séparation des features et de la cible
        X = df[num_features + cat_features]
        y = df[target_column]

        # Pipeline de transformation pour les données numériques
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Remplissage des valeurs manquantes par la médiane
            ('scaler', StandardScaler())
        ])

        # Pipeline de transformation pour les données catégorielles
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Remplissage par une valeur constante
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Transformation combinée
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ])

        # Transformation des données
        X = preprocessor.fit_transform(X)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return X, y, preprocessor, label_encoder
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données : {e}")
        st.error(f"Erreur lors du prétraitement des données : {e}")
        return None, None, None, None

# Création du modèle Keras
# Fonction pour créer le modèle Keras
# Fonction pour créer le modèle Keras
def create_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Remplacer num_classes par le nombre de classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Entraînement et évaluation du modèle
def train_model(X_train, y_train, X_test, y_test):
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        keras_clf = KerasClassifierWrapper(build_fn=create_model, epochs=100, batch_size=64, verbose=1)

        # Utilisation de la validation croisée pour évaluer les performances du modèle
        scores = cross_val_score(keras_clf, X_train_res, y_train_res, cv=5, scoring='accuracy')
        st.write(f"Validation Croisée - Moyenne des Précisions : {scores.mean()}")

        # Entraînement final
        keras_clf.fit(X_train_res, y_train_res)
        y_pred = keras_clf.predict(X_test)

        st.write("**Rapport de Classification :**")
        st.text(classification_report(y_test, y_pred))
        st.write("**Précision :**", accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        st.write("**Matrice de Confusion :**")
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        cmd.plot(ax=ax)
        st.pyplot(fig)
        st.success("Entraînement et évaluation du modèle terminés avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement du modèle : {e}")


# Connexion à la base de données PostgreSQL
def load_data_from_db(user1, password1, host1, port1, database1, user2, password2, host2, port2, database2):
    try:
        # Connexion à la première base de données
        conn1 = psycopg2.connect(
            dbname=database1,
            user=user1,
            password=password1,
            host=host1,
            port=port1
        )
        # Connexion à la deuxième base de données
        conn2 = psycopg2.connect(
            dbname=database2,
            user=user2,
            password=password2,
            host=host2,
            port=port2
        )
        return conn1, conn2
    except Exception as e:
        raise Exception(f"Erreur lors de la connexion PostgreSQL : {e}")
# Exécution d'une requête SQL
def execute_query(connection, query):
    try:
        df = pd.read_sql_query(query, connection)
        return df
    except Exception as e:
        raise Exception(f"Erreur lors de l'exécution de la requête SQL : {e}")


# Application principale
def main():
    show_homepage()
    st.title("Application Générale de Prévision")
    st.write("Cette application permet de faire des prévisions basées sur des données SQL personnalisées.")

    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False

    if not st.session_state['data_loaded']:
        st.subheader("Connexion à la Base de Données ou Chargement de Fichier")
        option = st.selectbox("Sélectionner la source des données", ["Se Connecter à une Base de Données", "Télécharger CSV/Excel"])

        if option == "Se Connecter à une Base de Données":
            db_type = st.selectbox("Sélectionner le Type de Base de Données", ["PostgreSQL"])

            if db_type == "PostgreSQL":
                st.header("Connexion aux bases PostgreSQL")

                postgres_host1 = st.text_input("Hôte PostgreSQL (Banque 1)", "197.140.18.127")
                postgres_port1 = st.text_input("Port PostgreSQL (Banque 1)", "6432")
                postgres_user1 = st.text_input("Utilisateur PostgreSQL (Banque 1)", "salam_report")
                postgres_password1 = st.text_input("Mot de passe PostgreSQL (Banque 1)", type="password")
                postgres_database1 = st.text_input("Base de Données PostgreSQL (Banque 1)", "dbsalamprod")

                postgres_host2 = st.text_input("Hôte PostgreSQL (Banque 2)", "197.140.18.127")
                postgres_port2 = st.text_input("Port PostgreSQL (Banque 2)", "6432")
                postgres_user2 = st.text_input("Utilisateur PostgreSQL (Banque 2)", "cpa_report")
                postgres_password2 = st.text_input("Mot de passe PostgreSQL (Banque 2)", type="password")
                postgres_database2 = st.text_input("Base de Données PostgreSQL (Banque 2)", "dbcpaprod")

                if st.button("Se Connecter aux deux bases de données PostgreSQL"):
                    try:
                        # Vérification des mots de passe
                        if not postgres_password1 or not postgres_password2:
                            st.error("Les mots de passe pour les bases de données PostgreSQL sont requis.")
                        else:
                            conn1, conn2 = load_data_from_db(postgres_user1, postgres_password1, postgres_host1,
                                                             postgres_port1, postgres_database1,
                                                             postgres_user2, postgres_password2, postgres_host2,
                                                             postgres_port2, postgres_database2)
                            if conn1 and conn2:
                                st.session_state['conn1'] = conn1
                                st.session_state['conn2'] = conn2
                                st.write("Connexion réussie aux deux bases PostgreSQL")

                    except Exception as e:
                        st.error(f"Erreur lors de la connexion PostgreSQL : {e}")

            if 'conn1' in st.session_state and 'conn2' in st.session_state:
                query = st.text_area("Entrez votre requête SQL pour les données fusionnées",
                                     "SELECT * FROM (SELECT * FROM table1 UNION ALL SELECT * FROM table2) AS merged_data LIMIT 10;")
                if st.button("Exécuter la Requête SQL"):
                    try:
                        df1 = execute_query(st.session_state['conn1'], query)
                        df2 = execute_query(st.session_state['conn2'], query)

                        if df1.empty:
                            st.error("Aucune donnée retournée par la première base de données.")
                        if df2.empty:
                            st.error("Aucune donnée retournée par la deuxième base de données.")

                        df = pd.concat([df1, df2], ignore_index=True)

                        if df.empty:
                            st.error("Les données fusionnées sont vides.")
                        else:
                            st.session_state['dataframe'] = df
                            st.session_state['original_dataframe'] = df.copy()
                            st.session_state['data_loaded'] = True

                            st.write("Aperçu des données :")
                            st.dataframe(df.head())

                    except Exception as e:
                        st.error(f"Erreur lors de l'exécution de la requête SQL : {e}")

        elif option == "Télécharger CSV/Excel":
            st.header("Télécharger Fichier CSV/Excel")
            uploaded_file = st.file_uploader("Choisir un fichier CSV/Excel", type=["csv", "xlsx"])
            if uploaded_file is not None:
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['dataframe'] = df
                st.session_state['data_loaded'] = True
                st.success("Données chargées avec succès.")

    if st.session_state.get('data_loaded', False):
        df = st.session_state['dataframe']

        st.write("Fusionner les colonnes :")
        column_to_merge = st.text_input("Nom de la colonne fusionnée", "merged_column")
        columns_to_merge = st.multiselect("Colonnes à fusionner", df.columns.tolist())

        if column_to_merge and columns_to_merge:
            df_copy = df.copy()
            df_copy[column_to_merge] = df_copy[columns_to_merge].apply(
                lambda row: ', '.join(row.dropna().astype(str)), axis=1)
            df_copy = df_copy.drop(columns=columns_to_merge)
            st.session_state['final_dataframe'] = df_copy

            if 'date_of_birth' in df_copy.columns:
                df_copy['date_of_birth'] = pd.to_datetime(df_copy['date_of_birth'], format='%Y-%m-%d %H:%M:%S',
                                                          errors='coerce')
                invalid_dates = df_copy['date_of_birth'].isna()

                if invalid_dates.any():
                    st.warning(
                        "Certaines valeurs dans la colonne 'date_of_birth' sont invalides et ont été converties en NaT.")
                    st.write(df_copy[invalid_dates])

                valid_dates = df_copy['date_of_birth'].notna()
                if valid_dates.any():
                    today = pd.Timestamp.now()
                    df_copy.loc[valid_dates, 'age'] = today.year - df_copy.loc[valid_dates, 'date_of_birth'].dt.year

            st.write("Données après fusion des colonnes :", df_copy.head())

            if st.checkbox("Afficher les statistiques descriptives"):
                st.write(df_copy.describe())

            if st.checkbox("Afficher les valeurs manquantes avant suppression"):
                st.write("Valeurs manquantes par colonne :")
                st.write(df_copy.isna().sum())

                st.write("Pourcentage de valeurs manquantes par colonne :")
                st.write(df_copy.isna().mean() * 100)

            if st.checkbox("Supprimer les lignes avec des valeurs manquantes"):
                initial_shape = df_copy.shape
                df_cleaned = df_copy.dropna()
                final_shape = df_cleaned.shape

                st.write(f"Nombre initial de lignes : {initial_shape[0]}")
                st.write(f"Nombre de lignes après suppression : {final_shape[0]}")

                if final_shape[0] == 0:
                    st.warning("Le DataFrame est devenu vide après suppression des lignes.")
                else:
                    st.write("Aperçu du DataFrame après suppression des lignes :")
                    st.dataframe(df_cleaned)

                st.session_state['final_dataframe'] = df_cleaned

            if st.checkbox("Gérer les données manquantes"):
                imputer_method = st.selectbox("Méthode d'imputation", ["Moyenne", "Médiane", "KNN"])
                if imputer_method == "Moyenne":
                    df_copy.fillna(df_copy.mean(numeric_only=True), inplace=True)
                elif imputer_method == "Médiane":
                    df_copy.fillna(df_copy.median(numeric_only=True), inplace=True)
                elif imputer_method == "KNN":
                    imputer = KNNImputer(n_neighbors=5)
                    df_copy.iloc[:, :] = imputer.fit_transform(df_copy)
                st.success("Les données manquantes ont été gérées.")
                st.write(df_copy)

                st.session_state['final_dataframe'] = df_copy

            if st.checkbox("Supprimer les colonnes avec des valeurs manquantes"):
                threshold = st.slider("Seuil de valeurs manquantes par colonne (%)", 0, 100, 50) / 100.0
                df_cleaned = df_copy.loc[:, df_copy.isna().mean() < threshold]
                st.write("Aperçu du DataFrame après suppression des colonnes :")
                st.dataframe(df_cleaned)

                st.session_state['final_dataframe'] = df_cleaned

            if not st.session_state['final_dataframe'].empty:
                st.write("Colonnes disponibles :", st.session_state['final_dataframe'].columns.tolist())

                st.subheader("Prétraitement des Données")
                st.write("Effectuer les étapes nécessaires pour nettoyer et préparer les données.")

                # Sélection des colonnes numériques et catégorielles
                num_features = st.multiselect("Sélectionnez les colonnes numériques",
                                              st.session_state['final_dataframe'].select_dtypes(
                                                  include=np.number).columns)
                cat_features = st.multiselect("Sélectionnez les colonnes catégorielles",
                                              st.session_state['final_dataframe'].select_dtypes(
                                                  include=['object', 'category']).columns)
                target_column = st.selectbox("Sélectionner la colonne cible",
                                             st.session_state['final_dataframe'].columns)
                st.write("Colonne cible sélectionnée : ", target_column)

                # Prétraitement des données en utilisant la table nettoyée
                X, y, preprocessor, label_encoder = preprocess_data(
                    st.session_state['final_dataframe'], num_features, cat_features, target_column
                )



                # Après avoir prétraité les données
                if X is not None and y is not None:
                    # Séparation des données en ensembles d'entraînement et de test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Appliquer SMOTE uniquement sur les données d'entraînement
                    smote = SMOTE(random_state=42)
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

                    # Récupérer les dimensions d'entrée et le nombre de classes
                    input_dim = X_train_res.shape[1]
                    num_classes = len(np.unique(y_train_res))

                    # Créer une instance du modèle Keras
                    keras_clf = KerasClassifierWrapper(build_fn=lambda: create_model(input_dim, num_classes),
                                                       epochs=100, batch_size=64, verbose=1)

                    # Validation croisée pour évaluer les performances
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cross_val_scores = []

                    for train_index, test_index in skf.split(X_train_res, y_train_res):
                        X_train_cv, X_test_cv = X_train_res[train_index], X_train_res[test_index]
                        y_train_cv, y_test_cv = y_train_res[train_index], y_train_res[test_index]

                        try:
                            keras_clf.fit(X_train_cv, y_train_cv)
                            score = keras_clf.score(X_test_cv, y_test_cv)
                            cross_val_scores.append(score)
                        except Exception as e:
                            st.warning(f"Erreur lors de l'évaluation croisée : {e}")
                            cross_val_scores.append(np.nan)

                    st.write(
                        f"Validation Croisée - Moyenne des Précisions : {np.nanmean(cross_val_scores)} (scores : {cross_val_scores})")

                    # Entraînement final du modèle
                    keras_clf.fit(X_train_res, y_train_res)

                    # Prédiction sur l'ensemble de test
                    y_pred = keras_clf.predict(X_test)

                    # Évaluation du modèle
                    st.write("**Rapport de Classification :**")
                    st.text(classification_report(y_test, y_pred))
                    st.write("**Précision :**", accuracy_score(y_test, y_pred))

                    # Affichage de la matrice de confusion
                    cm = confusion_matrix(y_test, y_pred)
                    st.write("**Matrice de Confusion :**")
                    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
                    fig, ax = plt.subplots()
                    cmd.plot(ax=ax)
                    st.pyplot(fig)

                    # Analyse des erreurs
                    errors = np.where(y_pred != y_test)[0]


                    # Courbes ROC pour chaque classe
                    fpr = {}
                    tpr = {}
                    roc_auc = {}
                    for i in range(len(np.unique(y_test))):
                        fpr[i], tpr[i], _ = roc_curve(y_test, keras_clf.predict_proba(X_test)[:, i], pos_label=i)
                        roc_auc[i] = auc(fpr[i], tpr[i])

                    plt.figure()
                    for i in range(len(np.unique(y_test))):
                        plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Taux de Faux Positifs')
                    plt.ylabel('Taux de Vrais Positifs')
                    plt.title('Courbes ROC par Classe')
                    plt.legend(loc='lower right')
                    st.pyplot(plt)

                    st.success("Entraînement et évaluation du modèle terminés avec succès.")
                else:
                    st.error("Les données X ou y ne sont pas disponibles.")

                # Charger les données
                df = st.session_state['final_dataframe']

                # Regrouper les données par client
                df_grouped = df.groupby(['client']).agg({
                    'gender': 'first',
                    'age': 'first',
                    'salaire': 'first',
                    'professional_client_category': 'first',
                    'social_client_category': 'first',
                    'educational_level': 'first',
                    'dependent_child': 'first',
                    'level1': lambda x: x.value_counts().index[0]
                }).reset_index()

                # Nettoyer la colonne 'salaire'
                df_grouped['salaire'] = df_grouped['salaire'].replace({r'\s*,\s*': '', r'\.': '', '': np.nan},
                                                                      regex=True)
                df_grouped['salaire'] = df_grouped['salaire'].astype(float).fillna(df_grouped['salaire'].median())

                # Séparer les caractéristiques et la cible
                X = df_grouped.drop(columns=['client', 'level1'])
                y = df_grouped['level1']

                # Diviser les données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Préparer les pipelines pour les caractéristiques numériques et catégorielles
                numeric_features = ['age', 'salaire', 'dependent_child']
                categorical_features = ['gender', 'professional_client_category', 'social_client_category',
                                        'educational_level']

                numeric_transformer = Pipeline(steps=[
                    ('scaler', StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])

                # Construire le pipeline complet avec le modèle
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42))
                ])

                # Entraîner le modèle
                model.fit(X_train, y_train)

                # Obtenir les valeurs uniques pour les options de sélection
                gender_options = df['gender'].unique()
                professional_client_category_options = df['professional_client_category'].unique()
                social_client_category_options = df['social_client_category'].unique()
                educational_level_options = df['educational_level'].unique()

                # Interface utilisateur Streamlit
                st.title('Prédiction pour un nouveau client')

                gender = st.selectbox('Genre', gender_options)
                age = st.slider('Âge', 18, 100, 30)
                salaire = st.number_input('Salaire', min_value=0)
                professional_client_category = st.selectbox('Catégorie Professionnelle',
                                                            professional_client_category_options)
                social_client_category = st.selectbox('Catégorie Sociale', social_client_category_options)
                educational_level = st.selectbox('Niveau Éducatif', educational_level_options)
                dependent_child = st.slider('Enfants Dépendants', 0, 10, 0)

                # Prédiction pour le nouveau client
                new_client = pd.DataFrame([{
                    'gender': gender,
                    'age': age,
                    'salaire': salaire,
                    'professional_client_category': professional_client_category,
                    'social_client_category': social_client_category,
                    'educational_level': educational_level,
                    'dependent_child': dependent_child
                }])

                if st.button('Prédire'):
                    predicted_level1 = model.predict(new_client)
                    st.write("Level1 prévu pour le nouveau client:", predicted_level1[0])


if __name__ == "__main__":
    main()
