import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder


def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/model.jpg", use_column_width=True)
def fetch_data():
    if 'final_dataframe' in st.session_state:
        return st.session_state['final_dataframe']
    else:
        st.warning("Aucun dataframe trouvé dans l'état de session.")
        return None

def preprocess_data(df, selected_features, target_column):
    # Ensure the selected features and target column exist in the DataFrame
    if not all([col in df.columns for col in selected_features]):
        raise ValueError("Certaines caractéristiques sélectionnées ne sont pas présentes dans le dataframe.")
    if target_column not in df.columns:
        raise ValueError("La colonne cible sélectionnée n'est pas présente dans le dataframe.")

    df = df[selected_features + [target_column]].copy()

    # Fill missing values for numeric columns with the mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

    # Ensure selected features after encoding
    selected_features_encoded = [col for col in df_encoded.columns if col in df_encoded.columns]

    X = df_encoded[selected_features_encoded]
    y = df_encoded[target_column]

    # Debugging output
    st.write("Données après encodage et prétraitement :")
    st.dataframe(df_encoded.head())
    st.write(f"Dimensions de X: {X.shape}, Dimensions de y: {y.shape}")

    return X, y

def main():
    st.title("Évaluation et Entraînement des Modèles de Machine Learning")

    df = fetch_data()

    if df is not None:
        st.header("Prétraitement des Données")

        # Select features and target column
        selected_features = st.multiselect("Sélectionner les caractéristiques à inclure dans le modèle",
                                           df.columns.tolist())
        target_column = st.selectbox("Sélectionner la colonne cible", df.columns.tolist())

        if selected_features and target_column:
            try:
                X, y = preprocess_data(df, selected_features, target_column)

                st.header("Sélection et Personnalisation du Modèle")

                # Model selection
                model_name = st.selectbox("Sélectionner le Modèle", ["Régression Linéaire", "Régression Logistique",
                                                                     "Arbre de Décision", "SVM", "Naive Bayes",
                                                                     "Forêt Aléatoire"])

                # Model initialization with parameter customization
                if model_name == "Régression Linéaire":
                    fit_intercept = st.checkbox("Ajuster l'intercept", value=True)
                    model = LinearRegression(fit_intercept=fit_intercept)
                elif model_name == "Régression Logistique":
                    C = st.slider("Force de régularisation (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    model = LogisticRegression(C=C, max_iter=1000)
                elif model_name == "Arbre de Décision":
                    criterion = st.selectbox("Critère", ["gini", "entropy"])
                    max_depth = st.slider("Profondeur Maximale", min_value=1, max_value=20, value=3, step=1)
                    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
                elif model_name == "SVM":
                    C = st.slider("Paramètre de régularisation (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    kernel = st.selectbox("Noyau", ["linéaire", "poly", "rbf", "sigmoid"])
                    model = SVC(C=C, kernel=kernel)
                elif model_name == "Naive Bayes":
                    model = GaussianNB()
                elif model_name == "Forêt Aléatoire":
                    n_estimators = st.slider("Nombre d'estimateurs", min_value=10, max_value=200, value=100, step=10)
                    max_depth = st.slider("Profondeur Maximale", min_value=1, max_value=20, value=None, step=1)
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

                if st.button("Entraîner le Modèle"):
                    if len(selected_features) == 0:
                        st.warning("Veuillez sélectionner au moins une caractéristique.")
                        return
                    if not target_column:
                        st.warning("Veuillez sélectionner une colonne cible.")
                        return

                    # Train the model
                    model.fit(X, y)
                    y_pred = model.predict(X)

                    # Model evaluation
                    if isinstance(model, LinearRegression):
                        score = r2_score(y, y_pred)
                        st.write(f"Score R²: {score:.2f}")
                        st.write("""
                        **Interprétation du Score R²**: Le score R² mesure à quel point le modèle de régression linéaire s'adapte aux données. 
                        Une valeur proche de 1.0 indique que le modèle explique une grande proportion de la variance de la variable cible. 
                        Si le score R² est faible, cela peut indiquer que le modèle ne capture pas bien les tendances des données.
                        """)
                    else:
                        accuracy = accuracy_score(y, y_pred)
                        st.write(f"Précision: {accuracy:.2f}")
                        st.write("""
                        **Interprétation de la Précision**: La précision mesure le pourcentage d'instances correctement prédites parmi toutes les instances.
                        Une précision élevée indique que le modèle est efficace pour classer les instances correctement. 
                        Une faible précision peut suggérer que le modèle a du mal à faire des prédictions précises ou que les données sont mal équilibrées.
                        """)

                    # Save the model
                    filename = f"{model_name}_model.pkl"
                    with open(filename, 'wb') as file:
                        pickle.dump(model, file)
                    st.write(f"Modèle entraîné et sauvegardé sous {filename}")

                # Evaluation and display of the best model
                st.header("Évaluation des Modèles")

                # Initialization of models for evaluation
                models = {
                    "Régression Linéaire": LinearRegression(),
                    "Régression Logistique": LogisticRegression(max_iter=1000),
                    "Arbre de Décision": DecisionTreeClassifier(),
                    "SVM": SVC(),
                    "Naive Bayes": GaussianNB(),
                    "Forêt Aléatoire": RandomForestClassifier()
                }

                best_model = None
                best_score = -1
                best_model_name = ""

                for model_name, model in models.items():
                    st.subheader(f"Évaluation du {model_name}")

                    # Data splitting
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                    # Model training and evaluation
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if isinstance(model, LinearRegression):
                        score = r2_score(y_test, y_pred)
                        st.write(f"Score R²: {score:.2f}")
                        st.write("""
                        **Interprétation du Score R²**: Le score R² mesure l'ajustement du modèle de régression linéaire aux données test. 
                        Un score plus élevé indique une meilleure capacité du modèle à expliquer la variance de la variable cible sur les données de test.
                        """)
                    else:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Précision: {accuracy:.2f}")
                        st.write("""
                        **Interprétation de la Précision**: La précision mesure le taux de classification correcte du modèle sur les données test. 
                        Une précision plus élevée indique une meilleure performance du modèle sur des données invisibles. 
                        Une précision plus faible peut signaler des problèmes de surapprentissage ou de sous-apprentissage.
                        """)

                    # Tracking the best model
                    if (isinstance(model, LinearRegression) and score > best_score) or \
                            (not isinstance(model, LinearRegression) and accuracy > best_score):
                        best_score = score if isinstance(model, LinearRegression) else accuracy
                        best_model = model
                        best_model_name = model_name

                if best_model:
                    st.header("Meilleur Modèle")
                    st.write(f"Le meilleur modèle est: {best_model_name}")
                    st.write(f"Avec un score de: {best_score:.2f}")
                    st.write("""
                    **Interprétation du Meilleur Modèle**: Le meilleur modèle est celui avec la meilleure performance (score R² pour la régression linéaire et précision pour les autres modèles). 
                    Ce modèle a montré la capacité la plus élevée à prédire les valeurs cibles sur les données test. 
                    Assurez-vous que ce modèle est bien adapté aux données et qu'il n'y a pas de biais ou de surapprentissage.
                    """)

            except ValueError as e:
                st.error(f"Erreur de prétraitement: {e}")

if __name__ == "__main__":
    show_homepage()
    main()
