import streamlit as st
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/cluster.jpg", use_column_width=True)

# Récupération des données
def fetch_data():
    if 'final_dataframe' in st.session_state:
        return st.session_state['final_dataframe']
    else:
        st.warning("Aucun dataframe trouvé dans l'état de session.")
        return None


# Prétraitement des données pour le clustering
def preprocess_for_clustering(df, features):
    available_columns = df.columns.tolist()
    missing_columns = [feature for feature in features if feature not in available_columns]
    if missing_columns:
        st.error(f"Les colonnes suivantes sont manquantes dans le DataFrame: {missing_columns}")
        return None
    df_clustering = df[features].copy()
    df_encoded = pd.get_dummies(df_clustering)
    return df_encoded


# Aligner les colonnes pour la prédiction
def align_columns(new_data, model_df):
    new_data = new_data.reindex(columns=model_df.columns, fill_value=0)
    return new_data


# Méthode du coude pour déterminer le nombre optimal de clusters
def elbow_method(df, features):
    st.write("#### Méthode du Coude pour Déterminer le Nombre Optimal de Clusters")
    X = preprocess_for_clustering(df, features)
    if X is None:
        return
    sse = {}
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        try:
            kmeans.fit(X)
            sse[k] = kmeans.inertia_
        except ValueError as e:
            st.error(f"Erreur lors de l'entraînement du modèle KMeans avec {k} clusters: {e}")
            return
    fig, ax = plt.subplots()
    ax.plot(list(sse.keys()), list(sse.values()), marker='o')
    ax.set_xlabel("Nombre de clusters")
    ax.set_ylabel("SSE")
    ax.set_title("Méthode du Coude", fontweight='bold')
    st.pyplot(fig)


# Afficher la matrice de corrélation
def plot_correlation_matrix(df, features):
    st.write("#### Matrice de Corrélation des Caractéristiques")
    df_corr = df[features].copy()
    df_encoded = pd.get_dummies(df_corr)
    corr = df_encoded.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title("Matrice de Corrélation", fontweight='bold')
    st.pyplot(fig)


# Entraînement du modèle KMeans et évaluation avec visualisation 3D
def kmeans_clustering(df, features, n_clusters):
    X = preprocess_for_clustering(df, features)
    if X is None:
        return
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['kmeans_model'] = model
    st.session_state['features_for_clustering'] = features

    st.write(f"**Nombre de clusters sélectionné : {n_clusters}**")
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    # Compter le nombre de clients dans chaque cluster
    cluster_counts = df['Cluster'].value_counts().sort_values(ascending=False)
    top_clusters = cluster_counts.head(3).index.tolist()

    st.write("#### Top 3 Clusters avec le plus grand nombre de clients")
    st.write(cluster_counts.head(3))

    cluster_explanations = []
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    # Réduction de dimension à 3 composantes principales pour la visualisation 3D
    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = df['Cluster']

    # Visualisation 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], c=df_pca['Cluster'], cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title("Visualisation des Clusters avec PCA en 3D (KMeans)")
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    st.pyplot(fig)

    # Interprétation graphique des top 3 clusters
    for cluster in top_clusters:
        st.write(f"### Interprétation du Cluster {cluster}")
        cluster_data = df[df['Cluster'] == cluster]

        # Visualiser les distributions des caractéristiques dans le cluster
        fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(15, 4))
        for i, feature in enumerate(features):
            sns.histplot(cluster_data[feature], bins=20, ax=axes[i], kde=True, color='skyblue')
            axes[i].set_title(f'{feature} dans Cluster {cluster}')
            axes[i].set_xlabel('')
        plt.tight_layout()
        st.pyplot(fig)

    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    st.success("Modèle KMeans entraîné et enregistré avec succès!")


# Entraînement du modèle Agglomerative Clustering et évaluation
def agglomerative_clustering(df, features, n_clusters):
    X = preprocess_for_clustering(df, features)
    if X is None:
        return
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['agglomerative_model'] = model

    st.write(f"**Nombre de clusters sélectionné : {n_clusters}**")
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (Agglomerative Clustering)")
    st.pyplot(fig)


# Entraînement du modèle DBSCAN et évaluation
def dbscan_clustering(df, features, eps, min_samples):
    X = preprocess_for_clustering(df, features)
    if X is None:
        return
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['dbscan_model'] = model

    num_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
    st.write(f"**Nombre de clusters détectés : {num_clusters}**")

    if num_clusters > 1:
        silhouette_avg = silhouette_score(X, df['Cluster'])
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Cluster {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Cluster {cluster}** : "
        explanation += f"Ce cluster contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (DBSCAN)")
    st.pyplot(fig)


# Entraînement du modèle Gaussian Mixture Model et évaluation
def gmm_clustering(df, features, n_components):
    X = preprocess_for_clustering(df, features)
    if X is None:
        return
    model = GaussianMixture(n_components=n_components, random_state=42)
    df['Cluster'] = model.fit_predict(X)
    st.session_state['df_transformed'] = df
    st.session_state['gmm_model'] = model

    st.write(f"**Nombre de composants (clusters) : {n_components}**")
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    cluster_explanations = []
    for cluster in range(n_components):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"Composant {cluster} - Données")
        st.write(cluster_data.head())
        explanation = f"- **Composant {cluster}** : "
        explanation += f"Ce composant contient {len(cluster_data)} clients. "
        explanation += f"Moyenne des valeurs des features : "
        numeric_cols = cluster_data.select_dtypes(include=['number']).columns
        explanation += ", ".join([f"{col}: {cluster_data[col].mean():.2f}" for col in numeric_cols])
        cluster_explanations.append(explanation)

    for explanation in cluster_explanations:
        st.write(explanation)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df['Cluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
    ax.set_title("Visualisation des Clusters avec PCA (GMM)")
    st.pyplot(fig)


# Ajouter un nouveau client et prédire son cluster
def add_and_predict_client(df, features):
    st.write("### Ajouter un Nouveau Client")

    # Dictionnaire pour stocker les entrées de l'utilisateur
    new_client = {}

    # Boucle pour afficher des champs d'entrée basés sur les variables réelles (non encodées)
    for feature in features:
        feature_type = df[feature].dtype
        if feature_type == 'object':  # Si c'est une variable catégorielle
            unique_values = df[feature].unique()
            new_client[feature] = st.selectbox(f"Sélectionner {feature}", unique_values)
        else:  # Si c'est une variable numérique
            new_client[feature] = st.number_input(f"Entrer {feature}", value=float(df[feature].mean()))

    if st.button("Prédire le Cluster"):
        # Convertir l'entrée utilisateur en DataFrame
        new_client_df = pd.DataFrame([new_client])

        # Prétraitement des nouvelles données client (encodage)
        new_client_encoded = pd.get_dummies(new_client_df)

        # Aligner les colonnes du nouveau client avec les colonnes du modèle
        X_encoded = preprocess_for_clustering(df, features)
        new_client_encoded_aligned = align_columns(new_client_encoded, X_encoded)

        # Prédiction du cluster
        if 'kmeans_model' in st.session_state:
            model = st.session_state['kmeans_model']
            predicted_cluster = model.predict(new_client_encoded_aligned)[0]
            st.success(f"Le nouveau client appartient au cluster: {predicted_cluster}")
        else:
            st.error("Le modèle KMeans n'est pas disponible pour la prédiction. Entraînez le modèle d'abord.")

def get_encoding_explanations(df, features):
    explanations = {}
    for feature in features:
        if df[feature].dtype == 'object':
            encoding_dict = {i: category for i, category in enumerate(df[feature].astype('category').cat.categories)}
            explanations[feature] = encoding_dict
    return explanations

def main():
    st.title("Analyse de Clustering des Clients")
    data = fetch_data()
    if data is not None:
        st.write("## Exploration des Données")
        st.dataframe(data.head())

        features = st.multiselect(
            "Sélectionnez les fonctionnalités à inclure dans le clustering",
            options=data.columns.tolist(),
            default=data.columns.tolist()
        )

        if features:
            st.write("## Analyse de Clustering")
            elbow_method(data, features)
            plot_correlation_matrix(data, features)

            # Afficher les explications de l'encodage
            encodings = get_encoding_explanations(data, features)
            st.write("## Explication des Encodages")
            for feature, encoding_dict in encodings.items():
                st.write(f"### {feature}")
                st.write("Chaque chiffre représente :")
                for code, category in encoding_dict.items():
                    st.write(f"- {code} : {category}")

            algo = st.selectbox("Choisissez l'algorithme de clustering", ["KMeans", "Agglomerative", "DBSCAN", "GMM"])
            if algo == "KMeans":
                n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=3)
                kmeans_clustering(data, features, n_clusters)
            elif algo == "Agglomerative":
                n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=3)
                agglomerative_clustering(data, features, n_clusters)
            elif algo == "DBSCAN":
                eps = st.slider("Epsilon", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
                min_samples = st.slider("Min Samples", min_value=1, max_value=20, value=5)
                dbscan_clustering(data, features, eps, min_samples)
            elif algo == "GMM":
                n_components = st.slider("Nombre de composants", min_value=1, max_value=10, value=3)
                gmm_clustering(data, features, n_components)

            add_and_predict_client(data, features)
        else:
            st.warning("Veuillez sélectionner au moins une fonctionnalité pour effectuer le clustering.")


if __name__ == "__main__":
    show_homepage()
    main()
