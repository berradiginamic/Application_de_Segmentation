import streamlit as st

def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/logo1.jpg", use_column_width=True)

    # Titre de la page principale
    st.markdown('<h1 class="title">DATA DIGITAL ONE</h1>', unsafe_allow_html=True)

    # Sous-titre
    st.markdown('<h2 class="subtitle">Together, we combine Data and Digital</div>', unsafe_allow_html=True)

    st.write("""
        La segmentation des clients est une tâche fondamentale en marketing et en gestion de la relation client. 
        Avec les avancées en analyse de données et en apprentissage automatique, il est maintenant possible de 
        regrouper les clients en segments distincts avec une grande précision, permettant aux entreprises d'adapter 
        leurs stratégies marketing et leurs offres aux besoins et préférences uniques de chaque segment.

        **Problème/Requis** : Utiliser des techniques d'apprentissage automatique et d'analyse de données en Python pour effectuer la segmentation des clients.
        """)

    # Section sur l'utilisation de l'application avec des icônes
    st.markdown("""
        <div class="instruction">
            <p>Cette application propose des fonctionnalités pour :</p>
            <ul>
                <li><span class="icon">🔗</span>Connexion aux Données : Téléchargez un fichier CSV, Excel ou connectez-vous à une base de données.</li>
                <li><span class="icon">📝</span>Description des Données : Obtenez un résumé et une vue d'ensemble de vos de données.</li>
                <li><span class="icon">📈</span>Analyse des Données : Effectuez une analyse exploratoire des données.</li>
                <li><span class="icon">🏋️</span>Entraînement du Modèle : Entraînez vos modèles d'apprentissage automatique.</li>
                <li><span class="icon">📊</span>Evaluation modèle avancée : Déterminer le nombre optimal de clusters avec l'analyse de la silhouette.</li>
                <li><span class="icon">📉</span>Segmentation prévision : Visualisez les résultats de prediction.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def main():
    """
    Fonction principale pour afficher l'application Streamlit avec une barre de navigation.
    """
    show_homepage()  # Appel de la fonction pour afficher la page d'accueil

if __name__ == "__main__":
    if 'dataframe' not in st.session_state:
        st.session_state['dataframe'] = None
    if 'original_dataframe' not in st.session_state:
        st.session_state['original_dataframe'] = None
    if 'final_dataframe' not in st.session_state:
        st.session_state['final_dataframe'] = None
    main()
