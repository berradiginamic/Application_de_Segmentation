import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import io



def show_homepage():
    # Affichage du logo
    st.image("images/logo.png", width=150)

    # Affichage de la bannière
    st.image("images/connexion.png", use_column_width=True)

def main():
    show_homepage()
    st.title("Page de Connexion aux Données")
    st.write("Ceci est la page de connexion aux données.")

    option = st.selectbox("Sélectionner la Source de Données",
                          ["Télécharger CSV/Excel", "Se Connecter à une Base de Données"])

    if option == "Télécharger CSV/Excel":
        uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['dataframe'] = df
                st.session_state['original_dataframe'] = df.copy()
                st.dataframe(df)
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

    elif option == "Se Connecter à une Base de Données":
        db_type = st.selectbox("Sélectionner le Type de Base de Données", ["PostgreSQL", "MySQL", "MongoDB"])

        if db_type == "PostgreSQL":
            st.header("Connexion à la première banque")
            postgres_host1 = st.text_input("Hôte PostgreSQL (Banque 1)", "197.140.18.127")
            postgres_port1 = st.text_input("Port PostgreSQL (Banque 1)", "6432")
            postgres_user1 = st.text_input("Utilisateur PostgreSQL (Banque 1)", "salam_report")
            postgres_password1 = st.text_input("Mot de passe PostgreSQL (Banque 1)", type="password")
            postgres_database1 = st.text_input("Base de Données PostgreSQL (Banque 1)", "dbsalamprod")

            st.header("Connexion à la deuxième banque")
            postgres_host2 = st.text_input("Hôte PostgreSQL (Banque 2)", "197.140.18.127")
            postgres_port2 = st.text_input("Port PostgreSQL (Banque 2)", "6432")
            postgres_user2 = st.text_input("Utilisateur PostgreSQL (Banque 2)", "cpa_report")
            postgres_password2 = st.text_input("Mot de passe PostgreSQL (Banque 2)", type="password")
            postgres_database2 = st.text_input("Base de Données PostgreSQL (Banque 2)", "dbcpaprod")

            if st.button("Se Connecter aux deux bases de données PostgreSQL"):
                try:
                    # Connexion à la première base de données
                    engine1 = create_engine(
                        f'postgresql+psycopg2://{postgres_user1}:{postgres_password1}@{postgres_host1}:{postgres_port1}/{postgres_database1}')
                    conn1 = engine1.connect()

                    # Connexion à la deuxième base de données
                    engine2 = create_engine(
                        f'postgresql+psycopg2://{postgres_user2}:{postgres_password2}@{postgres_host2}:{postgres_port2}/{postgres_database2}')
                    conn2 = engine2.connect()

                    st.session_state['postgres_conn1'] = conn1
                    st.session_state['postgres_conn2'] = conn2

                    # Charger les tables des deux bases de données
                    result1 = conn1.execute(
                        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                    tables1 = [row[0] for row in result1]
                    st.session_state['tables1'] = tables1

                    result2 = conn2.execute(
                        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                    tables2 = [row[0] for row in result2]
                    st.session_state['tables2'] = tables2

                    st.write("Connexions réussies et tables chargées.")
                    st.write(f"Tables disponibles dans la Banque 1 : {tables1}")
                    st.write(f"Tables disponibles dans la Banque 2 : {tables2}")
                except Exception as e:
                    st.error(f"Erreur lors de la connexion PostgreSQL : {e}")

        # Si les connexions sont établies
        if 'postgres_conn1' in st.session_state and 'postgres_conn2' in st.session_state:
            st.write("Connexions PostgreSQL établies.")
            tables1 = st.session_state.get('tables1', [])
            tables2 = st.session_state.get('tables2', [])

            if tables1 and tables2:
                selected_tables1 = st.multiselect("Sélectionner les Tables de la Banque 1", tables1)
                selected_tables2 = st.multiselect("Sélectionner les Tables de la Banque 2", tables2)

                if selected_tables1 and selected_tables2 and len(selected_tables1) == len(selected_tables2):
                    try:
                        conn1 = st.session_state['postgres_conn1']
                        conn2 = st.session_state['postgres_conn2']

                        final_dfs = []
                        all_columns = []

                        for table1, table2 in zip(selected_tables1, selected_tables2):
                            df1 = pd.read_sql(text(f'SELECT * FROM "{table1}"'), conn1)
                            df2 = pd.read_sql(text(f'SELECT * FROM "{table2}"'), conn2)

                            combined_df = pd.concat([df1, df2], ignore_index=True)
                            final_dfs.append(combined_df)
                            all_columns.extend(combined_df.columns.tolist())

                            st.write(f"Table combinée: {table1} + {table2}")
                            st.dataframe(combined_df)

                        # Sélection des colonnes après concaténation
                        st.write("Sélectionner les colonnes des tables combinées pour la table finale :")
                        selected_columns = st.multiselect("Colonnes disponibles", list(set(all_columns)),
                                                          default=list(set(all_columns)))

                        # Combiner tous les DataFrames concaténés en un seul
                        if len(final_dfs) > 1:
                            final_df = pd.concat(final_dfs, ignore_index=True)
                        else:
                            final_df = final_dfs[0]

                        # Filtrer les colonnes sélectionnées
                        final_df = final_df[selected_columns]

                        st.write("Fusionner les colonnes :")
                        column_to_merge = st.text_input("Nom de la colonne fusionnée", "merged_column")
                        columns_to_merge = st.multiselect("Colonnes à fusionner", final_df.columns.tolist())

                        if column_to_merge and columns_to_merge:
                            # Fusionner les colonnes sélectionnées
                            final_df[column_to_merge] = final_df[columns_to_merge].apply(
                                lambda row: ', '.join(row.dropna().astype(str)), axis=1)
                            final_df = final_df.drop(columns=columns_to_merge)
                            st.write("Table après fusion des colonnes :")
                            st.dataframe(final_df)

                        st.session_state['final_dataframe'] = final_df

                        # Boutons de téléchargement
                        csv = final_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Télécharger le tableau combiné en CSV",
                            data=csv,
                            file_name='combined_tables.csv',
                            mime='text/csv'
                        )

                        towrite = io.BytesIO()
                        final_df.to_excel(towrite, index=False, engine='xlsxwriter')
                        towrite.seek(0)
                        st.download_button(
                            label="Télécharger le tableau combiné en Excel",
                            data=towrite,
                            file_name='combined_tables.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )

                    except Exception as e:
                        st.error(f"Erreur lors de la lecture des tables ou de la fusion : {e}")
                else:
                    st.warning("Veuillez sélectionner un nombre équivalent de tables dans chaque banque.")
            else:
                st.write("Aucune table trouvée dans l'une des bases de données.")


if __name__ == "__main__":
    if 'dataframe' not in st.session_state:
        st.session_state['dataframe'] = None
    if 'original_dataframe' not in st.session_state:
        st.session_state['original_dataframe'] = None
    if 'final_dataframe' not in st.session_state:
        st.session_state['final_dataframe'] = None
    if 'tables1' not in st.session_state:
        st.session_state['tables1'] = []
    if 'tables2' not in st.session_state:
        st.session_state['tables2'] = []
    if 'postgres_conn1' not in st.session_state:
        st.session_state['postgres_conn1'] = None
    if 'postgres_conn2' not in st.session_state:
        st.session_state['postgres_conn2'] = None
    main()
