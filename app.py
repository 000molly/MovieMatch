###Biblio
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import base64
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from datetime import datetime

#init streamlit
st.set_page_config(page_title="MovieMatch", page_icon="🎦", layout = "wide")

###Variables
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)
df_full = load_data(r"C:\Users\eliot\Desktop\Projet_2\full_bdd.csv")
dico_film = load_data(r"C:\Users\eliot\Desktop\Projet_2\dico_films.csv")
film_genre = load_data(r"C:\Users\eliot\Desktop\Projet_2\Streamlit\BDD utiles\film_par_genre.csv")
tmdb = load_data(r"C:\Users\eliot\Desktop\Projet_2\Streamlit\BDD utiles\tmdb.csv")
titleB = load_data(r"C:\Users\eliot\Desktop\Projet_2\Streamlit\BDD utiles\titleB_&_rating.csv")
plotly_tab = load_data(r"C:\Users\eliot\Desktop\Projet_2\Streamlit\BDD utiles\plotly_tab.csv")
movie_countries_total = load_data(r"C:\Users\eliot\Desktop\Projet_2\Streamlit\BDD utiles\movie_countries_total.csv")
film_vs_acteur = load_data(r"C:\Users\eliot\Desktop\Projet_2\Streamlit\BDD utiles\film_vs_acteur.csv")

#filtres
dico_film.dropna(subset=['poster_path'], inplace=True) #supprime les films sans posters
dico_film['poster_path']=dico_film['poster_path'].astype(str)
df_full = df_full[df_full['startYear'] > 2015]
df_full = df_full[df_full['primaryTitle'].isin(dico_film['primaryTitle'])]
df = df_full.copy()
roles_to_keep = ['actor', 'actress', 'director']
df = df[df['role'].isin(roles_to_keep)]
df.drop(columns=["numVotes", "runtimeMinutes", "primaryName", "birthYear",
                 "deathYear", "role", "averageRating"], inplace=True)
genres = ['Documentary', 'Drama', 'Action', 'Crime', 'Biography',
          'Adventure','Animation', 'Comedy', 'Horror', 'Fantasy',
          'Family', 'Mystery', 'Sci-Fi', 'Thriller', 'Romance',
          'Musical', 'War', 'Music', 'History', 'News', 'Sport',
          'Western', 'Adult', 'Reality-TV', 'Talk-Show', 'Game-Show']
rename_dict = {genre: f'Genre_{genre}' for genre in genres}


###Fonctions
def sidebar_bg(side_bg):
   side_bg_ext = 'png'
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
def get_poster(poster_path):
    base_url = "https://image.tmdb.org/t/p/w342/"
    full_url = base_url + poster_path
    return full_url

@st.cache_data
def resize(url, size=(342, 513)):
    res = requests.get(url)
    img = Image.open(BytesIO(res.content))
    img = img.resize(size, Image.LANCZOS)
    return img

def actor_column(df, film_title):
    specific_film_df = df[df['primaryTitle'] == film_title]
    specific_actors = specific_film_df['actorID'].unique()
    for actor in specific_actors:
        df[f'Actor_{actor}'] = (df['actorID'] == actor).astype(int)
    return specific_actors

@st.cache_data
def reco(film_title):
    #ml
    specific_actors = actor_column(df, film_title)
    genre_columns = [col for col in df.columns if col.startswith('Genre_')]
    actor_columns = [f'Actor_{actor}' for actor in specific_actors]
    feature_columns = genre_columns + actor_columns

    grouped_df = df.groupby('primaryTitle')[feature_columns].max().reset_index()

    scaler = StandardScaler()
    X = scaler.fit_transform(grouped_df[feature_columns])

    nn_model = NearestNeighbors(n_neighbors=6)
    nn_model.fit(X)

    film_index = grouped_df[grouped_df['primaryTitle'] == film_title].index[0]
    distances, indices = nn_model.kneighbors([X[film_index]])
    nearest_neighbors_indices = indices[0]

    nearest_neighbors_films = grouped_df.iloc[nearest_neighbors_indices].reset_index(drop=True)
    return nearest_neighbors_films['primaryTitle'][1:]


###Streamlit
st.markdown("<h2 style='text-align: center;'>MovieMatch</h2>", unsafe_allow_html=True)
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(90deg, RGB(22, 36, 84), RGB(109, 127, 234));
    }
</style>""",
unsafe_allow_html=True)
side_bg = r"C:\Users\eliot\Desktop\Projet_2\Capture d'écran 2024-05-24 095024.png"
sidebar_bg(side_bg)

st.sidebar.title('Navigation')
page = st.sidebar.radio('', ('Recommendé pour vous', 'Nouveautés', 'Quelques statistiques', 'Histoire du cinéma'))

if page == 'Recommendé pour vous':
    film_list = df['primaryTitle'].unique().tolist()
    selected_film = st.selectbox('Sélectionner le dernier film que vous avez aimé :', film_list, placeholder='Choisissez un film')
    #st.selectbox('Select one option:', ['', 'First one', 'Second one'], format_func=lambda x: 'Select an option' if x == '' else x)
    if selected_film:
        recommandation = reco(selected_film)
        st.write('Nos recommandations :')

        cols = st.columns(5)
        for index, film in enumerate(recommandation):
            film_info = dico_film[dico_film['primaryTitle'] == film]
            if not film_info.empty:
                path = film_info['poster_path'].iloc[0]
                url = get_poster(path)
                img = resize(url)
                with cols[index % 5]:
                    cols[index % 5].image(img, width=270, output_format="PNG")
                    cols[index % 5].write(film)
            else:
                cols[index % 5].write(film)

elif page == 'Nouveautés':
    st.title('Nouveautés')

    df_full['startYear'] = pd.to_datetime(df_full['startYear'], format='%Y')

    #filtrer les films les plus récents qui sont déjà sortis
    today = pd.to_datetime('today')
    recent_films = df_full[df_full['startYear'] <= today].sort_values(by='startYear', ascending=False).drop_duplicates('FilmID').head(5)
    
    #afficher les films
    cols = st.columns(5)
    for index, film in enumerate(recent_films.iterrows()):
        film_info = dico_film[dico_film['primaryTitle'] == film[1]['primaryTitle']] #film[1] est la ligne actuelle du df
        if not film_info.empty:
            path = film_info['poster_path'].iloc[0]
            url = get_poster(path)
            img = resize(url)
            #img = resize(get_poster(film_info['poster_path'].iloc[0]))
            with cols[index]:
                cols[index].image(img, width=270, output_format="PNG")
                cols[index].write(film[1]['primaryTitle'])
        else:
            with cols[index]:
                cols[index].write(film[1]['primaryTitle'])

elif page == 'Quelques statistiques':
    st.title("Comment l'industrie du cinéma a t'elle évoluée au fil des années ?")

    # visualisation statitique des pays producteurs de film toutes années confondues
    fig1 = px.scatter_geo(movie_countries_total,
                        lat='latitude',
                        lon='longitude',
                        size='nb_film',
                        size_max=80,
                        hover_name='LABEL EN',
                        color_discrete_sequence=['red'],
                        width=1200,
                        height=800)

    fig1.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))

    st.header("Nombre de films produits par pays toutes années confondues", divider="blue")
    st.plotly_chart(fig1)

    #limiter le data set au 10 premiers
    top10 = movie_countries_total.sort_values('nb_film', ascending=False).head(10)

    #chart top 10
    fig2 = px.funnel(top10, x='nb_film', y='LABEL EN', width=800, height=600)
    fig2.update_traces(marker_color='#CD5C5C')
    st.header("TOP 10 des pays produisant le plus de films", divider="blue")
    st.plotly_chart(fig2)

    #construction data set groupé pour 1er chart
    from datetime import datetime
    tmdb['release_date'] = pd.to_datetime(tmdb['release_date'])

    tmdb = tmdb.drop(columns='status')
    titleB = titleB.drop(columns='primaryTitle')

    numeric_t = pd.merge(film_genre, tmdb, how='left', left_on='FilmID', right_on='FilmID')
    numeric_t = pd.merge(numeric_t, titleB, how='left', left_on='FilmID', right_on='FilmID')

    #créer 2 colonnes Mois et Année
    numeric_t['MONTH'] = numeric_t['release_date'].dt.month
    numeric_t['YEAR'] = numeric_t['release_date'].dt.year

    genres_evo = numeric_t.groupby("YEAR")["Drama"].sum()
    genres_evo = genres_evo.reset_index()

    comedy_evo = numeric_t.groupby("YEAR")["Comedy"].sum()
    comedy_evo = comedy_evo.reset_index()

    genres_evo = pd.merge(genres_evo, comedy_evo, how='left', left_on='YEAR', right_on='YEAR')

    #boucle pour faire les autres genres
    genres = ["Documentary","Action","Crime","Biography","Adventure","Animation","Horror","Fantasy","Family",
    "Mystery","Sci-Fi","Thriller","Romance","Musical","War","Music","History","News","Sport","Western","Adult","Reality-TV","Talk-Show","Game-Show"]

    for genre in genres:
        evo_g = numeric_t.groupby("YEAR")[genre].sum()
        evo_g = evo_g.reset_index()
        genres_evo = pd.merge(genres_evo, evo_g, how='left', left_on='YEAR', right_on='YEAR')

    #chart evolution des genres au fil des années
    genres = ["Drama","Comedy","Documentary","Action","Crime","Biography","Adventure","Animation","Horror","Fantasy","Family",
    "Mystery","Sci-Fi","Thriller","Romance","Musical","War","Music","History","News","Sport","Western","Adult","Reality-TV","Talk-Show","Game-Show"]

    fig = px.bar(genres_evo, x="YEAR", y=genres, range_x = [1920, 2023], width=800, height=600)
    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'category descending'})

    st.header("Nombre de films par genre et par année", divider="blue")

    st.plotly_chart(fig, theme="streamlit")

    #4ème chart sur ratio revenu / budget pour TOP 4 des genres de films
    st.header("Corrélation coût/revenu d'un film pour les 4 genres majoritaires", divider="blue")
    col1, col2 = st.columns(2)

    with col1:
        fig5 = px.scatter(plotly_tab, x='Dramaavg_budget', y='Drama_avg_revenue', color="Drama_avg_revenue", size='Drama_avg_revenue', color_continuous_scale='bluered')
        fig5.update_layout(xaxis_range=[0, 4000000], yaxis_range=[0, 10000000]);
        st.subheader("films dramatiques")
        st.plotly_chart(fig5)

        fig6 = px.scatter(plotly_tab, x='Comedyavg_budget', y='Comedy_avg_revenue', color='Comedy_avg_revenue', color_continuous_scale='bluered', size='Comedy_avg_revenue')
        fig6.update_layout(xaxis_range=[0, 4000000], yaxis_range=[0, 10000000]);
        st.subheader("films de comédie")
        st.plotly_chart(fig6)

    with col2:
        fig7 = px.scatter(plotly_tab, x='Documentaryavg_budget', y='Documentary_avg_revenue', color="Documentary_avg_revenue", size='Documentary_avg_revenue', color_continuous_scale='bluered')
        fig7.update_layout(xaxis_range=[0, 4000000], yaxis_range=[0, 10000000]);
        st.subheader("films documentaires")
        st.plotly_chart(fig7)

        fig8 = px.scatter(plotly_tab, x='Actionavg_budget', y='Action_avg_revenue', color="Action_avg_revenue", size='Action_avg_revenue', color_continuous_scale='bluered')
        fig8.update_layout(xaxis_range=[0, 4000000], yaxis_range=[0, 10000000]);
        st.subheader("films d'Action")
        st.plotly_chart(fig8)

    ##AJOUT DECADE DANS DATA SET ACTOR_VS_FILM
    # Convertir la colonne 'year' en type numérique
    film_vs_acteur['year'] = pd.to_numeric(film_vs_acteur['year'])

    # Définir les bins de décennies à partir de 1900 jusqu'à 2030 (pour inclure 2028)
    bins = list(range(1900, 2040, 10))

    # Créer une nouvelle colonne 'decade' qui contient les bins de décennies
    film_vs_acteur['decade'] = pd.cut(film_vs_acteur['year'], bins=bins, labels=bins[:-1], right=False)

    #TOP 10 des acteurs ayant le plus tourné
    nb_film_actor = film_vs_acteur.groupby(['decade', 'category','primaryName']).agg(nb_film= ('FilmID', 'count'))
    nb_film_actor = nb_film_actor.reset_index()
    #enlever les acteurs n'ayant tourné aucun film
    nb_film_actor = nb_film_actor.loc[nb_film_actor['nb_film'] != 0, :]

    #pour la prez alléger le fichier en affichant uniquement les films à partir de 2000
    nb_film_actor_light = nb_film_actor.loc[nb_film_actor['decade'] >= 2000, :]

    #maj ordre à l'affichage
    nb_film_actor_light.sort_values(by=['decade','nb_film'], ascending=[False, False])

    st.header("TOP 10 des acteurs / actrices ayant tourné le plus de films", divider="blue")
    col1, col2 = st.columns(2)
    with col1:
        choix_decade = st.selectbox('Veuillez sélectionner dans la liste ci-dessous (1):', nb_film_actor_light['decade'].unique())

    with col2:
        choix_cat = st.selectbox('Veuillez sélectionner dans la liste ci-dessous (2):', nb_film_actor_light['category'].unique())

    #filtrer le data set selon le choix de l'internaute
    best_actor = nb_film_actor_light.loc[(nb_film_actor_light['decade'] == choix_decade) & (nb_film_actor_light['category'] == choix_cat), :].sort_values(by='nb_film', ascending=False).head(10)

    fig3 = px.funnel(best_actor, x='nb_film', y='primaryName', width=800, height=600)
    fig3.update_traces(marker_color='#008080')
    st.plotly_chart(fig3)

    #dernier chart sur l'évolution de l'age des intervenants à travers les années
    #data set par decade pour moyenne age au moment du film
    avg_age_decade = film_vs_acteur.groupby(['category', 'decade']).agg(avg_age= ('age_at_film', 'mean'))
    avg_age_decade = avg_age_decade.reset_index()

    #pour la prez limiter le data set aux principales categories de job
    avg_age_light = avg_age_decade.loc[avg_age_decade['category'].isin(['actor', 'actress', 'producer', 'director', 'self', 'writer']), :]

    st.header("Evolution de l'age des intervenants au fil des années", divider="blue")
    choix_cat2 = st.selectbox('Veuillez sélectionner dans la liste ci-dessous (3):', avg_age_light['category'].unique())

    #filtrer le data set selon le choix de l'internaute
    cat_job = avg_age_light.loc[avg_age_light['category'] == choix_cat2,:]

    fig4 = px.bar(cat_job, x='decade', y='avg_age', color_discrete_sequence =['Teal']*len(cat_job), width=800, height=600)
    st.plotly_chart(fig4)

elif page == 'Histoire du cinéma':
    st.image('https://64.media.tumblr.com/tumblr_laju5j5USQ1qe6mn3o1_500.gifv', use_column_width=True)
    st.markdown("<h1 style='text-align: center;'>Un peu d'histoire</h1>", unsafe_allow_html=True)

    st.header("1891")
    st.write("Thomas Edisson dépose le 1er brevet sur l'animation des images. C'est aussi l'année de l'invention de la caméra argentique.")

    st.header('1895')
    st.write("1ere projection publique des frères Lumière.")
    st.video("https://www.youtube.com/watch?v=MSU99gmzn0Q&t=22s")

    st.header('1905')
    st.write("20% de la pop américaine se rend chaque semaine au cinéma. Apparition des nickelodeons.")
    st.write("Un nickelodéon (en anglais : nickelodeon) était un type de petite salle de cinéma de quartier au début du xxe siècle aux États-Unis et au Canada. Le nom provient de l'américain « nickel » et du grec « odéon », qui désignent respectivement la pièce de 5 cents (celle que les spectateurs devaient glisser dans un tourniquet pour accéder à la salle), et un édifice destiné à écouter de la musique. Les nickelodéons sont considérés comme le premier réseau de salles de cinéma, après celui des Kinétoscope Parlors de Thomas Edison au tarif d'entrée plus élevé (une pièce de 25 cents, appelée « quarter »)")

    st.header('1912')
    st.write("Carl Laemmle inaugure le star-system pour mettre en valeur l'actrice Mary Pickford et le cinéma devient une véritable industrie, installant son centre à Los Angeles. Hollywood devient la « Mecque du Cinéma » et « l'usine à rêves ».")

    st.header('1927')
    st.write("'The Jazz Singer' est considéré comme le 1er film parlant de l'Histoire.")
    st.image("https://brittrose.com/wp-content/uploads/2023/03/The-Jazz-Singer-1927.png", width=700)

    st.header('1928')
    st.write("Les 4 principaux studios de production ouvrent la voie de la diffusion du parlant.")

    st.header('1930')
    st.write("Le cinéma sonore s'impose partout.")

    st.header('1932')
    st.write("Apparition de la caméra Technicolor trichrome et début de la couleur.")

    st.header('1939 - 45')
    st.write("Hollywood et ses studios participent à l'effort de guerre")
    st.link_button("Quand Hollywood s'en allait en guerre", "https://www.film-documentaire.fr/4DACTION/w_fiche_film/57426_0")

    st.header('1945 - 55')
    st.write("L'usine à rêve hollywoodienne envahit l'Europe. 2 genres renforcent leur influence : western et film noir")

    st.header('1946')
    st.write("1er festival à Cannes")
    st.image("https://i.f1g.fr/media/figaro/1194x804/2017/05/16/XVM3cd8e044-3a1b-11e7-b5b5-21a5cdc791d1.jpg", width=700)

    st.header('1951')
    st.write("La concurrence de la télévision oblige les Studios à développer le grand écran (Cinémascope) et à généraliser la couleur.")

    st.header('1958')
    st.write("Nouveau souffle pour le cinéma français : l'apparation des tournages en extérieur")

    st.header('1972')
    st.write("Le système Dolby arrive dans les salles")

    st.header('1980 - 90')
    st.write("Début du cinéma d'animation")
    st.image('https://media.senscritique.com/media/000006380560/1200/qui_veut_la_peau_de_roger_rabbit.jpg', width=700)

    st.header('1990')
    st.write("arrivée des multiplexes (cinéma de 10 à 20 salles)")

    st.header('2009')
    st.write("Avènement du numérique 3D avec Avatar, le 1er film de ce genre")
    st.video('https://www.youtube.com/watch?v=MJ3Up7By5cw')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image("https://m.media-amazon.com/images/M/MV5BYjlkMGIwOTktZGNkYy00ZWIwLTk5MmYtMDQzNzBhMDM5MzEyXkEyXkFqcGdeQXVyNjc1NTYwNDk@._V1_.jpg", width=300)

    with col3:
        st.write(' ')
