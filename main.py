import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os


load_dotenv()  # Load from .env file
API_KEY = os.getenv("TMDB_API_KEY")
movies = pickle.load(open("movies.pkl","rb"))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


st.title("Movies Recommendation Model")

#dropdown for movie selection

selected_movie = st.selectbox("choose a movie recommendation: ", movies['title'].values)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmZTE1NDQ4ODRlZTNhNDg3ZjI3YmUyZDQwNWY2N2ZlOSIsIm5iZiI6MTc0NDQ0MzU5MC44NTksInN1YiI6IjY3ZmExOGM2ZjgxYjAyYjA4MDk5MjkzNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.zWEC57qeycIA9UFhKrF3StX4Qdu6Va5qKVs4aLHExNE"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    return "https://image.tmdb.org/t/p/w500" + data['poster_path']


def recommendation(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(enumerate(distances), reverse=True, key= lambda x: x[1] )[1:21]

    movies_name = []
    movies_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        movies_name.append(movies.iloc[i[0]].title)
        movies_posters.append(fetch_poster(movie_id))

    return movies_name,movies_posters


if st.button('Recommend'):
    names, posters = recommendation(selected_movie)

    st.subheader('Top 5 Similar Movies:')
    for row in range(4):  # 2 rows
        cols = st.columns(5)
        for i in range(5):  # 5 columns per row
            movie_index = row * 5 + i
            if movie_index < len(names):
                with cols[i]:
                    st.image(posters[movie_index])
                    st.caption(names[movie_index])



