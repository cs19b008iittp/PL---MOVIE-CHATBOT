import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
from ast import literal_eval
import requests
import time
import streamlit.components.v1 as components


# def triu_custom(A, k=0):
#     return np.triu(A, k)
    
# TMDB API key
TMDB_API_KEY = 'aad48407a8c1adecea9cc23891d3181a'  # Replace with your actual TMDB API key

# Load and preprocess data only once
@st.cache_data
def load_and_preprocess_data():
    # Load datasets
    movies_data = pd.read_csv('./data/movies_metadata.csv')
    link_small = pd.read_csv('./data/links_small.csv')
    credits = pd.read_csv('./data/credits.csv')
    keyword = pd.read_csv('./data/keywords.csv')

    # Data preprocessing
    movies_data = movies_data.drop([19730, 29503, 35587])
    movies_data['id'] = movies_data['id'].astype('int')
    link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    
    smd = movies_data[movies_data['id'].isin(link_small)]
    smd['tagline'] = smd['tagline'].fillna('')
    smd['overview'] = smd['overview'].fillna('')
    
    # Merging
    movies_data_merged = movies_data.merge(keyword, on='id').merge(credits, on='id')
    smd2 = movies_data_merged[movies_data_merged['id'].isin(link_small)]
    smd2['cast'] = smd2['cast'].apply(literal_eval).apply(lambda x: [i['name'] for i in x][:3] if isinstance(x, list) else [])
    smd2['crew'] = smd2['crew'].apply(literal_eval)
    smd2['keywords'] = smd2['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    # Get directors
    def get_director(crew):
        for i in crew:
            if i.get('job') == 'Director':
                return i['name']
        return ""
    
    smd2['directors'] = smd2['crew'].apply(get_director).apply(lambda x: [x, x, x] if x else [])
    smd2['overview'] = smd2['overview'].apply(lambda x: [x] if isinstance(x, str) else x)
    
    # Create "soup"
    smd2["soup"] = smd2['keywords'] + smd2['cast'] + smd2['directors'] + smd2['overview']
    smd2['soup'] = smd2['soup'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    smd2['soup'] = smd2['soup'].apply(lambda x: remove_stopwords(x)).apply(lambda x: " ".join(SnowballStemmer('english').stem(word) for word in x.split()))
    
    return smd2

smd2 = load_and_preprocess_data()

@st.cache_data
def get_tfidf_matrix(smd):
    # Vectorize once and cache the matrix
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['soup'])
    return tfidf_matrix, tf

tfidf_matrix, tf = get_tfidf_matrix(smd2)

def get_movie_poster(movie_id, retries=3):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    for i in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            # Default blank poster if no poster URL is available
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i + 1}: Connection error occurred: {e}")
            time.sleep(2 ** i)
    return "https://via.placeholder.com/500x750?text=No+Poster+Available"

def getPredictionsV2(user_input, smd, num):
    stopword_removed_soup = remove_stopwords(user_input)
    stemmed_soup = " ".join(SnowballStemmer('english').stem(word) for word in stopword_removed_soup.split())
    
    input_tfidf = tf.transform([stemmed_soup])
    cosine_sim = linear_kernel(input_tfidf, tfidf_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    return smd.iloc[movie_indices][['title', 'id']]




def get_movie_details(movie_name):
    movie_row = smd2[smd2['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return f"No information found for '{movie_name}'. Please check the title and try again."

    # Use only the first result if multiple matches
    row = movie_row.iloc[0]
    title = row['title']
    overview = row['overview']
    director = ', '.join(set(row['directors']))
    rating = row['vote_average'] if 'vote_average' in row else "N/A"
    release_date = row['release_date'] if 'release_date' in row else "N/A"
    poster_url = get_movie_poster(row['id'])

    # Card-style UI with HTML and CSS
    card_html = f"""
    <style>
        .movie-card {{
            display: flex;
            align-items: center;
            padding: 20px;
            background-color: #333;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            color: #fff;
        }}
        .movie-poster {{
            max-width: 200px;
            border-radius: 8px;
            margin-right: 20px;
        }}
        .movie-details {{
            flex: 1;
        }}
        .movie-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .movie-info {{
            font-size: 1em;
            margin: 5px 0;
        }}
    </style>
    <div class="movie-card">
        <img class="movie-poster" src="{poster_url}" alt="{title} poster">
        <div class="movie-details">
            <div class="movie-title">{title}</div>
            <div class="movie-info"><strong>Release Date:</strong> {release_date}</div>
            <div class="movie-info"><strong>Director:</strong> {director}</div>
            <div class="movie-info"><strong>Rating:</strong> {rating}</div>
            <div class="movie-info"><strong>Overview:</strong> {overview}</div>
        </div>
    </div>
    """
    return components.html(card_html, height=500)

def get_vote_average(movie_name):
    movie_row = smd2[smd2['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return f"No information found for '{movie_name}'. Please check the title and try again."
    
    try:
        return int(movie_row['vote_average'].values[0] * 10)  # Convert to percentage
    except (ValueError, TypeError):
        return "Rating not available."

def get_movie_release_date(movie_name):
    movie_row = smd2[smd2['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return "Release date not available."
    
    release_date = movie_row['release_date'].values[0] if 'release_date' in movie_row.columns else None
    return release_date if pd.notna(release_date) else "No release date"

st.title("🎬 Movie Recommendation Bot")
st.write("Welcome! This bot will help you discover movies and get details about your favorites.")

option = st.selectbox("What would you like to do?", ("Get Movie Recommendations", "Get Movie Details"))

if option == "Get Movie Recommendations":
    st.subheader("Tell me about your movie preferences:")
    user_input = st.text_area("Enter your preferences (e.g., genre, director, keywords, etc.):")

    if st.button("Get Recommendations"):
        if user_input:
            recommendations = getPredictionsV2(user_input, smd2, 10)
            st.write("🎉 Here are some movie recommendations based on your preferences:")

            # HTML and CSS for fixed horizontal carousel


            carousel_html = """
<style>
    .carousel-container {
        display: flex;
        overflow-x: auto;
        overflow-y: hidden;
        gap: 20px;  /* Increase the gap between items */
        padding: 10px;
        white-space: nowrap;
        scrollbar-width: thin;
    }
    .carousel-item {
        flex: 0 0 170px;
        width: 170px;
        text-align: center;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        padding: 10px;
        position: relative;
    }
    .carousel-item img {
        border-radius: 5px;
        width: 100%;
        height: 240px;
        object-fit: cover;
        position: relative;
    }
    .carousel-item .title {
        font-weight: bold;
        margin-top: 10px;
        font-size: 0.9em;
        text-align: center;  /* Center-align text */
        white-space: normal;  /* Allow text to wrap */
        word-wrap: break-word; /* Break long words if necessary */
        line-height: 1.2em;   /* Adjust line height for better readability */
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 2.4em;    /* Minimum height for two lines */
    }
    .carousel-item .release-date {
        font-size: 0.8em;
        color: #555;
        margin-top: 5px;
    }
    /* Circular rating positioned in the bottom-left corner of the poster */
    .carousel-item .rating-circle {
        --score: calc(var(--rating) * 1%);  /* Set rating percentage */
        position: absolute;
        bottom: 0px; /* Adjust to position within the poster */
        left: 0px;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background: #000000;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: bold;
        font-size: 0.8em;
        border: 2px solid #008000; /* Optional: add a white border for better visibility */
    }
    .carousel-item .rating-circle::before {
        content: attr(data-rating) "%";
    }
</style>
<div class="carousel-container">
"""

            for _, movie in recommendations.iterrows():
                poster_url = get_movie_poster(movie['id'])
                rating = get_vote_average(movie['title'])
                release_date = get_movie_release_date(movie['title'])
                poster_img = f'<img src="{poster_url}" alt="{movie["title"]} poster">' if poster_url else "No Poster"

                carousel_html += f"""
                <div class="carousel-item">
        <div style="position: relative;">
            {poster_img}
            <div class="rating-circle" style="--rating: {rating};" data-rating="{rating if isinstance(rating, int) else 'N/A'}"></div>
        </div>
        <div class="title">{movie['title']}</div>
        <div class="release-date"><strong>{release_date}</strong></div>
    </div>
                """
            carousel_html += "</div>"
            components.html(carousel_html, height=350, scrolling=False)
        else:
            st.warning("Please enter your movie preferences to get recommendations.")

elif option == "Get Movie Details":
    movie_name = st.text_input("Enter the movie name:")
    if st.button("Get Details"):
        if movie_name:
            get_movie_details(movie_name)
        else:
            st.warning("Please enter a movie name.")
