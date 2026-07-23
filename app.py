"""
MovieLens Recommender System – Baseline + Collaborative Filtering
Built with Streamlit for interactive web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import requests

# ------------------------------------------------------------
# 1. Data downloader (auto-fetch if not present)
# ------------------------------------------------------------
DATA_DIR = 'ml-100k'
ZIP_PATH = 'ml-100k.zip'
URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

@st.cache_resource
def load_data():
    """Download and load MovieLens 100k dataset."""
    if not os.path.exists(DATA_DIR):
        with st.spinner('Downloading MovieLens 100k dataset...'):
            r = requests.get(URL)
            with open(ZIP_PATH, 'wb') as f:
                f.write(r.content)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(ZIP_PATH)

    # Load ratings (u.data)
    user_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, 'u.data'),
        sep='\t',
        names=user_cols,
        encoding='latin-1'
    )

    # Load movie titles (u.item) - only columns 0 and 1
    movies = pd.read_csv(
        os.path.join(DATA_DIR, 'u.item'),
        sep='|',                    # important: pipe separator
        names=['item_id', 'title'], # only two columns
        encoding='latin-1',
        usecols=[0, 1]              # read only these two columns
    )

    data = ratings.merge(movies, on='item_id')
    return data

# ------------------------------------------------------------
# 2. Baseline models (cached)
# ------------------------------------------------------------
@st.cache_data
def compute_models(data):
    global_avg = data['rating'].mean()
    movie_avg = data.groupby('title')['rating'].mean()
    movie_counts = data.groupby('title')['rating'].count()

    def bayes_avg(group, g_mean=global_avg, prior=10):
        n = len(group)
        avg = group.mean()
        return (avg * n + g_mean * prior) / (n + prior)

    movie_bayes = data.groupby('title')['rating'].agg(bayes_avg)

    pred_global = np.full(len(data), global_avg)
    pred_movie = data['title'].map(movie_avg)
    pred_bayes = data['title'].map(movie_bayes)

    return {
        'global_avg': global_avg,
        'movie_avg': movie_avg,
        'movie_counts': movie_counts,
        'movie_bayes': movie_bayes,
        'rmse_global': np.sqrt(mean_squared_error(data['rating'], pred_global)),
        'rmse_movie': np.sqrt(mean_squared_error(data['rating'], pred_movie)),
        'rmse_bayes': np.sqrt(mean_squared_error(data['rating'], pred_bayes)),
        'data': data
    }

# ------------------------------------------------------------
# 3. Collaborative Filtering (User-Based KNN)
# ------------------------------------------------------------
@st.cache_data
def compute_cf(data):
    """Build user-item matrix and compute cosine similarity."""
    user_item_pivot = data.pivot_table(
        index='user_id',
        columns='title',
        values='rating'
    ).fillna(0)

    user_item_matrix = user_item_pivot.values
    user_sim = cosine_similarity(user_item_matrix)

    user_sim_df = pd.DataFrame(
        user_sim,
        index=user_item_pivot.index,
        columns=user_item_pivot.index
    )

    return {
        'user_item_pivot': user_item_pivot,
        'user_sim_df': user_sim_df,
        'user_ids': user_item_pivot.index.tolist()
    }

def predict_cf_rating(user_id, movie_title, cf_data, data, k=10):
    """Predict rating for a given user and movie using user-based CF."""
    pivot = cf_data['user_item_pivot']
    sim_df = cf_data['user_sim_df']

    if movie_title not in pivot.columns:
        return None

    users_who_rated = pivot[pivot[movie_title] > 0].index.tolist()
    if not users_who_rated:
        return None

    if user_id not in pivot.index:
        return None

    sim_scores = sim_df.loc[user_id, users_who_rated]
    ratings = pivot.loc[users_who_rated, movie_title]

    top_k_idx = sim_scores.nlargest(k).index
    top_k_sim = sim_scores[top_k_idx]
    top_k_ratings = ratings[top_k_idx]

    if top_k_sim.sum() > 0:
        pred = (top_k_sim * top_k_ratings).sum() / top_k_sim.sum()
    else:
        pred = data['rating'].mean()

    return round(pred, 3)

def recommend_cf_for_user(user_id, cf_data, data, n=10, k=10):
    """Recommend top N movies for a user using CF."""
    pivot = cf_data['user_item_pivot']
    if user_id not in pivot.index:
        return pd.Series()

    user_rated = pivot.loc[user_id]
    unrated_movies = user_rated[user_rated == 0].index.tolist()

    predictions = {}
    for movie in unrated_movies:
        pred = predict_cf_rating(user_id, movie, cf_data, data, k)
        if pred is not None:
            predictions[movie] = pred

    if predictions:
        sorted_pred = pd.Series(predictions).sort_values(ascending=False)
        return sorted_pred.head(n)
    return pd.Series()

# ------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 MovieLens Recommender – Baseline + Collaborative Filtering")
st.markdown("**Popularity models + User-based Collaborative Filtering (KNN)**")

data = load_data()
models = compute_models(data)
cf_data = compute_cf(data)

# Sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model to predict ratings:",
    ("Bayesian (recommended)", "Per‑movie average", "Global average")
)

model_map = {
    "Bayesian (recommended)": models['movie_bayes'],
    "Per‑movie average": models['movie_avg'],
    "Global average": None
}
pred_dict = model_map[model_choice] if model_choice != "Global average" else None

st.sidebar.subheader("RMSE Comparison")
st.sidebar.write(f"Global: `{models['rmse_global']:.4f}`")
st.sidebar.write(f"Per‑movie: `{models['rmse_movie']:.4f}`")
st.sidebar.write(f"Bayesian: `{models['rmse_bayes']:.4f}`")
st.sidebar.caption("Lower is better")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Top Recommendations",
    "🎯 Predict Rating",
    "📊 Data Analysis",
    "👥 Collaborative Filtering"
])

# ---------- TAB 1: Popularity Recommendations ----------
with tab1:
    st.header("Top N Recommendations for a New User (Popularity‑Based)")
    col1, col2 = st.columns([1, 3])
    with col1:
        n = st.slider("Number of recommendations", 5, 30, 10)
        show_counts = st.checkbox("Show number of ratings", value=True)

    top_movies = models['movie_bayes'].sort_values(ascending=False).head(n)
    display_df = pd.DataFrame({
        'Movie': top_movies.index,
        'Predicted Rating': top_movies.values
    })
    if show_counts:
        display_df['# Ratings'] = [models['movie_counts'][title] for title in top_movies.index]
    st.dataframe(display_df, use_container_width=True)
    st.caption("📌 *Bayesian smoothing avoids over‑rating movies with very few votes.*")

# ---------- TAB 2: Predict Rating ----------
with tab2:
    st.header("Predict Rating for a Specific Movie")
    all_titles = sorted(models['movie_bayes'].index.tolist())
    movie_input = st.selectbox("Start typing a movie title:", all_titles)

    if movie_input:
        if model_choice == "Global average":
            pred = models['global_avg']
            st.info(f"Global average always predicts: **{pred:.3f}**")
        else:
            pred = pred_dict.get(movie_input, models['global_avg'])
            actual_count = models['movie_counts'].get(movie_input, 0)
            st.success(f"Predicted rating for **{movie_input}**: **{pred:.3f}**")
            st.write(f"Number of ratings in training set: {actual_count}")

        with st.expander("ℹ️ How is this calculated?"):
            if model_choice == "Global average":
                st.write("Simply the mean of all 100k ratings.")
            elif model_choice == "Per‑movie average":
                st.write("Average of all ratings for this movie. If the movie is new, it falls back to global average.")
            else:
                st.write("Bayesian average = (movie_avg * n + global_avg * 10) / (n + 10). This pulls extreme ratings toward the global mean.")

# ---------- TAB 3: Data Analysis ----------
with tab3:
    st.header("Dataset Exploration & Cold‑Start Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ratings", f"{len(data):,}")
    col2.metric("Unique Users", data['user_id'].nunique())
    col3.metric("Unique Movies", data['item_id'].nunique())

    counts = models['movie_counts']
    few = (counts < 5).sum()
    one = (counts == 1).sum()
    st.warning(f"⚠️ **Cold‑start problem**: {few} movies have < 5 ratings, and {one} have exactly 1 rating.")
    st.info("💡 *Bayesian smoothing partially mitigates this, but new movies with zero ratings still fall back to global average.*")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    data['rating'].hist(bins=5, edgecolor='black', ax=ax1)
    ax1.set_title('Rating Distribution')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    counts.hist(bins=50, edgecolor='black', ax=ax2)
    ax2.set_title('Number of Ratings per Movie')
    ax2.set_xlabel('Number of ratings')
    ax2.set_ylabel('Count')
    st.pyplot(fig)

    raw_top = models['movie_avg'].sort_values(ascending=False).head(5)
    bayes_top = models['movie_bayes'].sort_values(ascending=False).head(5)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 by Raw Average (biased)")
        for title, val in raw_top.items():
            st.write(f"**{title}**: {val:.3f} ({counts[title]} ratings)")
    with col2:
        st.subheader("Top 5 by Bayesian (smoothed)")
        for title, val in bayes_top.items():
            st.write(f"**{title}**: {val:.3f} ({counts[title]} ratings)")
    st.caption("📊 Notice how obscure movies with one 5‑star rating disappear from the Bayesian top list.")

# ---------- TAB 4: Collaborative Filtering ----------
with tab4:
    st.header("👥 User‑Based Collaborative Filtering")
    st.markdown("""
    **How it works:** 
    - We find users with similar rating patterns (cosine similarity).
    - For a movie you haven't seen, we look at what similar users rated it.
    - The prediction is a weighted average of their ratings.

    ⚡ **Time complexity:** Naive KNN is O(n²) for building similarity, O(k) per prediction. At scale, we'd use Approximate Nearest Neighbors (ANN).
    """)

    user_ids = cf_data['user_ids']
    selected_user = st.selectbox("Select a user ID (1-943):", user_ids, index=0)
    k = st.slider("Number of neighbors (k):", 5, 50, 20, step=5)

    if st.button("🎯 Get Recommendations for This User"):
        with st.spinner(f"Computing recommendations for User {selected_user}..."):
            recs = recommend_cf_for_user(selected_user, cf_data, data, n=10, k=k)
            if recs.empty:
                st.warning(f"User {selected_user} not found or has rated all movies.")
            else:
                st.success(f"Top 10 recommendations for User {selected_user}:")
                display_cf = pd.DataFrame({
                    'Movie': recs.index,
                    'Predicted Rating': recs.values
                })
                st.dataframe(display_cf, use_container_width=True)

                pivot = cf_data['user_item_pivot']
                user_ratings = pivot.loc[selected_user]
                top_rated = user_ratings[user_ratings > 0].sort_values(ascending=False).head(5)
                st.subheader(f"User {selected_user}'s Top 5 Rated Movies:")
                for title, rating in top_rated.items():
                    st.write(f"⭐ {rating:.1f} – {title}")

                st.info("💡 **Interview talking point:** Collaborative Filtering captures user preferences better than popularity, but suffers from the cold‑start problem (new users). In production, we'd use a hybrid approach — popularity for new users, CF for active users.")

    with st.expander("📊 View User Similarity Matrix (Sample)"):
        st.write("Similarity between first 10 users (1 = identical, 0 = orthogonal):")
        st.dataframe(cf_data['user_sim_df'].iloc[:10, :10])

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Interview Talking Points")
st.sidebar.markdown("""
- **Cold‑start** – how to handle new users/items.
- **RMSE vs MAE** – why square errors?
- **Bayesian smoothing** – form of regularisation.
- **Collaborative Filtering** – KNN, cosine similarity, sparsity problem.
- **Time complexity** – O(n²) for similarity, O(k) per prediction.
- **Hybrid approach** – combine popularity + CF for best results.
""")
