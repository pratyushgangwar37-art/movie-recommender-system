 🎬 Movie Recommender System

A production-ready movie recommendation web app built for the Amazon internship interview process. This project demonstrates a structured approach to building, evaluating, and deploying a recommender system.

## 🧠 Project Architecture
- **Data**: Movielens 100k (100,000 ratings from 943 users on 1682 movies).
- **Models Implemented**:
  - *Global Average* (Dummy baseline)
  - *Per-Movie Average* (Popularity baseline)
  - *Bayesian Smoothing* (Solves the "cold-start" problem for movies with few ratings).
  - *User-Based Collaborative Filtering* (KNN with Cosine Similarity).
- **Evaluation Metric**: RMSE (Root Mean Squared Error).
- **Deployment**: Interactive UI built with Streamlit.

 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (RMSE, Cosine Similarity)
- Streamlit (Front-end)
- Matplotlib (Visualization)

 📊 Key Results
- Global Average RMSE: '1.124'
- Per-Movie Average RMSE: `0.943`
- Bayesian Average RMSE: `0.944` (Handles cold-start effectively)

 ⚙️ How to Run Locally
bash
# 1. Clone the repository
git clone https://github.com/pratyushgangwar37-art/movie-recommender-system.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py
