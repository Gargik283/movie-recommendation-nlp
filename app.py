import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CineMatch | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — CINEMATIC UI
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(120, 70, 255, 0.16), transparent 28%),
            radial-gradient(circle at 85% 15%, rgba(255, 60, 120, 0.12), transparent 25%),
            #090b12;
        color: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0d1019;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e9edf5;
    }

    .hero {
        padding: 55px 20px 35px 20px;
        text-align: center;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        background: rgba(130, 90, 255, 0.15);
        border: 1px solid rgba(160, 130, 255, 0.30);
        color: #bdaeff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.4px;
        margin-bottom: 18px;
    }

    .hero h1 {
        font-size: clamp(38px, 6vw, 68px);
        line-height: 1.03;
        margin: 0;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #ffffff, #cfc5ff, #ff9fc2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        max-width: 760px;
        margin: 20px auto 0;
        color: #aeb6c8;
        font-size: 17px;
        line-height: 1.7;
    }

    .search-panel {
        max-width: 850px;
        margin: 10px auto 30px;
        padding: 25px;
        background: rgba(18, 22, 34, 0.82);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 22px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        margin: 30px 0 16px;
        color: #ffffff;
    }

    .match-box {
        padding: 13px 17px;
        border-radius: 13px;
        background: rgba(113, 87, 255, 0.10);
        border: 1px solid rgba(145, 120, 255, 0.25);
        color: #d9d2ff;
        margin: 12px 0 20px;
    }

    .movie-card {
        min-height: 145px;
        padding: 23px;
        margin: 8px 0 14px;
        border-radius: 18px;
        background: linear-gradient(145deg, #151a29, #10131e);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .movie-card:hover {
        transform: translateY(-3px);
        border-color: rgba(160,130,255,0.35);
    }

    .movie-rank {
        color: #a997ff;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .movie-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        margin-top: 8px;
    }

    .movie-score {
        color: #8f98aa;
        font-size: 13px;
        margin-top: 9px;
    }

    .stat-card {
        padding: 18px;
        border-radius: 16px;
        background: #111521;
        border: 1px solid rgba(255,255,255,0.07);
        text-align: center;
    }

    .stat-number {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
    }

    .stat-label {
        color: #8f98aa;
        font-size: 12px;
        margin-top: 4px;
    }

    .sidebar-brand {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .sidebar-sub {
        color: #8f98aa;
        font-size: 13px;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        margin: 10px 0;
        color: #aeb6c8;
        font-size: 13px;
        line-height: 1.6;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 0;
        padding: 12px 20px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #7157ff, #b14cff);
        box-shadow: 0 8px 25px rgba(113,87,255,0.25);
    }

    div.stButton > button:hover {
        border: 0;
        color: white;
        background: linear-gradient(90deg, #8069ff, #bd5cff);
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FILES
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Keep these filenames exactly the same as your local files.
INDICES_FILE = BASE_DIR / "indices.pkl"
MATRIX_FILE = BASE_DIR / "tfidf_matrix.pkl"
VECTORIZER_FILE = BASE_DIR / "tfidf.pkl"
DF_FILE = BASE_DIR / "df.pkl"


# ============================================================
# LOAD PICKLE FILES
# ============================================================
@st.cache_resource
def load_data():
    with open(INDICES_FILE, "rb") as f:
        indices = pickle.load(f)

    with open(MATRIX_FILE, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Your TF-IDF.pkl is a fitted TfidfVectorizer.
    # It is loaded for completeness, although recommendations
    # can be generated directly from the saved TF-IDF matrix.
    vectorizer = None
    if VECTORIZER_FILE.exists():
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)

    # df.pkl is optional for this UI because indices.pkl
    # already contains every movie title and its row index.
    df = None
    if DF_FILE.exists():
        with open(DF_FILE, "rb") as f:
            df = pickle.load(f)

    return indices, tfidf_matrix, vectorizer, df


# ============================================================
# RECOMMENDATION FUNCTIONS
# ============================================================
def normalize_title(title):
    """Normalize a title for easier matching."""
    return " ".join(str(title).lower().strip().split())


def find_best_movie(query, movie_titles):
    """
    Finds the closest movie title even when:
    - spelling is imperfect
    - capitalization is different
    - only part of the title is entered
    """
    query = normalize_title(query)

    if not query:
        return None, 0

    normalized_titles = {
        normalize_title(title): title
        for title in movie_titles
    }

    # Exact normalized match
    if query in normalized_titles:
        return normalized_titles[query], 100

    # Fuzzy matching
    match = process.extractOne(
        query,
        list(normalized_titles.keys()),
        scorer=fuzz.WRatio
    )

    if match is None:
        return None, 0

    matched_normalized, score, _ = match
    return normalized_titles[matched_normalized], score


def get_recommendations(movie_title, indices, tfidf_matrix, n=10):
    """
    Generate recommendations using cosine similarity
    between the selected movie and every movie in the
    saved TF-IDF matrix.
    """
    movie_index = indices[movie_title]

    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Highest scores first
    similar_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []

    for idx in similar_indices:
        title = indices.index[idx]

        # Don't recommend the movie itself
        if title == movie_title:
            continue

        recommendations.append(
            (title, float(similarity_scores[idx]))
        )

        if len(recommendations) >= n:
            break

    return recommendations


# ============================================================
# LOAD
# ============================================================
try:
    indices, tfidf_matrix, vectorizer, df = load_data()

    movie_titles = list(indices.index)

except FileNotFoundError as e:
    st.error(
        f"Missing file: **{e.filename}**. "
        "Please keep all pickle files in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load the recommendation files: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🎬 CineMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Movie Recommendation System</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📌 About the Project")
    st.markdown(
        """
        <div class="info-box">
        CineMatch recommends movies that are most similar to your
        selected movie using <b>TF-IDF text features</b> and
        <b>cosine similarity</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Topics Covered")
    st.markdown(
        """
        <div class="info-box">
        • Natural Language Processing<br>
        • TF-IDF Vectorization<br>
        • Cosine Similarity<br>
        • Content-Based Recommendation<br>
        • Fuzzy Movie Search<br>
        • Streamlit Deployment
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ Model Details")
    st.markdown(
        f"""
        <div class="info-box">
        <b>Dataset:</b> {len(movie_titles):,} movies<br>
        <b>TF-IDF Features:</b> 5,000<br>
        <b>N-grams:</b> 1–2<br>
        <b>Recommendation:</b> Content-Based
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💡 Search Tip")
    st.markdown(
        """
        <div class="info-box">
        You don't need the exact spelling. Try typing a
        partial or misspelled movie title and CineMatch
        will find the closest match.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✦ AI-POWERED MOVIE DISCOVERY</div>
        <h1>Find Your Next Favorite Movie</h1>
        <p>
            Enter a movie you love and discover similar movies
            powered by Natural Language Processing, TF-IDF
            and content-based recommendation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SEARCH AREA
# ============================================================
st.markdown('<div class="search-panel">', unsafe_allow_html=True)

search_mode = st.radio(
    "Choose how you want to find a movie",
    ["🔎 Search by title", "🎞️ Select from movies"],
    horizontal=True,
    label_visibility="collapsed"
)

selected_movie = None
match_score = 0

if search_mode == "🔎 Search by title":

    query = st.text_input(
        "Movie title",
        placeholder="Try: Spider Man, Avatr, Inception...",
        label_visibility="collapsed"
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    recommend_clicked = st.button(
        "✨ Find Similar Movies",
        use_container_width=True
    )

    if recommend_clicked:
        if not query.strip():
            st.warning("Please enter a movie title first.")
        else:
            selected_movie, match_score = find_best_movie(
                query,
                movie_titles
            )

else:

    selected_movie = st.selectbox(
        "Select a movie",
        movie_titles,
        index=None,
        placeholder="Choose a movie...",
        label_visibility="collapsed"
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    recommend_clicked = st.button(
        "✨ Find Similar Movies",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# RECOMMENDATIONS
# ============================================================
if recommend_clicked and selected_movie:

    if search_mode == "🔎 Search by title":
        if normalize_title(query) != normalize_title(selected_movie):
            st.markdown(
                f"""
                <div class="match-box">
                    🎯 We matched your search to
                    <b>{selected_movie}</b>
                    <span style="opacity:0.7;">({match_score:.0f}% match)</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.spinner("Finding movies you may love..."):
        recommendations = get_recommendations(
            selected_movie,
            indices,
            tfidf_matrix,
            number_of_recommendations
        )

    st.markdown(
        f'<div class="section-title">🍿 Because you liked {selected_movie}</div>',
        unsafe_allow_html=True
    )

    # Stats
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(recommendations)}</div>
                <div class="stat-label">RECOMMENDATIONS</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(movie_titles):,}</div>
                <div class="stat-label">MOVIES IN DATABASE</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">TF-IDF</div>
                <div class="stat-label">RECOMMENDATION ENGINE</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    # Recommendation cards
    for rank, (title, score) in enumerate(recommendations, start=1):
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="movie-rank">#{rank} • Recommended for You</div>
                <div class="movie-title">🎬 {title}</div>
                <div class="movie-score">
                    Content similarity score: {score:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif recommend_clicked and not selected_movie:
    st.info("Select or search for a movie to get recommendations.")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:60px;
        padding:25px;
        color:#687286;
        font-size:12px;
        border-top:1px solid rgba(255,255,255,0.06);
    ">
        Built with Python • Scikit-learn • NLP • TF-IDF • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)