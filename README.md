# 🎬 Movie Recommendation System

A **content-based movie recommendation system** built with Python, NLP, TF-IDF, Cosine Similarity, and Streamlit.

🔗 **Live Demo:** https://movie-recommendation-nlp-2209.streamlit.app/

## ✨ Features

- 🔎 Search for movies by title
- ✍️ Fuzzy matching for misspelled/partial movie names
- 🎬 Get similar movie recommendations
- 📊 Adjustable number of recommendations
- 🌙 Clean, professional cinematic UI
- 🚀 Deployed using Streamlit Community Cloud

## 🧠 How It Works

**Movie Data → Text Processing → TF-IDF → Cosine Similarity → Similar Movies → Streamlit UI**

The system uses a **content-based recommendation approach**, where movies are recommended based on the similarity of their textual features.

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **NLP**
- **TF-IDF Vectorization**
- **Cosine Similarity**
- **RapidFuzz**
- **Streamlit**
- **Git & GitHub**

## 📂 Project Files

```text
Movie-Recommendation-System/
│
├── app.py
├── indices.pkl
├── tfidf_matrix.pkl
├── tfidf.pkl
├── df.pkl
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Movie-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Example

Enter a movie such as:

```text
Inception
```

The application finds similar movies based on their content.

Even an approximate search such as:

```text
incepton
```

can be matched to the correct movie title.

## 👩‍💻 Author

**Gargi Kundu**

B.Tech in Electronics and Communication Engineering (VLSI Design)

**Skills:** Python • Machine Learning • NLP • Data Analysis • Streamlit • Git/GitHub

⭐ If you like this project, consider giving the repository a star!
