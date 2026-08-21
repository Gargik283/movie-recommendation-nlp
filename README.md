# Content-Based Movie Recommendation Engine

[![Streamlit App](https://streamlit.io)](https://streamlit.app)

An end-to-end Natural Language Processing (NLP) production pipeline that delivers semantic movie recommendations. This project features a vectorised similarity engine engineered for real-time textual retrieval and is fully deployed as a live cloud application.

🚀 **Live Production Demo:** [movie-recommendation-nlp-2209.streamlit.app](https://streamlit.app)

## 🚀 Key Production Features
* **Live Cloud Deployment:** Fully containerized and operational via Streamlit Cloud for real-time user testing.
* **Semantic Analysis:** Leverages natural language preprocessing to clean, parse, and handle unstructured movie text data.
* **TF-IDF Vectorization:** Transforms raw metadata and plot summaries into high-dimensional numerical feature matrices using Term Frequency-Inverse Document Frequency.
* **Cosine Similarity Engine:** Evaluates directional proximity across text feature vectors to retrieve contextual recommendations instantly.
* **Optimized Serialization:** Utilizes pre-computed data structures (`.pkl`) to eliminate runtime computation bottlenecks and maintain sub-second app latency.

## 🛠️ Architecture & Tech Stack
* **Frontend UI & Deployment:** Streamlit Cloud
* **NLP & Vectorization:** Scikit-Learn (`TfidfVectorizer`), NumPy
* **Data Engineering:** Pandas
* **Model Pipeline Storage:** Pickle (for instantaneous binary matrix loading)

## 📁 Repository Structure
* `app.py`: Production code managing the Streamlit UI, session states, and vector matching logic.
* `movies_metadata.csv`: Core corpus containing rich textual descriptions and movie metadata.
* `tfidf_matrix.pkl`: Serialized high-dimensional sparse matrix representing text features.
* `tfidf.pkl`: Serialized model state of the trained text vectorizer.
* `indices.pkl`: Optimized inverse-lookup dictionary mapping titles to sparse matrix indices ($O(1)$ complexity).
* `df.pkl`: Cleaned, memory-optimized DataFrame footprint ready for fast server parsing.

## ⚙️ Mathematical & NLP Methodology
1. **Lexical Preprocessing:** Movie summaries are tokenized, with standard English stop words stripped away to isolate significant contextual terms.
2. **Vector Space Modeling:** Statistical word importance weights are derived via:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)$$
3. **Similarity Assessment:** Vector alignments between the target film ($A$) and catalog matrices ($B$) are ranked utilizing Cosine Similarity:
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd movie-recommendation-nlp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the local development server:**
   ```bash
   streamlit run app.py
   ```

## 🎯 Interviewer Focus: Strategic Engineering Choices
* **Eliminating Latency ($O(N^2) \rightarrow O(1)$):** Computing a dense TF-IDF matrix from scratch during an active user web request introduces severe lag. By training the vectorizer ahead of time and saving it into optimized binary `.pkl` formats, the web server performs a static memory lookup instead of an active heavy calculation.
* **Production Resource Management:** The dataset and matrices are structured carefully to operate comfortably within Streamlit Cloud's free-tier container memory limitations, ensuring high application uptime without scaling costs.
