# Task5

# **Movie Recommendation System**

## **Project Overview**

This project implements a **content-based movie recommendation system** using the MovieLens dataset.
The system analyzes a movie’s **genres, overview, and tagline** to recommend similar movies.
It’s built with **Python, Pandas, Scikit-learn, and NLP techniques** for preprocessing and similarity calculation.

---

## **Motivation**

Finding the right movie to watch can be overwhelming due to the sheer number of options.
This system provides **personalized recommendations** based on the content of a movie, helping users quickly discover similar movies they might enjoy.

---

## **Dataset**

* **Source:** [MovieLens 100K / Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
* **Size:** 45,000+ movies
* **Key Features Used:**

  * `title` → Movie title
  * `genres` → Movie genres
  * `overview` → Movie summary
  * `tagline` → Movie tagline
  * `vote_average` → Average user rating
  * `popularity` → Popularity score

---

## **Libraries & Tools**

* **Python 3.10+**
* **Pandas** → Data manipulation
* **NumPy** → Numerical computations
* **Matplotlib / Seaborn** → Visualization (optional)
* **NLTK** → Text preprocessing (stopwords, lemmatization)
* **Scikit-learn** → TF-IDF vectorization, cosine similarity
* **Pickle** → Saving models & matrices

---

## **Project Workflow**

### **1. Data Cleaning & Preprocessing**

* Removed duplicates and irrelevant columns
* Handled missing values (`overview`, `tagline`)
* Combined `overview`, `genres`, and `tagline` into a single `tags` column

### **2. Text Preprocessing**

* Lowercased all text
* Removed special characters and numbers
* Removed stopwords
* Lemmatized words using NLTK

### **3. Feature Extraction**

* Used **TF-IDF Vectorizer** with `max_features=50,000` and `(1,2)` n-grams
* Created a **movie-content matrix** representing each movie

### **4. Similarity Computation**

* Calculated **cosine similarity** between all movies
* Implemented a **recommend function** that returns top N similar movies for a given title

### **5. Saving Objects**

* Saved preprocessed objects with Pickle:

  * `tfidf_matrix.pkl` → TF-IDF features
  * `indices.pkl` → Movie title to index mapping
  * `df.pkl` → Cleaned DataFrame
  * `tfidf.pkl` → TF-IDF vectorizer

---

## **Usage**

```python
import pickle

# Load pre-trained objects
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl','rb'))
indices = pickle.load(open('indices.pkl','rb'))
df = pd.read_pickle('df.pkl')

# Recommendation function
def recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx]

# Example
recommend('Avatar 2', 5)
```

---

## **Example Recommendations**

| Input Movie | Recommended Movies                                                                              |
| ----------- | ----------------------------------------------------------------------------------------------- |
| Avatar 2    | The Three Musketeers, Frank Herbert's Dune, Nightmare City 2035, France société anonyme, Désiré |
| Toy Story   | Toy Story 2, A Bug’s Life, Monsters, Inc., Finding Nemo, Cars                                   |

---

## **Key Features & Strengths**

* Content-based recommendation system
* Handles large datasets efficiently using TF-IDF and cosine similarity
* Flexible and extendable: can incorporate more features (cast, director, keywords)
* Saved models allow deployment without retraining

---

## **Potential Improvements**

* **Item-based collaborative filtering** for better recommendations
* **Matrix factorization (SVD)** for more advanced similarity
* Include **user ratings** to make hybrid recommendations
* Deploy as a **web app or API** using Flask/FastAPI

---

## **Conclusion**

This project demonstrates **NLP preprocessing, feature extraction, and similarity computation** for building a content-based recommendation system.
It’s a strong portfolio project highlighting **practical ML and data engineering skills**.