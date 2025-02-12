import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import difflib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Remove stopwords & stem
    return text

def load_faq():
    # Load the FAQ data from an Excel file
    excel_path = "Customer Support FAQ set.xlsx"  # Ensure this file is in the same directory
    df = pd.read_excel(excel_path)
    df["Processed_Question"] = df["Question"].apply(preprocess_text)
    return df

def compute_tfidf_similarity(user_query, questions):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Use unigrams, bigrams, and trigrams
    tfidf_matrix = vectorizer.fit_transform(questions)
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return similarities

def compute_jaccard_similarity(user_query, questions):
    vectorizer = CountVectorizer(binary=True)
    question_matrix = vectorizer.fit_transform(questions).toarray()
    query_vec = vectorizer.transform([user_query]).toarray()
    similarities = 1 - pairwise_distances(query_vec, question_matrix, metric='jaccard')
    return similarities.flatten()

def compute_shtein_similarity(user_query, questions):
    similarities = [difflib.SequenceMatcher(None, user_query.lower(), q.lower()).ratio() for q in questions]
    return np.array(similarities)

def compute_word_overlap_similarity(user_query, questions):
    user_set = set(user_query.split())
    similarities = [len(user_set.intersection(set(q.split()))) / len(user_set.union(set(q.split()))) for q in questions]
    return np.array(similarities)

def get_best_match(user_query, faq_df):
    user_query = preprocess_text(user_query)
    questions = faq_df["Processed_Question"].tolist()
    
    tfidf_similarities = compute_tfidf_similarity(user_query, questions)
    jaccard_similarities = compute_jaccard_similarity(user_query, questions)
    levenshtein_similarities = compute_levenshtein_similarity(user_query, questions)
    word_overlap_similarities = compute_word_overlap_similarity(user_query, questions)
    
    # Normalize similaritlevenies to prevent bias towards any metric
    tfidf_similarities /= tfidf_similarities.max() if tfidf_similarities.max() > 0 else 1
    jaccard_similarities /= jaccard_similarities.max() if jaccard_similarities.max() > 0 else 1
    levenshtein_similarities /= levenshtein_similarities.max() if levenshtein_similarities.max() > 0 else 1
    word_overlap_similarities /= word_overlap_similarities.max() if word_overlap_similarities.max() > 0 else 1
    
    # Weighted combination of similarity scores (More weight to TF-IDF Cosine Similarity)
    combined_similarity = (tfidf_similarities * 0.6) + (jaccard_similarities * 0.15) + (levenshtein_similarities * 0.15) + (word_overlap_similarities * 0.1)
    
    best_match_idx = np.argmax(combined_similarity)
    return faq_df.iloc[best_match_idx]["Answer"], combined_similarity[best_match_idx]
def chat_page():
    
    faq_df = load_faq()  # Ensure load_faq() is defined correctly in CSV_QNA.py
    
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        answer, score = get_best_match(user_query, faq_df)
        st.write(f"**Best match score:** {score:.2f}")
        st.success(f"**Answer:** {answer}")
