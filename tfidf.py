import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_faq():
    # Load the FAQ data from a CSV file
    csv_path = "Customer Support FAQ set.xlsx"  # Ensure this file is in the same directory
    return pd.read_excel(csv_path)

def compute_tfidf_similarity(user_query, questions):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return similarities

def get_best_match(user_query, faq_df):
    questions = faq_df["Question"].tolist()
    similarities = compute_tfidf_similarity(user_query, questions)
    best_match_idx = np.argmax(similarities)
    return faq_df.iloc[best_match_idx]["Answer"], similarities[best_match_idx]

def chat_page():
   
    faq_df = load_faq()  # Ensure load_faq() is defined correctly in CSV_QNA.py
    
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        answer, score = get_best_match(user_query, faq_df)
        st.write(f"**Best match score:** {score:.2f}")
        st.success(f"**Answer:** {answer}")
