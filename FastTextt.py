import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_embedding(sentence, vectorizer):
    return vectorizer.transform([sentence]).toarray()[0]

def load_faq():
    excel_path = "Customer Support FAQ set.xlsx"  # Ensure this file is in the same directory
    df = pd.read_excel(excel_path)
    return df

def get_best_match(user_query, faq_df, vectorizer):
    user_vec = get_sentence_embedding(user_query, vectorizer)
    faq_vectors = vectorizer.transform(faq_df["Question"]).toarray()
    similarities = cosine_similarity([user_vec], faq_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    return faq_df.iloc[best_match_idx]["Answer"], similarities[best_match_idx]

def chat_page():
    
    
    faq_df = load_faq()
    vectorizer = CountVectorizer()
    vectorizer.fit(faq_df["Question"])
    
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        answer, score = get_best_match(user_query, faq_df, vectorizer)
        st.write(f"**Best match score:** {score:.2f}")
        st.success(f"**Answer:** {answer}")
