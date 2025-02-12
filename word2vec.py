import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def load_faq():
    excel_path = "Customer Support FAQ set.xlsx"
    try:
        df = pd.read_excel(excel_path)
        if "Question" not in df.columns or "Answer" not in df.columns:
            st.error("Excel file must contain 'Question' and 'Answer' columns.")
            return pd.DataFrame(columns=["Question", "Answer"])
        return df
    except Exception as e:
        st.error(f"Error loading FAQ data: {e}")
        return pd.DataFrame(columns=["Question", "Answer"])

def train_word2vec(faq_df):
    sentences = faq_df["Question"].apply(lambda x: str(x).split()).tolist()  # Ensure text is not NaN
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_sentence_embedding(words, model):
    embeddings = [model.wv[word] for word in words if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

def compute_word2vec_similarity(user_query, questions, model):
    user_vec = get_sentence_embedding(user_query.split(), model)
    question_vectors = np.array([get_sentence_embedding(q.split(), model) for q in questions])
    similarities = cosine_similarity([user_vec], question_vectors).flatten()
    return similarities

def get_best_match(user_query, faq_df, model):
    questions = faq_df["Question"].fillna("").tolist()  # Handle NaN values
    
    word2vec_similarities = compute_word2vec_similarity(user_query, questions, model)
    
    if word2vec_similarities.max() == 0:
        return "No relevant match found.", 0.0

    best_match_idx = np.argmax(word2vec_similarities)
    return faq_df.iloc[best_match_idx]["Answer"], word2vec_similarities[best_match_idx]

def chat_page():
    st.title("FAQ Semantic Search with Word2Vec")
    
    faq_df = load_faq()
    if faq_df.empty:
        st.error("FAQ data could not be loaded. Please check the Excel file.")
        return

    model = train_word2vec(faq_df)

    user_query = st.text_input("Ask a question:")

    if user_query:
        answer, score = get_best_match(user_query, faq_df, model)
        st.write(f"**Best match score:** {score:.2f}")
        st.success(f"**Answer:** {answer}")
