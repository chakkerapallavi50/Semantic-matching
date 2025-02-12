import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics.pairwise import cosine_similarity

# Load T5 model and tokenizer once (to avoid reloading for every query)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5EncoderModel.from_pretrained("t5-small")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def load_faq():
    excel_path = "Customer Support FAQ set.xlsx"
    try:
        df = pd.read_excel(excel_path)
        if "Question" not in df.columns or "Answer" not in df.columns:
            st.error("Excel file must contain 'Question' and 'Answer' columns.")
            return pd.DataFrame(columns=["Question", "Answer"])
        
        # Compute embeddings and store them
        df["Embedding"] = df["Question"].apply(lambda q: encode_text(str(q)))
        return df
    except Exception as e:
        st.error(f"Error loading FAQ data: {e}")
        return pd.DataFrame(columns=["Question", "Answer"])

def get_best_match(user_query, faq_df):
    if faq_df.empty:
        return "No data available.", 0.0

    user_vec = encode_text(user_query)
    faq_vectors = np.vstack(faq_df["Embedding"].values)
    similarities = cosine_similarity(user_vec, faq_vectors).flatten()

    if similarities.max() == 0:
        return "No relevant match found.", 0.0

    best_match_idx = np.argmax(similarities)
    return faq_df.iloc[best_match_idx]["Answer"], similarities[best_match_idx]

def chat_page():
    

    faq_df = load_faq()
    if faq_df.empty:
        st.error("FAQ data could not be loaded. Please check the Excel file.")
        return

    user_query = st.text_input("Ask a question:")

    if user_query:
        answer, score = get_best_match(user_query, faq_df)
        st.write(f"**Best match score:** {score:.2f}")
        st.success(f"**Answer:** {answer}")
