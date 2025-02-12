import streamlit as st
from CSV_QNA import chat_page

st.set_page_config(page_title="Semantic matching", page_icon="📚")

st.sidebar.success("Select a task from the sidebar.")

st.markdown("<center><h1 style='text-align: center;'>FAQ Semantic Search using Jaccob and levenshtein_similarity</h1></center>", unsafe_allow_html=True)


chat_page()

