import streamlit as st
from t5 import chat_page

st.set_page_config(page_title="Semantic matching", page_icon="📚")

st.sidebar.success("Select a task from the sidebar.")

st.markdown("<center><h1 style='text-align: center;'>FAQ Semantic Search using T5</h1></center>", unsafe_allow_html=True)


chat_page()

