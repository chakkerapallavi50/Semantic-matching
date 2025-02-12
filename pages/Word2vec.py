import streamlit as st
from word2vec import chat_page

st.set_page_config(page_title="Semantic matching", page_icon="📚")

st.sidebar.success("Select a task from the sidebar.")


chat_page()

