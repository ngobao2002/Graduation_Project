import streamlit as st
from PIL import Image

st.set_page_config(
    page_title = "Multipage App",
    page_icon="vnuk-symbol-only-official-800x237.png",
    initial_sidebar_state="auto" 
)


page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1562813733-b31f71025d54?q=80&w=2069&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

logo_url = "icons/vnuk-symbol-only-official-800x237.png"
st.sidebar.image(logo_url)

st.title("Main page")
st.sidebar.markdown(
    "<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2024 | baodeptrai</h1>", unsafe_allow_html=True)