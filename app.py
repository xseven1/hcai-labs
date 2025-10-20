import streamlit as st
from labs import labnew

st.set_page_config(page_title="HW Manager", page_icon="📚", layout="wide")

st.title("HW Manager")

st.sidebar.header("🧭 Navigation")
st.sidebar.markdown("---")

# Sidebar navigation
selection = st.sidebar.radio("Go to", ["HW1"])

if selection == "HW1":
    labnew.app()
