import streamlit as st

st.set_page_config(
    page_icon=":material/edit:"
)

# Sidebar navigation goes first
with st.sidebar:
    st.header("Lab Manager")

    lab1 = st.Page("lab1.py", title="LAB 1", icon=":material/arrow_outward:")
    lab2 = st.Page("lab2.py", title="LAB 2", icon=":material/arrow_outward:")
    lab3 = st.Page("lab3.py", title="LAB 3", icon=":material/arrow_outward:")
    lab4 = st.Page("lab4.py", title="LAB 4", icon=":material/arrow_outward:")

    pg = st.navigation([lab1, lab2, lab3, lab4])

# Now run the selected page
pg.run()
