import streamlit as st

st.set_page_config(
    page_title="🧪 Lab Manager",
    page_icon=":material/edit:"
)

# Sidebar navigation goes first
with st.sidebar:
    st.header("Lab Manager")

    lab1 = st.Page("lab1.py", title="LAB 1", icon=":material/arrow_outward:")
    lab2 = st.Page("lab2.py", title="LAB 2", icon=":material/arrow_outward:")
    lab3 = st.Page("lab3.py", title="LAB 3", icon=":material/arrow_outward:")
    lab4 = st.Page("lab4.py", title="LAB 4", icon=":material/arrow_outward:")
    lab5 = st.Page("lab5.py", title="LAB 5", icon=":material/arrow_outward:")
    lab6 = st.Page("lab6.py", title="LAB 6", icon=":material/arrow_outward:")
    hw6  = st.Page("hw6.py",  title="Academic Paper Recommendation Bot",  icon=":material/description:")

    pg = st.navigation([lab1, lab2, lab3, lab4, lab5, lab6, hw6])

# Now run the selected page
pg.run()
