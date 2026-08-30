import sys

import streamlit as st


st.title("Upload Probe")

st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)

uploaded = st.file_uploader(
    "Upload a small test PDF or XLSX",
    type=["pdf", "xlsx"],
)

if uploaded is not None:
    st.success("Upload completed")
    st.write("Filename:", uploaded.name)
    st.write("Size:", uploaded.size)
