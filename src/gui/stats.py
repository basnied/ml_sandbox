import sys
import streamlit as st
import utils

st.title("Stats")
st.write(st.session_state.df.head())
fig, dendro = utils.dispaly_cluster_dendrogram(st.session_state.df)
st.pyplot(fig)

if st.button("Let's go training..."):
    st.switch_page('machine_learning.py')