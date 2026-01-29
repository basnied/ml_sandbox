import sys
sys.path.append('../..')
import streamlit as st
from ml_template import utils

st.title("Stats")
st.write(st.session_state.df.head())
fig, dendro = utils.dispaly_cluster_dendrogram(st.session_state.df)
st.pyplot(fig)

if st.button("Let's go training..."):
    st.switch_page('machine_learning.py')