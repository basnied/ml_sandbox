import streamlit as st
import pandas as pd


@st.cache_data
def read_table(file_path):
    return pd.read_csv(file_path, delimiter = ";")


st.title("ML GUI")

st.file_uploader("Upload you dataframe here!",
                    type=['txt', 'csv'],
                    accept_multiple_files=False,
                    key='file')

if "file" in st.session_state and st.session_state.file is not None:
    try:
        st.session_state.df = read_table(st.session_state.file)
    except Exception as e:
        print(e)
    st.write(st.session_state.df.head())

    with st.expander('Define Input data'):
        date_col, target_col = st.columns(2, gap = 'medium')
        with date_col:
            st.selectbox("Select datetime column",
                            list(st.session_state.df.columns),
                            key='date_col')

    if st.button("Next step: Stats!"):
        st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df[st.session_state.date_col])
        st.session_state.df.set_index("datetime", inplace=True)
        try:
            st.session_state.df.drop(columns=st.session_state.date_col, inplace=True)
        except:
            pass
        st.switch_page('stats.py')
