import streamlit as st
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
import shap

st.title('Results')
X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, 
                                                    st.session_state.y, 
                                                    train_size=.7, 
                                                    shuffle=True, 
                                                    random_state=42)

y_pred = st.session_state.model.predict(X_test)

st.write(f"""
         Your Model achieved the following performance metrics:\\
            R²: {round(r2_score(y_test, y_pred), 3)}\\
            RMSE: {round(root_mean_squared_error(y_test, y_pred), 3)}
            """)

explainer = shap.TreeExplainer(st.session_state.model)
shap_values = explainer.shap_values(X_train)
fig, ax = plt.subplots()
st.pyplot(fig, shap.summary_plot(shap_values, X_train, plot_type="dot"))

with st.form('ts_plot'):
    with st.expander('TS plot params'):
        cmap_col, trans_col = st.columns(2, gap='medium')
        with cmap_col:
            st.selectbox("select colormap",
                         plt.colormaps(),
                         key='color')
        with trans_col:
            st.slider('Opacity', min_value=0.0, max_value=1.0,
                      step=.1,
                      value=1.0,
                      key='opacity')
    if st.form_submit_button("Create Chart"):
        shap_values_doy = pd.DataFrame(shap_values).groupby(X_train.index.day_of_year).mean().to_numpy()
        features_doy = st.session_state.X.groupby(st.session_state.X.index.day_of_year).mean()
        targets_doy = st.session_state.y.groupby(st.session_state.y.index.day_of_year).mean()
        sorted = pd.DataFrame(shap_values_doy, columns=features_doy.columns).mean().abs().sort_values().index.to_list()
        shap_melt = pd.DataFrame(shap_values_doy, columns=features_doy.columns).melt(ignore_index=False)
        fig, ax = plt.subplots(1,1, figsize=(15,8), dpi=500)
        sns.barplot(data=shap_melt, 
                    x=shap_melt.index, 
                    y='value', 
                    hue='variable', 
                    dodge=False, 
                    palette=st.session_state.color, 
                    fill=True, 
                    gap=0, 
                    width=1, 
                    hue_order=sorted, 
                    alpha=st.session_state.opacity)
        plt.plot(targets_doy, label=y_train.name, color='black', linestyle='--')
        ticks = ax.set_xticks(list(np.arange(0,375,25)))
        ticks_lab = ax.set_xticklabels(list(np.arange(0,375,25)))
        plt.margins(x=0)
        sns.set_style('dark')
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig)