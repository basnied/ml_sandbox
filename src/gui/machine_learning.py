import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score


st.title("Model training 💪")

with st.form('ml_params'):
    st.selectbox("Select target column",
                    list(st.session_state.df.columns),
                    key='target_col')

    if 'target_col' in st.session_state:
        st.multiselect("Select features.",
                        list(st.session_state.df.drop(columns=st.session_state.target_col).columns),
                        key='feature_col')
        
    if st.form_submit_button("Submit model params"):
        X = st.session_state.df[st.session_state.feature_col]
        y = st.session_state.df[st.session_state.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True, random_state=42)

                
        model = xgb.XGBRegressor(n_estimators=2500, 
                         max_depth=4, 
                         learning_rate=.03, gamma=14, 
                         early_stopping_rounds=50, 
                         min_child_weight=3, 
                         colsample_bytree=1, 
                         subsample=.5, 
                         eval_metric=root_mean_squared_error)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        st.session_state.model = model
        st.session_state.X = X
        st.session_state.y = y
        
        st.switch_page('results.py')


