import streamlit as st 
import eda
import predict

page = st.sidebar.selectbox('Pilih halaman:', ('EDA', 'Pothole Detection'))

if page == 'EDA':
    eda.show_eda()
else :
    predict.show_prediction()