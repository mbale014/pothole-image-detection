import streamlit as st
import os
import random
import pandas as pd
from PIL import Image

def show_eda(data_dir="data"):
    st.title("Exploratory Data Analysis (EDA)")
    image = Image.open('deeplearning.png')
    st.image(image, caption='figure 1.1 Introduction ')
    
    #Introduction
    st.subheader('About')
    st.write(' Project by ***Muhammad iqbal***')
    st.write(' Graded Challenge 7 ')
    st.write(' Hacktiv8 RMT-040')
    
    st.markdown('---')
    
    # 1. Sample Images
    st.header("1. Sample Images from Both Classes")
    classes = ['normal', 'pothole']
    cols = st.columns(2)

    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        sample_images = random.sample(os.listdir(cls_path), 3)
        with cols[i]:
            st.subheader(cls.capitalize())
            for img_name in sample_images:
                img = Image.open(os.path.join(cls_path, img_name))
                st.image(img, caption=img_name, use_container_width=True)

    # 2. Basic Stats
    st.header("2. Basic Dataset Statistics")
    df_stats = pd.DataFrame({'Class':['Normal', 'Pothole'],
                             'Count':[352, 329]})
    st.table(df_stats)

    # 3. Class Distribution Chart
    st.header("3. Class Distribution")
    st.bar_chart(df_stats.set_index("Class")["Count"])
