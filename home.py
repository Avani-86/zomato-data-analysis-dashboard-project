import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zomato Restaurant Analysis", page_icon="🍽️", layout="wide")

st.title("🍽Zomato Restaurant Analysis Dashboard")
st.markdown(""" Insights into Food Trends,Rating, and Customer Preferences  """)
st.image("zomato.jpg")

df=pd.read_csv("zomato.csv", encoding='latin1')

st.subheader("📊Key Highlights:")
col1,col2,col3= st.columns(3)
col1.metric("Total Restaurants", f"{df.shape[0]}")
col2.metric("Total Cities", f"{df['location'].nunique()}")
col3.metric("Total Cuisines", f"{df['cuisines'].nunique()}")


st.subheader("🍕 Top 10 Cuisines by Popularity")
top_cuisines = df['cuisines'].value_counts().head(10)
fig, ax = plt.subplots()
top_cuisines.plot(kind='barh', color='orange', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("----")
st.markdown(" © Zomato Data Analysis Dashboard | Created by Awantika Shivhare ")
