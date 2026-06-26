
import seaborn as sns
import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
import sklearn.linear_model

st.title("Dashboard")
df = pd.read_csv("zomato.csv", encoding='latin1')
st.write("Preview of the dataset:")
st.dataframe(df.head(100))
st.download_button("Download full dataset", df.head(1000).to_csv(index=False), "zomato.csv")

# ---------------- Data Cleaning ----------------
df = df.copy()

# Clean rating column
df['rate'] = df['rate'].astype(str)
df['rate'] = df['rate'].str.replace('/5', '').str.strip()
df['rate'] = df['rate'].replace('NEW', None)
df['rate'] = df['rate'].replace('-', None)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Clean cost column
df['approx_cost(for two people)'] = (
    df['approx_cost(for two people)']
    .astype(str)
    .str.replace(',', '')
)
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'], errors='coerce'
)
    
st.sidebar.header("🔍 Filter Options")

online_filter = st.sidebar.multiselect(
    "Online Order",
    df['online_order'].unique(),
    default=df['online_order'].unique()
)

book_filter = st.sidebar.multiselect(
    "Book Table",
    df['book_table'].unique(),
    default=df['book_table'].unique()
)

filtered_df = df[
    (df['online_order'].isin(online_filter)) &
    (df['book_table'].isin(book_filter))
]

# bar chart - online_order status distribution
st.subheader("Online Order Status Distribution")
order_counts = df['online_order'].value_counts()
fig1, ax1 = plt.subplots()
ax1.bar(order_counts.index, order_counts.values, color=['skyblue', 'salmon'])
ax1.set_xlabel('Online Order')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Online Order Status')
st.pyplot(fig1)

# pie chart - book_table status distribution
st.subheader("Book Table Status Distribution")
table_counts = df['book_table'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(table_counts.values, labels=table_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
ax2.set_title('Distribution of Book Table Status')
st.pyplot(fig2)

# histogram - distribution of votes
st.subheader("Distribution of Votes")
fig3, ax3 = plt.subplots()
ax3.hist(df['votes'].dropna(), bins=30, color='purple', edgecolor='black')
ax3.set_xlabel('Votes')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Votes')
st.pyplot(fig3)

# box plot - distribution of ratings by online_order status
st.subheader("Distribution of Ratings by Online Order Status")
fig4, ax4 = plt.subplots()
sns.boxplot(x='online_order', y='rate', data=df, ax=ax4)
ax4.set_xlabel('Online Order')
ax4.set_ylabel('Rating')
ax4.set_title('Ratings by Online Order Status')
st.pyplot(fig4)

# heatmap - correlation matrix
st.subheader("Correlation Matrix Heatmap")
corr = df.select_dtypes(include=[np.number]).corr()
fig6, ax6 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax6)
ax6.set_title('Correlation Matrix Heatmap')
st.pyplot(fig6)

# horizontal bar chart - top 10 cuisines
st.subheader("Top 10 Cuisines") 
top_cuisines = df['cuisines'].value_counts().head(10)
fig7, ax7 = plt.subplots()
ax7.barh(top_cuisines.index, top_cuisines.values, color='teal')
ax7.set_xlabel('Count')
ax7.set_ylabel('Cuisine')
ax7.set_title('Top 10 Cuisines')
st.pyplot(fig7)

# histogram - distribution of average cost for two
st.subheader("Distribution of Average Cost for Two")    
fig8, ax8 = plt.subplots()
ax8.hist(df['approx_cost(for two people)'].dropna(), bins=30, color='orange', edgecolor='black')
ax8.set_xlabel('Approximate Cost for Two')
ax8.set_ylabel('Frequency')
ax8.set_title('Distribution of Average Cost for Two')
st.pyplot(fig8)

# pie chart - top 5 cuisines
st.subheader("Top 5 Cuisines Distribution")
top5_cuisines = df['cuisines'].value_counts().head(5)
fig9, ax9 = plt.subplots()
ax9.pie(top5_cuisines.values, labels=top5_cuisines.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
ax9.set_title('Top 5 Cuisines Distribution')
st.pyplot(fig9)

# pie chart -cuisine distribution using plotly
st.subheader("Cuisine Distribution (Plotly)")
cuisine_counts = df['cuisines'].value_counts().head(10)
fig10 = px.pie(names=cuisine_counts.index, values=cuisine_counts.values, title='Top 10 Cuisines Distribution')
st.plotly_chart(fig10)

# treemap - visualizing cuisine distribution
st.subheader("Cuisine Distribution Treemap")
cuisine_counts = df['cuisines'].value_counts().head(20).reset_index()
cuisine_counts.columns = ['cuisine', 'count']
fig11 = px.treemap(cuisine_counts, path=['cuisine'], values='count', title='Cuisine Distribution Treemap')
st.plotly_chart(fig11)

# heatmap - ratings vs average cost
st.subheader("Heatmap of Ratings vs Average Cost")
heatmap_data = df.pivot_table(index='rate', columns='approx_cost(for two people)', values='votes', aggfunc='mean')
fig12, ax12 = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', ax=ax12)
ax12.set_title('Heatmap of Ratings vs Average Cost')
st.pyplot(fig12)

#heatmap - online order vs book table
st.subheader("Heatmap of Online Order vs Book Table")
heatmap_data2 = df.pivot_table(index='online_order', columns='book_table', values='votes', aggfunc='mean')
fig13, ax13 = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_data2, cmap='YlOrBr', ax=ax13)
ax13.set_title('Heatmap of Online Order vs Book Table')
st.pyplot(fig13)

# bubble chart - popularity of cuisines
st.subheader("Bubble Chart of Cuisine Popularity")
cuisine_popularity = df['cuisines'].value_counts().reset_index()
cuisine_popularity.columns = ['cuisine', 'count']
fig14 = px.scatter(cuisine_popularity, x='cuisine', y='count',
                    size='count', color='cuisine',
                    title='Cuisine Popularity Bubble Chart',
                    size_max=60)
st.plotly_chart(fig14) 

# histogram - distribution of ratings
st.subheader("Distribution of Ratings")
fig15, ax15 = plt.subplots()
ax15.hist(df['rate'].dropna(), bins=20, color='cyan', edgecolor='black')
ax15.set_xlabel('Ratings')
ax15.set_ylabel('Frequency')
ax15.set_title('Distribution of Ratings')
st.pyplot(fig15)

# bubble chart- price vs rating
st.subheader("Bubble Chart of Price vs Rating")
fig16 = px.scatter(df, x='approx_cost(for two people)', y='rate',
                    size='votes', color='cuisines',
                    title='Price vs Rating Bubble Chart',
                    size_max=60)
st.plotly_chart(fig16)
# pair plot - numerical features
st.subheader("Pair Plot of Numerical Features")
numerical_df = df.select_dtypes(include=[np.number]).dropna()
fig17 = sns.pairplot(numerical_df)
st.pyplot(fig17.fig)              

# horizontal bar chart - top 10 restaurants by votes
st.subheader("Top 10 Restaurants by Votes")
top_restaurants = df[['name', 'votes']].sort_values(by='votes', ascending=False).head(10)
fig18, ax18 = plt.subplots()
ax18.barh(top_restaurants['name'], top_restaurants['votes'], color='magenta')
ax18.set_xlabel('Votes')
ax18.set_ylabel('Restaurant Name')
ax18.set_title('Top 10 Restaurants by Votes')
st.pyplot(fig18)

#histogram - distribution of reviews_list
st.subheader("Distribution of Reviews List")
fig19, ax19 = plt.subplots()
ax19.hist(df['reviews_list'].dropna().apply(lambda x: len(x.split(','))), bins=30, color='brown', edgecolor='black')
ax19.set_xlabel('Number of Reviews')
ax19.set_ylabel('Frequency')
ax19.set_title('Distribution of Reviews List')
st.pyplot(fig19)

st.subheader("📈 Popularity Trend Based on Votes")

votes_trend = filtered_df.groupby('rate')['votes'].mean()

fig, ax = plt.subplots()
votes_trend.plot(kind='line', marker='o', ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Average Votes")
st.pyplot(fig)

st.subheader("🗺 Restaurant Locations Map")


df['lat'] = 12.9250
df['long'] = 77.5938
map_df = df[['lat', 'long', 'name']].dropna()
map_df.columns = ['lat', 'lon', 'name']

st.map(map_df)
st.write(df['location'].value_counts())
st.bar_chart(df['location'].value_counts().head(10))

st.subheader("🍴 Restaurant Recommendation System")

cuisine_input = st.selectbox("Select Cuisine", df['cuisines'].unique(),key='cuisine_select')
budget = st.slider("Max Cost for Two", 100, 3000, 1000,key='budget_slider')

rec_df = filtered_df[
    (filtered_df['cuisines'] == cuisine_input) &
    (filtered_df['approx_cost(for two people)'] <= budget)
].sort_values(by='rate', ascending=False)

st.dataframe(rec_df[['name', 'rate', 'votes']].head(5))

# Remove commas and convert to number
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '')
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

st.subheader("🤖 Rating Prediction Model")

ml_df = filtered_df[['votes', 'approx_cost(for two people)', 'online_order', 'book_table', 'rate']].dropna()

# Convert Yes/No to 1/0
ml_df['online_order'] = ml_df['online_order'].map({'Yes':1, 'No':0})
ml_df['book_table'] = ml_df['book_table'].map({'Yes':1, 'No':0})

X = ml_df[['votes', 'approx_cost(for two people)', 'online_order', 'book_table']]
y = ml_df['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)

st.write("Model Accuracy (R² Score):", round(model.score(X_test, y_test), 2))







