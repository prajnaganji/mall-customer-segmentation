import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set up page
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title(" Mall Customer Segmentation with K-Means")

# Load dataset
st.header(" Load Dataset")
DATA_PATH = "mall_segmentation_project/data/mall_customers.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Select features
st.header(" Select Features for Clustering")
columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
selected_features = st.multiselect("Choose features:", columns, default=["Annual_Income", "Spending_Score"])

if len(selected_features) != 2:
    st.warning("Please select exactly two features for 2D visualization.")
    st.stop()

X = df[selected_features]

# Choose number of clusters
st.header(" Select Number of Clusters")
k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

# KMeans clustering
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X)

# Plot the clusters
st.header(" Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(
    x=X[selected_features[0]],
    y=X[selected_features[1]],
    hue=df["Cluster"],
    palette="tab10",
    ax=ax
)
plt.title(f"Customer Segments (K={k})")
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
st.pyplot(fig)

# Show cluster centroids
st.header(" Cluster Centers")
centroids_df = pd.DataFrame(model.cluster_centers_, columns=selected_features)
st.dataframe(centroids_df)

# Download clustered data
st.header(" Download Clustered Data")
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='clustered_customers.csv',
    mime='text/csv'
)
