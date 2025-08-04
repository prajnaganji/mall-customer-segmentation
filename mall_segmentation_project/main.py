import streamlit as st
from features.preprocessing import load_data, preprocess_data
from models.train import kmeans_clustering
import pandas as pd

def main():
    st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
    st.title("🛍️ Mall Customer Segmentation with K-Means")

    # Load Data
    st.subheader("📥 Load Dataset")
    try:
        df = load_data("data/mall_customers.csv")
        st.success("Data Loaded Successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Preprocessing
    st.subheader("🔧 Preprocess Data")
    try:
        X_scaled, original_df = preprocess_data(df)
        st.write("Scaled Data Shape:", X_scaled.shape)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    # Clustering
    st.subheader("🔄 Run K-Means Clustering")
    try:
        result_df = kmeans_clustering(X_scaled, original_df, n_clusters=5)
        st.success("Clustering Complete!")
        st.dataframe(result_df.head())

        # Optionally show cluster counts
        st.bar_chart(result_df['Cluster'].value_counts())
    except Exception as e:
        st.error(f"Clustering failed: {e}")

if __name__ == "__main__":
    main()
