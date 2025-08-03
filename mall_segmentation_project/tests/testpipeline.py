import pytest
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from features.preprocessing import load_data, preprocess_data
from models.train import kmeans_clustering

# Fixtures for test setup
@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [19, 21, 22],
        'Annual Income (k$)': [15, 16, 17],
        'Spending Score (1-100)': [39, 81, 6],
        'CustomerID': [1, 2, 3]
    })
    return df

def test_load_data():
    df = load_data("data/mall_customers.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'CustomerID' in df.columns

def test_preprocess_data(sample_data):
    scaled_data, original_df = preprocess_data(sample_data)
    assert scaled_data.shape[0] == original_df.shape[0]
    assert scaled_data.shape[1] == original_df.shape[1]
    assert isinstance(scaled_data, (list, pd.DataFrame)) or hasattr(scaled_data, "shape")

def test_kmeans_clustering(sample_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sample_data.drop(columns=['CustomerID']))
    
    model, labels, score = kmeans_clustering(scaled_data, n_clusters=2)
    assert isinstance(model, KMeans)
    assert len(labels) == scaled_data.shape[0]
    assert 0 <= score <= 1  # Silhouette score range
