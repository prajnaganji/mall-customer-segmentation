from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_clustering(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return model, labels, score
