from features.preprocessing import load_data, preprocess_data
from models.train import kmeans_clustering

def main():
    print(" main.py is running...")

    try:
        print(" Loading data...")
        df = load_data("data/mall_customers.csv")

        print(" Columns:", df.columns.tolist())
        print(" Data loaded. Shape:", df.shape)

        print(" Preprocessing data...")
        X_scaled, original_df = preprocess_data(df)
        print(" Preprocessing complete. Scaled shape:", X_scaled.shape)

        print(" Running KMeans clustering...")
        model, labels, score = kmeans_clustering(X_scaled)

        print(f" Clustering done. Silhouette Score: {score:.2f}")

        original_df["Cluster"] = labels
        print(" Clustered data preview:")
        print(original_df.head())

        # Optional output
        original_df.to_csv("data/clustered_output.csv", index=False)
        print(" Results saved to data/clustered_output.csv")

    except Exception as e:
        print(" Pipeline failed:")
        print(str(e))

if __name__ == "__main__":
    print(" Entry point reached.")
    main()
    print(" Script finished.")
