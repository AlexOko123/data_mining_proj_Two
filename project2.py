import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Skeleton Driver. It will contain:
# 1. Data Pre-processing and Normalization
# 2. Model Implementation
# 3. Evaluation Methodology


def load_dataset(Dataset):
    print("Reading in File:", Dataset)

    try:
        Gene_Data = pd.read_excel(Dataset, usecols="D:F")
       
        print("raw dataset | peeking data frame head:")
        print(Gene_Data.head())

    except Exception as e:
        print("Ah Hamburgers:", e)
        raise ValueError("Try Again Friend")

    return Gene_Data


def normalize_data(Gene_Data):
    #Using Z-Score Normalization
    #Z-Score Normalization Formula: Z = (X - mean) / std
    Normalize_cols = Gene_Data.select_dtypes(include=['int64', 'float64']).columns

    for col in Normalize_cols:
        #Use Pandas to pull metrics
        mean_val = Gene_Data[col].mean()
        std_val = Gene_Data[col].std()

        if std_val != 0:
            #Z = (X - mean) / std
            Gene_Data[col] = (Gene_Data[col] - mean_val) / std_val
    return Gene_Data

def k_means_cluster(Gene_Data, K):
    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(Gene_Data)
    return cluster_labels, kmeans.cluster_centers_, kmeans.inertia_


def fuzzy_cluster(Gene_Data, K=3, fuzzy_coeff = 2):
    # Fuzzy C Means Cluster Alg Brought to us by Skifuzzy!

    # You have to Convert DF to numpy and TP to fit the function call
    data = Gene_Data.to_numpy().T

    # Run fuzzy c means
    #Centroid is the list of Centroids
    # u is Final Membership of where each point belongs
    # u0 Is Inital Starting positions
    # d is the distance a point is from each centroid
    # jm Objection function tracks loss, Idk what to do with that
    # p is number of Iterations
    # fpc is the fuzzy partition Coefficent
    Centroid, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data,
        c=K,      # Number of clusters
        m=fuzzy_coeff,                 # Fuzziness coefficient
        error=0.005,
        maxiter=1000,
        init=None
    )
    print("Centroid:", Centroid)
    print("u:", u)
    print("u0:", u0)
    print("d:", d)
    print("jm:", jm)
    print("fpc:", fpc)

    # Force the max cluster ownership
    cluster_membership = np.argmax(u, axis=0)

    return cluster_membership, Centroid, u

if __name__ == "__main__":
    # Step 1: Data Pre-processing and Normalization
    # Load your dataset here
    Gene_data = load_dataset("Longotor1delta.xls")
    # Perform necessary pre-processing steps (e.g., handling missing values, encoding categorical variables)
    # Normalize the data if required (z-score normalization or min-max scaling)
    Gene_data = normalize_data(Gene_data)
    print(Gene_data.head())


    # Step 2: Model Implementation
    # Choose unsupervised learning model (e.g., K-Means and fuzzy C-Means)
    # Train the model on the training dataset
    inertia = []
    k_range = range(2, int(np.sqrt(len(Gene_data)))+1)
    #---------- K-MEANS
    for i in range(2,int(np.sqrt(len(Gene_data)))+1):
        cluster_labels, centroids, inertia_val = k_means_cluster(Gene_data, i)
        #print(f"K-Means with K={i}: Centroids: {centroids}")
        inertia.append(inertia_val)

    # visualize k-means with elbow method
    plt.plot(k_range , inertia, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    #---------- FUZZY
    data_membership, Centroids, fuzzy_matrix = fuzzy_cluster(Gene_data, K=3 ,fuzzy_coeff = 2)

    # Step 3: Evaluation Methodology
    # Evaluate the model using appropriate metrics (e.g., silhouette score, Davies-Bouldin index)
    # Print or visualize the results

