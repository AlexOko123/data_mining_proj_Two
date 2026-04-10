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

# ===================================== Data loading and processing
def load_dataset(Dataset, view_raw_data = False):
    print("Reading in File:", Dataset)

    try:
        Gene_Data = pd.read_excel(Dataset, usecols="D:F")

        if view_raw_data != False:

            print("raw dataset | peeking data frame head:")
            print(Gene_Data.head())

    except Exception as e:
       
        print("Ah Hamburgers:", e)
        raise ValueError("Try Again Friend")

    return Gene_Data


def z_score_normalization(Gene_Data):

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


# ========================================================= kmeans
def k_means_cluster(Gene_Data, k):
    # create and fit model to data
    model = KMeans(n_clusters=k, random_state=42)
    model.fit_predict(Gene_Data)

    return model


def test_for_best_k_val():

    inertia = []
    k_range = range(2, int(np.sqrt(len(Gene_data)))+1)

    # values needed to find optimal K value (elbow method) 
    for i in k_range:
        model = k_means_cluster(Gene_data, i)
        inertia_val = model.inertia_
        inertia.append(inertia_val)

    # plotting elbow
    plt.plot(k_range , inertia, marker='.')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()


# ========================================================== fuzzy c - means
def fuzzy_model_summary(centroid_list, final_feature_matrix, initial_feature_matrix, 
                        dist_from_centroid, objective_func, num_of_iterations, fuzz_partition_coef): 

    print("Centroids:", Centroid_list)
    print("initial matrix:\t", initial_feature_matrix)
    print("final matrix:\t", final_feature_matrix)
    print("distance from centroid:\t", dist_from_centroid)
    print("objective function:\t", objective_func)
    print("fuzzy partition coefficent:\t", fuzz_partition_coef)


def fuzzy_cluster(Gene_Data, K=3, fuzzy_coeff = 2, view_model_summary = False):
    # Fuzzy C Means Cluster Alg Brought to us by Skifuzzy!

    # You have to Convert DF to numpy and TP to fit the function call
    data = Gene_Data.to_numpy().T

    centroid, u0, u, dist, jm, p, fpc = fuzz.cluster.cmeans(
        data,
        c=K,           # Number of clusters
        m=fuzzy_coeff, # Fuzziness coefficient
        error=0.005,
        maxiter=1000,
        init=None
    )

    # variable renaming
    initial_membership_matrix  = u0  # Final Membership of where each point belongs
    final_membership_matrix = u # Inital Starting positions
    dist_from_centroid = dist  # the distance a point is from each centroid
    objective_func = jm        # Objection function tracks loss, Idk what to do with that
    num_of_iterations = p      # number of Iterations
    fuzz_partition_coef = fpc  # fuzzy partition Coefficent

    if view_model_summary != False:
        fuzzy_model_summary(centroid_list, 
                            final_feature_matrix, initial_feature_matrix, 
                            dist_from_centroid, objective_func, 
                            num_of_iterations, fuzz_partition_coef)

    # Force the max cluster ownership
    cluster_membership = np.argmax(u, axis=0)

    return cluster_membership, centroid, final_membership_matrix


# ======================================================== visualization methods
def dimension_reduction(to_be_reduced):

    magical_reduction_variable = PCA(n_components = 2)

    # --- X and Y dimension reduction 
    if(to_be_reduced.shape[1] > 2):
        reduced_data = magical_reduction_variable.fit_transform(to_be_reduced); 
    
    else:
        reduced_data = to_be_reduced

    X = reduced_data[:,0]
    Y = reduced_data[:,1]

    return X, Y

def plot_kmeans(x, y, cent_x, cent_y, labels):
    
    plt.scatter(x=x, y=y, c=labels, marker='x', cmap='viridis', s=10 )
    plt.scatter(x=cent_x, y=cent_y, c='red', marker='o', s=15, )
    plt.title("K-Means Clusters")
    plt.show() 


def plot_fuzzy_c(x, y, cent_x, cent_y, labels, membership_matrix):
    
    confidence = membership_matrix[labels, np.arange(len(labels))]
    sizes = confidence * 50

    plt.scatter(x=x, y=y, c=labels, marker='x', cmap='viridis', s=sizes)
    plt.scatter(x=cent_x, y=cent_y, c='red', marker='o', s=20, )
    plt.title("fuzzy c-Means Clusters")
    plt.show() 


# ======================================================= main
if __name__ == "__main__":
    
    view_elbow = False 
    k_val = 10
    
    # ===== step 1: data pre-processing and normalization 
    print("time to setup the data, big guy")

    # load dataset | preprocessing | normalization
    # e.g., handling missing values, encoding categorical variables
    # z-score normalization or min-max scaling

    Gene_data = load_dataset("Longotor1delta.xls")
    Gene_data = z_score_normalization(Gene_data)

    # ======= step 2: model 1 implementation | fuzzy C - means
    print("building fuzzy c-means model :: ")
    
    data_membership, Centroids, membership_matrix = fuzzy_cluster(Gene_data, K=3 ,fuzzy_coeff = 2)

    # ====== step 2.1: model 2 implementation | K - means
    print("ohhh yeah, fuzzy c-means done. building Kmeans now :: ")
    
    if view_elbow == True:
        test_for_best_k_val()
        input('press enter to continue')

    km_model = k_means_cluster(Gene_Data=Gene_data, k=k_val)

    # ====== step 3: visualizations for kmeans and fuzzy c means
    print("loading kmeans visualizer")

    red_x, red_y = dimension_reduction(Gene_data)
    red_cent_x, red_cent_y = dimension_reduction(km_model.cluster_centers_)

    labels = km_model.labels_

    plot_kmeans(red_x, red_y, red_cent_x, red_cent_y, labels)
    input('press enter to continue')

    print("loading fuzzy c-means visualizer")

    red_cent_x, red_cent_y = dimension_reduction(Centroids)

    plot_fuzzy_c(red_x, red_y, red_cent_x, red_cent_y, data_membership, membership_matrix)
    input('press enter to continue')

    
    # ====== step 3: evaluation and final outputs
    print("we're freaking doing the final evaluation now")

