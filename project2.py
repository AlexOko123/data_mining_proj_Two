import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as S_Score
from sklearn.metrics import calinski_harabasz_score as CH_score
from sklearn.metrics import davies_bouldin_score as DB_score

import os


# Skeleton Driver. It will contain:
# 1. Data Pre-processing and Normalization
# 2. Model Implementation
# 3. Evaluation Methodology

# ===================================== Data loading and processing
def load_dataset(Dataset, view_raw_data=False):
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
    # Z-Score Normalization Formula: Z = (X - mean) / std
    Normalize_cols = Gene_Data.select_dtypes(include=['int64', 'float64']).columns

    for col in Normalize_cols:

        # Use Pandas to pull metrics
        mean_val = Gene_Data[col].mean()
        std_val = Gene_Data[col].std()

        if std_val != 0:
            # Z = (X - mean) / std
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
    k_range = range(2, int(np.sqrt(len(Gene_data))) + 1)

    # values needed to find optimal K value (elbow method)
    for i in k_range:
        model = k_means_cluster(Gene_data, i)
        inertia_val = model.inertia_
        inertia.append(inertia_val)

    # plotting elbow
    plt.plot(k_range, inertia, marker='')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()


# ========================================================== fuzzy c - means
def fuzzy_model_summary(centroid_list, final_feature_matrix, initial_feature_matrix,
                        dist_from_centroid, objective_func, num_of_iterations, fuzz_partition_coef):
    print("Centroids:", centroid_list)
    print("initial matrix:\t", initial_feature_matrix)
    print("final matrix:\t", final_feature_matrix)
    print("distance from centroid:\t", dist_from_centroid)
    print("objective function:\t", objective_func)
    print("fuzzy partition coefficent:\t", fuzz_partition_coef)


def fuzzy_cluster(Gene_Data, K=3, fuzzy_coeff=2, view_model_summary=False):
    # Fuzzy C Means Cluster Alg Brought to us by Skifuzzy!

    # You have to Convert DF to numpy and TP to fit the function call
    data = Gene_Data.to_numpy().T

    centroid, u0, u, dist, jm, p, fpc = fuzz.cluster.cmeans(
        data,
        c=K,  # Number of clusters
        m=fuzzy_coeff,  # Fuzziness coefficient
        error=0.005,
        maxiter=1000,
        init=None
    )

    # variable renaming
    initial_membership_matrix = u0     # Inital Starting positions
    final_membership_matrix = u        # Final Membership of where each point belongs
    dist_from_centroid = dist          # the distance a point is from each centroid
    objective_func = jm                # Objection function tracks loss, Idk what to do with that
    num_of_iterations = p              # number of Iterations
    fuzz_partition_coef = fpc          # fuzzy partition Coefficent

    if view_model_summary != False:
        fuzzy_model_summary(centroid,
                            u, u0,
                            dist_from_centroid, objective_func,
                            num_of_iterations, fuzz_partition_coef)

    # Force the max cluster ownership
    cluster_membership = np.argmax(u, axis=0)

    return cluster_membership, centroid, final_membership_matrix, fuzz_partition_coef


# ======================================================== visualization methods
def dimension_reduction(to_be_reduced):
    magical_reduction_variable = PCA(n_components=2)

    # --- X and Y dimension reduction
    if (to_be_reduced.shape[1] > 2):
        reduced_data = magical_reduction_variable.fit_transform(to_be_reduced);

    else:
        reduced_data = to_be_reduced

    X = reduced_data[:, 0]
    Y = reduced_data[:, 1]

    return X, Y


def plot_kmeans(x, y, cent_x, cent_y, labels):
    plt.scatter(x=x, y=y, c=labels, marker='x', cmap='viridis', s=10)
    plt.scatter(x=cent_x, y=cent_y, c='red', marker='o', s=15, )
    plt.title("K-Means Clusters")
    plt.show()


def plot_fuzzy_c(x, y, cent_x, cent_y, labels, membership_matrix):
    # more confident classifications have larger data markers
    confidence = membership_matrix[labels, np.arange(len(labels))]
    sizes = confidence * 50

    plt.scatter(x=x, y=y, c=labels, marker='x', cmap='viridis', s=sizes)
    plt.scatter(x=cent_x, y=cent_y, c='red', marker='o', s=20, )
    plt.title("fuzzy c-Means Clusters")
    plt.show()


# ======================================================= evaluation metrics

def print_SScore(SScore):
    # note: measures both separation & cohesion. works at the point level
    # measures a single point's dist from points in its own cluster
    # then dist between points in neighboring cluster. then boring math steps get final S score

    print()
    if (-1.0 <= SScore < 0):
        print(f"Silhouette Score :: {SScore} | S-Score < 0.0; bad clustering")

    elif (0 <= SScore < 0.25):
        print(f"Silhouette Score :: {SScore} | 0.0 <= S-Score < 0.25; poor clustering")

    elif (0.25 <= SScore < 0.5):
        print(f"Silhouette Score :: {SScore} | 0.25 <= S-Score < 0.50; weak clustering")

    elif (0.5 <= SScore < 0.7):
        print(f"Silhouette Score :: {SScore} | 0.50 <= S-Score < 0.70; reasonable clustering")

    elif (0.7 <= SScore <= 1):
        print(f"Silhouette Score :: {SScore} | 0.70 <= S-Score <= 1.0; strong clustering")


def print_CH_score(CHScore):
    # note: measures both separation and cohesion, works at the cluster level not point level
    # measures a point's dist from it's centroid & a cluster's centroid's dist from global cent
    print()
    print(f"Calinski Score :: {CHScore}\t| ranges 0 -> +infinity; the higher the better")


def print_DB_score(DBScore):
    # note: measures separation & cohesion. works on a cluster level
    # checks how far the ave dist is between a cluster's centroid and assigned points
    # checks how far the cluster itself is from it's neighbor. the tighter and further the
    print()
    print(f"Davies Bouldin Score :: {DBScore} | ranges 0 -> +infinity; the closer to 0 the better ")


# ======================================================= main
if __name__ == "__main__":

    # ===== step 1: data pre-processing and normalization
    print("time to setup the data, big guy")
    print()

    # load dataset | preprocessing | normalization
    # e.g., handling missing values, encoding categorical variables
    # z-score normalization or min-max scaling

    Gene_data = load_dataset("Longotor1delta.xls")
    Gene_data = z_score_normalization(Gene_data)

    # ======= step 2: model 1 implementation | fuzzy C - means
    print()
    print("building fuzzy c-means model :: ")

    data_membership, Centroids, membership_matrix, fpc = fuzzy_cluster(Gene_data, K=3, fuzzy_coeff=2)

    # ====== step 2.1: model 2 implementation | K - means
    print()
    print("ohhh yeah, fuzzy c-means done. building Kmeans now :: ")

    print()
    if input("view kmeans Elbow graph <y/n> :: ") == 'y':
        test_for_best_k_val()

    k_val = 13
    print()
    usr_inp = input("pick a value for K <enter 0 for default: k = 13 > :: ")

    if (int)(usr_inp) > 1:
        k_val = (int)(usr_inp)

    km_model = k_means_cluster(Gene_Data=Gene_data, k=k_val)

    # gets labels
    km_pred_y = km_model.labels_

    # ====== step 3: visualizations for kmeans and fuzzy c means
    print()
    if input("view kmeans clustering graph <y/n> :: ") == 'y':
        print("loading kmeans visualizer")

        # dataset X & Y dimension reduction | centroid dimension reduction
        red_x, red_y = dimension_reduction(Gene_data)
        red_cent_x, red_cent_y = dimension_reduction(km_model.cluster_centers_)

        # plot kmeans
        plot_kmeans(red_x, red_y, red_cent_x, red_cent_y, km_pred_y)

    print()
    if input("view fuzzy c-means graph <y/n> :: ") == 'y':
        print("loading fuzzy c-means visualizer")

        # centroid dimension reduction
        red_x, red_y = dimension_reduction(Gene_data)
        red_cent_x, red_cent_y = dimension_reduction(Centroids)

        # plot fuzzy c-means
        plot_fuzzy_c(red_x, red_y, red_cent_x, red_cent_y, data_membership, membership_matrix)

    # ====== step 3: evaluation and final outputs
    print()
    print("we're freaking doing the final evaluation now")

    print()
    if input('clear screen for metrics <y/n> ') == 'y':
        os.system('cls')

    print()
    print("showing kmeans evaluation metrics")

    # cluster quality metrics | how good are the clusters themselves
    print_SScore(S_Score(Gene_data, km_pred_y))
    print_CH_score(CH_score(Gene_data, km_pred_y))
    print_DB_score(DB_score(Gene_data, km_pred_y))

    print()
    print()
    print("showing fuzzy c means evaluation metrics")

    # note : partition shows how much overlap 'fuzz' the data has
    # values closer to 1 mean less overlap (less fuzz) and more clear partitions
    print()
    print(
        f"fuzz partition coefficient :: {fpc} | range is 0 -> 1; closer to 1 means less overlap and clearer partitions")

print()
