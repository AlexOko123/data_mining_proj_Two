import pandas as pd
# Skeleton Driver. It will contain:
# 1. Data Pre-processing and Normalization
# 2. Model Implementation
# 3. Evaluation Methodology


def load_dataset(Dataset):
    print("Reading in File:", Dataset)

    try:
        Gene_Data = pd.read_csv(Dataset)
        # Take Only the Last 3 NOTE!!!! Excel had 3 extra columns I manually removed *
        Gene_Data = Gene_Data.iloc[:, -3:]

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


if __name__ == "__main__":
    # Step 1: Data Pre-processing and Normalization
    # Load your dataset here
    Gene_data = load_dataset("Longotor1delta.csv")
    Gene_data = normalize_data(Gene_data)
    print(Gene_data.head())
    # Perform necessary pre-processing steps (e.g., handling missing values, encoding categorical variables)
    # Normalize the data if required (z-score normalization or min-max scaling)

    # Step 2: Model Implementation
    # Choose unsupervised learning model (e.g., K-Means and fuzzy C-Means)
    # Train the model on the training dataset

    # Step 3: Evaluation Methodology
    # Evaluate the model using appropriate metrics (e.g., silhouette score, Davies-Bouldin index)
    # Print or visualize the results

