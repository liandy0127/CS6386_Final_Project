# src/dataclean/handle_outliers.py
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Use auto for compatibility
import json

# Define default input and output paths - CORRECTED
# Default input is the output of the previous step (missing handling)
DEFAULT_REVIEWS_INPUT_PATH = "data/processed/reviews_missing_handled.csv" # <-- Correct input
DEFAULT_META_INPUT_PATH = "data/processed/meta_missing_handled.csv"     # <-- Correct input
REVIEWS_OUTPUT_PATH = "data/processed/reviews_outliers_handled.csv"
META_OUTPUT_PATH = "data/processed/meta_outliers_handled.csv"
OUTLIER_STATS_PATH = "data/processed/outlier_stats.json"

# --- Outlier Handling Function (IQR method) ---

def handle_outliers_iqr(df, numeric_cols, cap_method='whisker'):
    """
    Identifies and handles outliers in specified numeric columns using the IQR method.
    Outliers can be capped at the whiskers or replaced with median/mean (default: whisker).
    Returns the handled dataframe and a dictionary of outlier counts per column.
    """
    df_handled = df.copy() # Work on a copy
    total_outliers_capped = 0
    outlier_counts = {}

    print(f"  Handling outliers using IQR method (capping at {cap_method})...")
    for col in tqdm(numeric_cols, desc=" Handling outliers in numeric columns"):
        if col in df_handled.columns:
            # Ensure column is numeric, coercing errors to NaN
            df_handled[col] = pd.to_numeric(df_handled[col], errors='coerce')

            # Drop NaN values for IQR calculation to avoid errors, but remember original length
            col_series = df_handled[col].dropna()
            if len(col_series) < 2: # Need at least 2 non-NaN values for quantiles
                print(f"    Skipping outlier handling for '{col}': Insufficient non-missing data.")
                outlier_counts[col] = 0
                continue

            q1 = col_series.quantile(0.25)
            q3 = col_series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Identify outliers - apply bounds to the original column which might have NaNs
            outlier_mask = (df_handled[col] < lower_bound) | (df_handled[col] > upper_bound)
            num_outliers = outlier_mask.sum()
            outlier_counts[col] = int(num_outliers)

            if num_outliers > 0:
                print(f"    Found {num_outliers} outliers in '{col}'. Applying capping.")
                if cap_method == 'whisker':
                    # Cap outliers at whiskers
                    df_handled[col] = np.where(df_handled[col] < lower_bound, lower_bound, df_handled[col])
                    df_handled[col] = np.where(df_handled[col] > upper_bound, upper_bound, df_handled[col])
                elif cap_method == 'median':
                     median_val = col_series.median()
                     df_handled.loc[outlier_mask, col] = median_val
                     print(f"    Replaced outliers with median ({median_val}).")
                elif cap_method == 'mean':
                     mean_val = col_series.mean()
                     df_handled.loc[outlier_mask, col] = mean_val
                     print(f"    Replaced outliers with mean ({mean_val}).")
                else:
                     print(f"    Warning: Unknown capping method '{cap_method}'. No outliers handled for '{col}'.")

                total_outliers_capped += num_outliers
            else:
                # print(f"    No outliers found in '{col}'.") # Suppress in combined view
                pass
        # else: print(f"    Warning: Numeric column '{col}' not found.") # Suppress in combined view
        else:
            outlier_counts[col] = 0
            # else: print(f"    Warning: Numeric column '{col}' not found.") # Suppress in combined view
    print(f"  Total outliers handled (capped/replaced): {total_outliers_capped}")

    return df_handled, outlier_counts

# --- Main Execution for standalone use ---
if __name__ == "__main__":
    print("Running handle_outliers.py as a standalone script.")
    # Attempt to load output from previous step (missing handling)
    reviews_input_path = DEFAULT_REVIEWS_INPUT_PATH
    meta_input_path = DEFAULT_META_INPUT_PATH

    print(f"Attempting to load data from: {reviews_input_path} and {meta_input_path}")
    try:
        reviews_df = pd.read_csv(reviews_input_path)
        meta_df = pd.read_csv(meta_input_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Could not find files at {reviews_input_path} and {meta_input_path}.")
        # Fallback to previous stages if intermediate files are not found
        print("Attempting to load previous cleaning stage data...")
        try:
            reviews_df = pd.read_csv("../../data/processed/reviews_missing_handled.csv")
            meta_df = pd.read_csv("../../data/processed/meta_missing_handled.csv")
            print("Loaded missing_handled data as fallback.")
        except FileNotFoundError:
            try:
                reviews_df = pd.read_csv("../../data/processed/reviews_text_cleaned.csv")
                meta_df = pd.read_csv("../../data/processed/meta_text_cleaned.csv")
                print("Loaded text_cleaned data as fallback.")
            except FileNotFoundError:
                try:
                     # Try noisy as fallback
                    reviews_df = pd.read_csv("../../data/processed/reviews_noisy.csv")
                    meta_df = pd.read_csv("../../data/processed/meta_noisy.csv")
                    print("Loaded noisy data as fallback.")
                except FileNotFoundError as e:
                    print(f"Error loading fallback data: {e}. Make sure input files exist.")
                    exit()


    print("\nHandling outliers in reviews data...")
    # Define numeric columns for reviews where outliers are relevant (rating)
    reviews_numeric_cols = ['rating', 'helpful_vote'] # Also check helpful_vote

    reviews_df_handled, reviews_outlier_stats = handle_outliers_iqr(reviews_df.copy(), reviews_numeric_cols, cap_method='whisker')

    print("\nHandling outliers in metadata...")
    # Define numeric columns for metadata where outliers are relevant
    meta_numeric_cols = ['average_rating', 'rating_number', 'price']

    meta_df_handled, meta_outlier_stats = handle_outliers_iqr(meta_df.copy(), meta_numeric_cols, cap_method='whisker')

    print(f"\nSaving data with outliers handled to: {REVIEWS_OUTPUT_PATH} and {META_OUTPUT_PATH}")
    reviews_df_handled.to_csv(REVIEWS_OUTPUT_PATH, index=False)
    meta_df_handled.to_csv(META_OUTPUT_PATH, index=False)
    print("Files with outliers handled saved.")

    # Save outlier stats to JSON file
    combined_outlier_stats = {
        "reviews": reviews_outlier_stats,
        "metadata": meta_outlier_stats
    }
    with open(OUTLIER_STATS_PATH, "w") as f:
        json.dump(combined_outlier_stats, f, indent=4)
    print(f"Outlier statistics saved to {OUTLIER_STATS_PATH}.")
