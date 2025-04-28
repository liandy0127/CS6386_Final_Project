# src/dataclean/denoise.py
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Import cleaning functions from individual scripts
# Assuming the scripts are in the same directory or accessible via relative import
try:
    # Correct imports assuming scripts are in the same package/directory
    from .clean_text import apply_text_cleaning # Only need the apply function
    from .handle_missing import handle_missing_values, TEXT_PLACEHOLDER # Need function and placeholder
    from .handle_outliers import handle_outliers_iqr # Need the function
except ImportError:
    print("Could not perform relative imports. Ensure scripts are in a package or adjust import paths.")
    print("Attempting direct imports (may fail if scripts are not in Python path)...")
    try:
        # Fallback for direct execution if not in a package
        from clean_text import apply_text_cleaning
        from handle_missing import handle_missing_values, TEXT_PLACEHOLDER
        from handle_outliers import handle_outliers_iqr
    except ImportError:
         print("Direct imports also failed. Please check your script location and Python path.")
         print("Cannot run denoise pipeline without cleaning functions.")
         # Define dummy functions to allow script to run without crashing, but it won't clean
         def apply_text_cleaning(reviews_df, meta_df): print("Dummy text cleaning..."); return reviews_df, meta_df
         def handle_missing_values(df, numeric_cols, text_cols, placeholder): print("Dummy missing handling..."); return df
         def handle_outliers_iqr(df, numeric_cols, cap_method): print("Dummy outlier handling..."); return df
         TEXT_PLACEHOLDER = "[Placeholder Not Imported]"


# Define input (baseline - noisy data) and final output paths - CORRECTED
REVIEWS_INPUT_PATH_BASELINE = "../../data/processed/reviews_noisy.csv" # <-- Correct baseline input
META_INPUT_PATH_BASELINE = "../../data/processed/meta_noisy.csv"     # <-- Correct baseline input
REVIEWS_OUTPUT_PATH_DENOISED = "../../data/processed/reviews_denoised_combined.csv"
META_OUTPUT_PATH_DENOISED = "../../data/processed/meta_denoised_combined.csv"

# --- Full Denoising Pipeline Function ---

def denoise_data_pipeline(reviews_df, meta_df):
    """
    Applies the full denoising pipeline by calling imported functions:
    Text Cleaning -> Missing Values -> Outliers.
    """
    print("Starting full denoising pipeline...")

    # Step 1: Text Cleaning
    print("\n--- Step 1: Text Cleaning ---")
    # apply_text_cleaning modifies dataframes in place and returns them
    # The dataframes passed here are the noisy ones loaded in main
    reviews_df, meta_df = apply_text_cleaning(reviews_df, meta_df)
    print("--- Text Cleaning Step Complete ---")

    # Step 2: Handling Missing Values
    print("\n--- Step 2: Handling Missing Values ---")
    # Define columns to handle missing values - include original and cleaned columns
    reviews_numeric_cols_missing = ['rating', 'helpful_vote']
    # Target all object/string cols (includes original and cleaned text)
    reviews_text_cols_missing = [col for col in reviews_df.columns if reviews_df[col].dtype == 'object' or reviews_df[col].dtype == 'string']

    meta_numeric_cols_missing = ['average_rating', 'rating_number', 'price']
    meta_text_cols_missing = [col for col in meta_df.columns if meta_df[col].dtype == 'object' or meta_df[col].dtype == 'string']

    # handle_missing_values returns a modified copy
    # Pass the dataframes modified by the previous step
    reviews_df = handle_missing_values(reviews_df, reviews_numeric_cols_missing, reviews_text_cols_missing, TEXT_PLACEHOLDER)
    meta_df = handle_missing_values(meta_df, meta_numeric_cols_missing, meta_text_cols_missing, TEXT_PLACEHOLDER)
    print("--- Missing Value Handling Step Complete ---")

    # Step 3: Handling Outliers
    print("\n--- Step 3: Handling Outliers ---")
    # Define numeric columns for outlier handling
    reviews_numeric_cols_outliers = ['rating', 'helpful_vote']
    meta_numeric_cols_outliers = ['average_rating', 'rating_number', 'price']

    # handle_outliers_iqr returns a modified copy
    # Pass the dataframes modified by the previous step
    reviews_df = handle_outliers_iqr(reviews_df, reviews_numeric_cols_outliers, cap_method='whisker')
    meta_df = handle_outliers_iqr(meta_df, meta_numeric_cols_outliers, cap_method='whisker')
    print("--- Outlier Handling Step Complete ---")

    print("\nFull denoising pipeline complete.")
    return reviews_df, meta_df

# --- Main Execution ---

if __name__ == "__main__":
    print("Running denoise.py (Full Denoising Pipeline).")
    print(f"Loading baseline data from: {REVIEWS_INPUT_PATH_BASELINE} and {META_INPUT_PATH_BASELINE}")
    try:
        # Load baseline data (noisy data) to start the pipeline
        reviews_df_baseline = pd.read_csv(REVIEWS_INPUT_PATH_BASELINE)
        meta_df_baseline = pd.read_csv(META_INPUT_PATH_BASELINE)
        print("Baseline data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading baseline data: {e}. Make sure {REVIEWS_INPUT_PATH_BASELINE} and {META_INPUT_PATH_BASELINE} exist.")
        exit()

    # Run the full denoising pipeline on copies of the baseline data
    # The pipeline functions modify/return copies, so passing copies of baseline is good practice
    reviews_denoised, meta_denoised = denoise_data_pipeline(reviews_df_baseline.copy(), meta_df_baseline.copy())

    print(f"\nSaving fully denoised data to: {REVIEWS_OUTPUT_PATH_DENOISED} and {META_OUTPUT_PATH_DENOISED}")
    reviews_denoised.to_csv(REVIEWS_OUTPUT_PATH_DENOISED, index=False)
    meta_denoised.to_csv(META_OUTPUT_PATH_DENOISED, index=False)
    print("Fully denoised files saved.")
