# src/dataclean/handle_missing.py
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Use auto for compatibility

# Define default input and output paths - CORRECTED
# When run standalone, this script should take text_cleaned data as input
DEFAULT_REVIEWS_INPUT_PATH = "../../data/processed/reviews_text_cleaned.csv" # <-- CORRECT
DEFAULT_META_INPUT_PATH = "../../data/processed/meta_text_cleaned.csv"     # <-- CORRECT
REVIEWS_OUTPUT_PATH = "../../data/processed/reviews_missing_handled.csv"
META_OUTPUT_PATH = "../../data/processed/meta_missing_handled.csv"

TEXT_PLACEHOLDER = "[Missing Value]"

# --- Missing Value Handling Function ---

def handle_missing_values(df, numeric_cols, text_cols, placeholder=TEXT_PLACEHOLDER):
    """
    Handles missing values in specified columns:
    - Imputes numeric columns with their median.
    - Imputes text columns with a placeholder string.
    Modifies dataframe in place and returns it.
    """
    df_handled = df.copy() # Work on a copy
    total_missing_numeric = 0
    total_missing_text = 0

    # print("  Handling missing numeric values...") # Suppress in function
    for col in tqdm(numeric_cols, desc=" Imputing numeric columns", leave=False): # Use leave=False for cleaner output in pipeline
        if col in df_handled.columns:
            # Ensure column is numeric before calculating median
            original_na_count = df_handled[col].isnull().sum()
            df_handled[col] = pd.to_numeric(df_handled[col], errors='coerce')
            current_na_count = df_handled[col].isnull().sum()

            if current_na_count > 0:
                median_val = df_handled[col].median()
                df_handled[col].fillna(median_val, inplace=True)
                # print(f"    Imputed {current_na_count} missing values in '{col}' with median ({median_val}).")
                total_missing_numeric += current_na_count
        # else: print(f"    Warning: Numeric column '{col}' not found.") # Suppress in combined view


    # print("  Handling missing text values...") # Suppress in function
    for col in tqdm(text_cols, desc=" Imputing text columns", leave=False): # Use leave=False
        if col in df_handled.columns:
             original_na_count = df_handled[col].isnull().sum()
             # Ensure column is of object/string type and handle 'nan' strings
             df_handled[col] = df_handled[col].astype(str).replace('nan', np.nan)
             current_na_count = df_handled[col].isnull().sum()

             if current_na_count > 0:
                df_handled[col].fillna(placeholder, inplace=True)
                # print(f"    Imputed {current_na_count} missing values in '{col}' with placeholder: '{placeholder}'.")
                total_missing_text += current_na_count
        # else: print(f"    Warning: Text column '{col}' not found.") # Suppress in combined view

    if total_missing_numeric > 0 or total_missing_text > 0:
        print(f"  Total numeric missing values imputed: {total_missing_numeric}")
        print(f"  Total text missing values imputed: {total_missing_text}")

    return df_handled

# --- Main Execution for standalone use ---
if __name__ == "__main__":
    print("Running handle_missing.py as a standalone script.")
    # Attempt to load output from previous step (text cleaning) as default input
    reviews_input_path = DEFAULT_REVIEWS_INPUT_PATH
    meta_input_path = DEFAULT_META_INPUT_PATH

    print(f"Attempting to load data from: {reviews_input_path} and {meta_input_path}")
    try:
        reviews_df = pd.read_csv(reviews_input_path)
        meta_df = pd.read_csv(meta_input_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Could not find files at {reviews_input_path} and {meta_input_path}.")
        # Fallback logic: try previous cleaning stages in order, then noisy
        print("Attempting to load previous cleaning stage data as fallback...")
        fallback_paths = [
            ("../../data/processed/reviews_text_cleaned.csv", "../../data/processed/meta_text_cleaned.csv"),
            ("../../data/processed/reviews_noisy.csv", "../../data/processed/meta_noisy.csv")
        ]
        loaded = False
        for rev_path, meta_path in fallback_paths:
            try:
                reviews_df = pd.read_csv(rev_path)
                meta_df = pd.read_csv(meta_path)
                print(f"Loaded fallback data from {rev_path} and {meta_path}.")
                loaded = True
                break
            except FileNotFoundError:
                continue
        if not loaded:
             print("Error: Could not load data from any specified path.")
             exit()


    print("\nHandling missing values in reviews data...")
    # Define numeric and text columns for reviews to handle missing values
    # Include common numeric cols and all object/string cols (includes original and cleaned text)
    reviews_numeric_cols = ['rating', 'helpful_vote']
    reviews_text_cols = [col for col in reviews_df.columns if reviews_df[col].dtype == 'object' or reviews_df[col].dtype == 'string']

    reviews_df_handled = handle_missing_values(reviews_df.copy(), reviews_numeric_cols, reviews_text_cols, TEXT_PLACEHOLDER)


    print("\nHandling missing values in metadata...")
    # Define numeric and text columns for metadata to handle missing values
    meta_numeric_cols = ['average_rating', 'rating_number', 'price']
    meta_text_cols = [col for col in meta_df.columns if meta_df[col].dtype == 'object' or meta_df[col].dtype == 'string']

    meta_df_handled = handle_missing_values(meta_df.copy(), meta_numeric_cols, meta_text_cols, TEXT_PLACEHOLDER)


    print(f"\nSaving data with missing values handled to: {REVIEWS_OUTPUT_PATH} and {META_OUTPUT_PATH}")
    reviews_df_handled.to_csv(REVIEWS_OUTPUT_PATH, index=False)
    meta_df_handled.to_csv(META_OUTPUT_PATH, index=False)
    print("Files with missing values handled saved.")
