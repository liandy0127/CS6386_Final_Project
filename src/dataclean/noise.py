# noise.py
import pandas as pd
import numpy as np
import random
import string
from tqdm.auto import tqdm # Import tqdm for progress bars

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Noise Injection Functions ---

def introduce_missing(df, cols, missing_frac=0.05):
    """
    Randomly set a fraction of values in specified columns to NaN.
    """
    df_noisy = df.copy()
    print(f"  Introducing missing values in columns: {cols} (fraction={missing_frac:.2f})")
    for col in cols:
        if col in df_noisy.columns: # Check if column exists
            mask = np.random.rand(len(df_noisy)) < missing_frac
            df_noisy.loc[mask, col] = np.nan
        else:
            print(f"    Warning: Column '{col}' not found for missing value injection.")
    return df_noisy


def introduce_misspellings(df, col, typo_frac=0.1):
    """
    Randomly introduce a single-character typo in a fraction of text values.
    Includes tqdm for progress tracking.
    """
    df_noisy = df.copy()
    if col not in df_noisy.columns: # Check if column exists
         print(f"  Warning: Column '{col}' not found for misspelling injection. Skipping.")
         return df_noisy

    print(f"  Introducing misspellings in column: '{col}' (fraction={typo_frac:.2f})")

    # Ensure the column is treated as string type to handle non-text data gracefully
    df_noisy[col] = df_noisy[col].astype(str)

    # Wrap the iteration with tqdm
    for idx, val in tqdm(df_noisy[col].items(), total=len(df_noisy), desc=f"  Adding typos to '{col}'"):
        if pd.isnull(val) or val == 'nan' or random.random() >= typo_frac: # Also check for 'nan' string
            continue
        s = list(str(val))
        if not s: # Skip empty strings
            continue
        i = random.randrange(len(s))
        # Using ascii_letters for simplicity; you might want a broader set
        # or more sophisticated typo injection here.
        s[i] = random.choice(string.ascii_letters)
        df_noisy.at[idx, col] = ''.join(s)
    return df_noisy


def inject_outliers(df, col, outlier_frac=0.02, factor=5):
    """
    Scale a small fraction of numeric values by a factor to create outliers.
    """
    df_noisy = df.copy()
    if col not in df_noisy.columns: # Check if column exists
         print(f"  Warning: Column '{col}' not found for outlier injection. Skipping.")
         return df_noisy

    print(f"  Injecting outliers in column: '{col}' (fraction={outlier_frac:.2f}, factor={factor})")

    # Ensure the column is numeric before multiplying
    # Coerce errors will turn non-numeric values into NaN
    df_noisy[col] = pd.to_numeric(df_noisy[col], errors='coerce')

    # Only select non-NaN values for potential outlier injection
    non_nan_indices = df_noisy[col].dropna().index.tolist()

    if not non_nan_indices:
        print(f"  Warning: No non-missing numeric values found in '{col}'. Skipping outlier injection.")
        return df_noisy

    n_non_nan = len(non_nan_indices)
    k = int(outlier_frac * n_non_nan)
    if k == 0:
         print(f"  Outlier fraction {outlier_frac:.2f} too small for {n_non_nan} non-missing values in '{col}', no outliers injected.")
         return df_noisy

    # Select k random indices from the non-NaN indices
    outlier_indices = np.random.choice(non_nan_indices, size=k, replace=False)

    # Apply the outlier factor only to the selected indices
    df_noisy.loc[outlier_indices, col] = df_noisy.loc[outlier_indices, col] * factor

    return df_noisy


def shuffle_values(df, col, shuffle_frac=0.05):
    """
    Randomly swap values in a column to disrupt consistency.
    Includes checks for column existence.
    """
    df_noisy = df.copy()
    if col not in df_noisy.columns: # Check if column exists BEFORE proceeding
         print(f"  Warning: Column '{col}' not found for value shuffling. Skipping.")
         return df_noisy

    print(f"  Shuffling values in column: '{col}' (fraction={shuffle_frac:.2f})")
    n = len(df_noisy)
    k = int(shuffle_frac * n)
    if k == 0:
        print(f"    Shuffle fraction {shuffle_frac:.2f} too small for {n} rows, no values shuffled.")
        return df_noisy

    # Ensure we don't try to swap more pairs than half the dataframe length
    num_swaps = min(k, n // 2)

    # Get all indices from the DataFrame
    all_indices = df_noisy.index.tolist()

    if len(all_indices) < num_swaps * 2:
         print(f"    Warning: Not enough unique indices ({len(all_indices)}) for {num_swaps} swaps. Shuffling fewer values.")
         num_swaps = len(all_indices) // 2
         if num_swaps == 0:
              print("    Cannot perform any swaps.")
              return df_noisy

    # Correctly sample pairs of unique indices from the DataFrame's index
    # Ensure replace=False for unique indices
    try:
        swap_indices_flat = random.sample(all_indices, num_swaps * 2)
        idxs = [(swap_indices_flat[i], swap_indices_flat[i+1]) for i in range(0, num_swaps * 2, 2)]
    except ValueError as e:
         # This can happen if num_swaps * 2 > len(all_indices) even after the check,
         # due to floating point issues or edge cases.
         print(f"    Error sampling indices for shuffling: {e}. Skipping shuffle for '{col}'.")
         return df_noisy


    # Use a list of swaps for potentially better performance than repeated .at
    values_to_swap = []
    for i, j in idxs:
        # Access values using .loc for potentially better performance than .at with default index
        # Also explicitly check column existence again just in case (though checked above)
        if col not in df_noisy.columns:
             print(f"    Error: Column '{col}' disappeared during shuffling. Skipping.")
             return df_noisy
        values_to_swap.append((df_noisy.loc[i, col], df_noisy.loc[j, col]))


    # Apply the swaps
    for swap_idx, (i, j) in enumerate(idxs):
         if col not in df_noisy.columns:
             print(f"    Error: Column '{col}' disappeared during shuffling. Skipping.")
             return df_noisy
         df_noisy.loc[i, col] = values_to_swap[swap_idx][1] # Value from j
         df_noisy.loc[j, col] = values_to_swap[swap_idx][0] # Value from i

    return df_noisy

# Helper function to clean column names
def clean_column_names(df):
    """Strips leading/trailing whitespace from column names."""
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip()
    if original_columns != df.columns.tolist():
        print(f"  Cleaned column names: {original_columns} -> {df.columns.tolist()}")
    return df


# --- Load original data ---
print("Loading original data...")
try:
    reviews = pd.read_csv('../../data/processed/raw_meta_baseline.csv')
    meta    = pd.read_csv('../../data/processed/raw_reviews_baseline.csv')
    print("Original data loaded.")

    # Clean column names immediately after loading
    reviews = clean_column_names(reviews)
    meta = clean_column_names(meta)
    print("Column names cleaned.")

except FileNotFoundError as e:
    print(f"Error loading original data: {e}. Make sure 'processed_reviews.csv' and 'processed_meta.csv' exist in ../../data/processed/")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading or column cleaning: {e}")
    exit()


# --- Inject noise into reviews ---
print("\nInjecting noise into reviews data...")
# 1. Random missing in text and rating
reviews_noisy = introduce_missing(reviews.copy(), ['title', 'text', 'rating'], missing_frac=0.05) # Work on a copy
# 2. Misspellings in title
reviews_noisy = introduce_misspellings(reviews_noisy.copy(), 'title', typo_frac=0.1) # Work on a copy
# 3. Outliers in numeric rating
reviews_noisy = inject_outliers(reviews_noisy.copy(), 'rating', outlier_frac=0.02, factor=5) # Work on a copy
# 4. Shuffle user_id associations (swap some user-item pairs)
reviews_noisy = shuffle_values(reviews_noisy.copy(), 'user_id', shuffle_frac=0.03) # Work on a copy
print("Noise injection complete for reviews.")


# --- Inject noise into metadata ---
print("\nInjecting noise into metadata...")
# 1. Random missing in description and price
meta_noisy = introduce_missing(meta.copy(), ['description', 'price'], missing_frac=0.05) # Work on a copy
# 2. Misspellings in main_category and title
meta_noisy = introduce_misspellings(meta_noisy.copy(), 'main_category', typo_frac=0.1) # Work on a copy
meta_noisy = introduce_misspellings(meta_noisy.copy(), 'title', typo_frac=0.05) # Work on a copy
# 3. Swap bought_together links randomly
meta_noisy = shuffle_values(meta_noisy.copy(), 'bought_together', shuffle_frac=0.05) # Work on a copy
print("Noise injection complete for metadata.")

# --- Save noisy datasets ---
print("\nSaving noisy datasets...")
try:
    reviews_noisy.to_csv('../../data/processed/reviews_noisy.csv', index=False)
    meta_noisy.to_csv('../../data/processed/meta_noisy.csv', index=False)

    print("\nNoisy datasets saved:")
    print(f" - processed_reviews_noisy.csv: {len(reviews_noisy)} rows")
    print(f" - processed_meta_noisy.csv:    {len(meta_noisy)} rows")
except Exception as e:
    print(f"An error occurred while saving the noisy data: {e}")