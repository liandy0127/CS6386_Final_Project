# src/dataclean/clean_text.py
import pandas as pd
import re
import nltk
from tqdm.auto import tqdm # Use auto for compatibility
import string
import json

# Download necessary NLTK data (run this script once to ensure downloads)
# Corrected exception handling for downloads
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Removed nltk.download('punkt_tab') as it seems non-standard

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
     print("Downloading NLTK wordnet...")
     nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4') # Often needed for wordnet
except LookupError:
     print("Downloading NLTK omw-1.4...")
     nltk.download('omw-1.4')


lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Define default input and output paths - CORRECTED
DEFAULT_REVIEWS_INPUT_PATH = "data/processed/reviews_noisy.csv" # <-- Correct input
DEFAULT_META_INPUT_PATH = "data/processed/meta_noisy.csv"     # <-- Correct input
REVIEWS_OUTPUT_PATH = "data/processed/reviews_text_cleaned.csv"
META_OUTPUT_PATH = "data/processed/meta_text_cleaned.csv"
CLEAN_TEXT_STATS_PATH = "data/processed/clean_text_stats.json"

# --- Text Cleaning Function ---

def preprocess_text_cleaning(text):
    """
    Cleans and preprocesses text: lower, remove punctuation, remove numbers,
    extra whitespace, tokenize, remove stopwords, lemmatize.
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers (optional, adjust if numbers are meaningful)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Default is 'n' for noun

    return ' '.join(tokens)

def safe_preprocess_meta_text(item):
    """
    Applies preprocess_text_cleaning, safely handling list formats in meta columns.
    """
    if isinstance(item, list):
        # Join list items into a single string before processing
        item = ' '.join(map(str, item))
    elif not isinstance(item, str):
         # Handle other non-string types gracefully
         item = str(item)
    return preprocess_text_cleaning(item)


def apply_text_cleaning(reviews_df, meta_df):
    """
    Applies text cleaning to specified columns in reviews and metadata.
    Creates new '_cleaned' columns and a 'combined_cleaned_text' column.
    """
    print("Applying text cleaning steps...")
    tqdm.pandas() # Enable tqdm for pandas apply

    # Apply cleaning to relevant review text columns
    # Ensure these columns exist before attempting to clean
    review_text_cols = ['title', 'text']
    for col in review_text_cols:
        if col in reviews_df.columns:
             print(f" Cleaning '{col}' in reviews...")
             reviews_df[f'{col}_cleaned'] = reviews_df[col].progress_apply(preprocess_text_cleaning)
        else:
            print(f" Warning: Review column '{col}' not found in reviews_df. Skipping.")

    # Create/Update combined cleaned text column for reviews
    cleaned_review_cols_exist = [f'{col}_cleaned' for col in review_text_cols if f'{col}_cleaned' in reviews_df.columns]
    if cleaned_review_cols_exist:
        print(" Creating/Updating 'combined_cleaned_text' in reviews...")
        reviews_df['combined_cleaned_text'] = reviews_df[cleaned_review_cols_exist].fillna('').agg(' '.join, axis=1)
        reviews_df['combined_cleaned_text'] = reviews_df['combined_cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    elif 'combined_text' in reviews_df.columns:
         # Fallback: if cleaned source cols don't exist, clean the original combined_text if it exists
         print(" Cleaning existing 'combined_text' in reviews...")
         reviews_df['combined_cleaned_text'] = reviews_df['combined_text'].progress_apply(preprocess_text_cleaning)
         reviews_df['combined_cleaned_text'] = reviews_df['combined_cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
         print(" Warning: Could not create 'combined_cleaned_text' for reviews (no source or combined_text found).")


    # Apply cleaning to relevant meta text columns
    # Based on sample: main_category, title, features, description, categories, details, bought_together
    # bought_together is a list of ASINs, cleaning it directly might not be useful for text analysis
    # Let's focus on descriptive text fields.
    meta_text_cols = ['main_category', 'title', 'features', 'description', 'categories', 'details', 'store']
    for col in meta_text_cols:
        if col in meta_df.columns:
            print(f" Cleaning '{col}' in metadata...")
            # Use safe_preprocess_meta_text which handles list formats
            meta_df[f'{col}_cleaned'] = meta_df[col].progress_apply(safe_preprocess_meta_text)
        else:
             print(f" Warning: Meta column '{col}' not found in meta_df. Skipping.")

    # Create combined cleaned text column for meta
    cleaned_meta_cols_exist = [f'{col}_cleaned' for col in meta_text_cols if f'{col}_cleaned' in meta_df.columns]
    if cleaned_meta_cols_exist:
        print(" Creating 'combined_cleaned_text' in metadata...")
        meta_df['combined_cleaned_text'] = meta_df[cleaned_meta_cols_exist].fillna('').agg(' '.join, axis=1)
        meta_df['combined_cleaned_text'] = meta_df['combined_cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    elif 'combined_text' in meta_df.columns:
        # Fallback: if cleaned source cols don't exist, clean the original combined_text if it exists
         print(" Cleaning existing 'combined_text' in metadata...")
         meta_df['combined_cleaned_text'] = meta_df['combined_text'].progress_apply(safe_preprocess_meta_text)
         meta_df['combined_cleaned_text'] = meta_df['combined_cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
        print(" Warning: Could not create 'combined_cleaned_text' for metadata (no source or combined_text found).")


    print("Text cleaning steps complete.")
    return reviews_df, meta_df


# --- Main Execution for standalone use ---
if __name__ == "__main__":
    print("Running clean_text.py as a standalone script.")
    print(f"Loading data from: {DEFAULT_REVIEWS_INPUT_PATH} and {DEFAULT_META_INPUT_PATH}")
    try:
        # Load noisy data as input for the first cleaning step
        reviews_df = pd.read_csv(DEFAULT_REVIEWS_INPUT_PATH)
        meta_df = pd.read_csv(DEFAULT_META_INPUT_PATH)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading input data: {e}. Make sure {DEFAULT_REVIEWS_INPUT_PATH} and {DEFAULT_META_INPUT_PATH} exist.")
        exit()

    # Pass copies to avoid modifying original in place if this script is imported elsewhere
    reviews_cleaned, meta_cleaned = apply_text_cleaning(reviews_df.copy(), meta_df.copy())

    print(f"\nSaving cleaned data to: {REVIEWS_OUTPUT_PATH} and {META_OUTPUT_PATH}")
    reviews_cleaned.to_csv(REVIEWS_OUTPUT_PATH, index=False)
    meta_cleaned.to_csv(META_OUTPUT_PATH, index=False)
    print("Text cleaned files saved.")
        # Calculate stats for cleaned text entries
    review_entries_cleaned = reviews_cleaned['combined_cleaned_text'].apply(lambda x: bool(x.strip())).sum() if 'combined_cleaned_text' in reviews_cleaned.columns else 0
    meta_entries_cleaned = meta_cleaned['combined_cleaned_text'].apply(lambda x: bool(x.strip())).sum() if 'combined_cleaned_text' in meta_cleaned.columns else 0

    stats = {
        "review_entries_cleaned": int(review_entries_cleaned),
        "meta_entries_cleaned": int(meta_entries_cleaned)
    }

    # Write stats to JSON file
    with open(CLEAN_TEXT_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Clean text stats saved to: {CLEAN_TEXT_STATS_PATH}")
