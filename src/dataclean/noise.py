import pandas as pd
import numpy as np
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Paths to the datasets
metadata_path = "../../data/processed/raw_meta_baseline.csv"
reviews_path = "../../data/processed/raw_reviews_baseline.csv"

# Load datasets
metadata = pd.read_csv(metadata_path)
reviews = pd.read_csv(reviews_path)

# Function to introduce typos in text
def introduce_typos(text, prob=0.1):
    if not isinstance(text, str):
        return text  # If not string, return as is
    words = text.split()
    for i in range(len(words)):
        if random.random() < prob:
            word = words[i]
            if len(word) > 1:
                # Swap two random letters
                idx1, idx2 = random.sample(range(len(word)), 2)
                word_list = list(word)
                word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
                words[i] = ''.join(word_list)
            elif len(word) == 1:
                # Add a random letter
                words[i] += random.choice('abcdefghijklmnopqrstuvwxyz')
    return ' '.join(words)

# --- Introduce noise to metadata ---

num_rows = len(metadata)

# 1. Set 10% of 'title' to empty string
missing_title_indices = random.sample(range(num_rows), int(0.1 * num_rows))
metadata.loc[missing_title_indices, 'title'] = ''

# 2. Set 10% of 'description' to NaN
missing_desc_indices = random.sample(range(num_rows), int(0.1 * num_rows))
metadata.loc[missing_desc_indices, 'description'] = np.nan

# 4. Set 5% of 'average_rating' to NaN
missing_rating_indices = random.sample(range(num_rows), int(0.05 * num_rows))
metadata.loc[missing_rating_indices, 'average_rating'] = np.nan

# 5. Introduce typos in 5% of 'main_category'
category_typo_indices = random.sample(range(num_rows), int(0.05 * num_rows))
for idx in category_typo_indices:
    metadata.loc[idx, 'main_category'] = introduce_typos(metadata.loc[idx, 'main_category'], prob=0.5)

# 6. Introduce invalid prices in 5% of rows
invalid_price_indices = random.sample(range(num_rows), int(0.05 * num_rows))
for idx in invalid_price_indices:
    if random.random() < 0.5:
        metadata.loc[idx, 'price'] = -1
    else:
        metadata.loc[idx, 'price'] = 1000000  # Unrealistically high price

# 7. Add duplicates (5% of rows)
duplicate_indices = random.sample(range(num_rows), int(0.05 * num_rows))
duplicates = metadata.iloc[duplicate_indices]
metadata = pd.concat([metadata, duplicates], ignore_index=True)

# --- Introduce noise to reviews ---

num_reviews = len(reviews)

# 1. Set 10% of 'rating' to NaN
missing_rating_indices = random.sample(range(num_reviews), int(0.1 * num_reviews))
reviews.loc[missing_rating_indices, 'rating'] = np.nan

# 2. Set 10% of 'text' to empty string
missing_text_indices = random.sample(range(num_reviews), int(0.1 * num_reviews))
reviews.loc[missing_text_indices, 'text'] = ''

# 3. Introduce typos in 10% of 'title' and 'text'
typo_review_indices = random.sample(range(num_reviews), int(0.1 * num_reviews))
for idx in typo_review_indices:
    reviews.loc[idx, 'title'] = introduce_typos(reviews.loc[idx, 'title'])
    reviews.loc[idx, 'text'] = introduce_typos(reviews.loc[idx, 'text'])

# 4. Introduce invalid timestamps in 5% of rows
invalid_timestamp_indices = random.sample(range(num_reviews), int(0.05 * num_reviews))
for idx in invalid_timestamp_indices:
    if random.random() < 0.5:
        reviews.loc[idx, 'timestamp'] = -1
    else:
        reviews.loc[idx, 'timestamp'] = int(time.time() * 1000) + 10000000000  # Future timestamp

# 5. Introduce invalid ratings in 5% of rows (e.g., 0 or 6 for a 1-5 scale)
invalid_rating_indices = random.sample(range(num_reviews), int(0.05 * num_reviews))
for idx in invalid_rating_indices:
    reviews.loc[idx, 'rating'] = random.choice([0, 6])

# 6. Introduce invalid 'helpful_vote' values (e.g., negative)
invalid_helpful_indices = random.sample(range(num_reviews), int(0.05 * num_reviews))
for idx in invalid_helpful_indices:
    reviews.loc[idx, 'helpful_vote'] = -1

# 7. Add duplicates (5% of rows)
duplicate_review_indices = random.sample(range(num_reviews), int(0.05 * num_reviews))
duplicates_reviews = reviews.iloc[duplicate_review_indices]
reviews = pd.concat([reviews, duplicates_reviews], ignore_index=True)

# --- Save noisy datasets ---
metadata.to_csv("../../data/processed/meta_noisy.csv", index=False)
reviews.to_csv("../../data/processed/reviews_noisy.csv", index=False)