from pybloom_live import BloomFilter
import pandas as pd

# Initialize Bloom filter with size and error rate
bloom_filter = BloomFilter(capacity=1000000, error_rate=0.1)  # Corrected the argument names

# Load preprocessed data
review_df = pd.read_csv('../data/processed/processed_reviews_baseline.csv')

# Bloom filter application to check if text is already encountered
def apply_bloom_filter(text):
    if text not in bloom_filter:
        bloom_filter.add(text)
        return text
    else:
        return None  # Return None if text is already encountered

# Apply the Bloom filter to the 'processed_text' column
review_df['filtered_text'] = review_df['processed_text'].apply(apply_bloom_filter)

# Drop rows where text is a duplicate
review_df = review_df.dropna(subset=['filtered_text'])

# Save the filtered data
review_df.to_csv('../data/processed/filtered_reviews_bloom.csv', index=False)

print("Filtered reviews using Bloom filter. Duplicates removed and saved.")
