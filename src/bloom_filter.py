from pybloom_live import BloomFilter
import pandas as pd

bloom_filter = BloomFilter(capacity=1000000, error_rate=0.1)  # Corrected the argument names
review_df = pd.read_csv('../data/processed/processed_reviews_baseline.csv')
def apply_bloom_filter(text):
    if text not in bloom_filter:
        bloom_filter.add(text)
        return text
    else:
        return None  
review_df['filtered_text'] = review_df['processed_text'].apply(apply_bloom_filter)
review_df = review_df.dropna(subset=['filtered_text'])
review_df.to_csv('../data/processed/filtered_reviews_bloom.csv', index=False)

print("Filtered reviews using Bloom filter. Duplicates removed and saved.")
