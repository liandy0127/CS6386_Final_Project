import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

review_df = pd.read_csv('../data/processed/processed_reviews_baseline.csv')
filtered_df = pd.read_csv('../data/processed/filtered_reviews_bloom.csv')
baseline_count = review_df.shape[0]
bloom_filtered_count = filtered_df.shape[0]

plt.figure(figsize=(6, 4))
plt.bar(['Baseline Method', 'Bloom Filtered'], [baseline_count, bloom_filtered_count], color=['blue', 'green'])
plt.title('Review Count Before and After Bloom Filter')
plt.ylabel('Number of Reviews')
plt.show()

def get_word_frequencies(text_data):
    all_words = ' '.join(text_data).split()
    word_freq = Counter(all_words)
    return word_freq

baseline_word_freq = get_word_frequencies(review_df['processed_text'])
filtered_word_freq = get_word_frequencies(filtered_df['filtered_text'])
top_baseline_words = dict(baseline_word_freq.most_common(10))
top_filtered_words = dict(filtered_word_freq.most_common(10))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].bar(top_baseline_words.keys(), top_baseline_words.values(), color='blue')
axes[0].set_title('Top 10 Words - Baseline Method')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(top_filtered_words.keys(), top_filtered_words.values(), color='green')
axes[1].set_title('Top 10 Words - After Bloom Filter')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
wordcloud_baseline = WordCloud(width=800, height=400).generate(' '.join(review_df['processed_text']))
wordcloud_filtered = WordCloud(width=800, height=400).generate(' '.join(filtered_df['filtered_text']))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_baseline, interpolation='bilinear')
plt.title('Word Cloud - Baseline Method')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_filtered, interpolation='bilinear')
plt.title('Word Cloud - After Bloom Filter')
plt.axis('off')

plt.tight_layout()
plt.show()
