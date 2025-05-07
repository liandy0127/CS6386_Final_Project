import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from textblob import TextBlob

# Load the processed data
def load_data():
    baseline = pd.read_csv('../data/processed/processed_reviews_baseline.csv')
    filtered = pd.read_csv('../data/processed/filtered_reviews_bloom.csv')
    return baseline, filtered

# Graph 1: Review Count Comparison
def plot_review_counts(baseline, filtered):
    baseline_count = len(baseline)
    filtered_count = len(filtered)
    methods = ['Baseline', 'Bloom Filter']
    counts = [baseline_count, filtered_count]
    
    plt.figure(figsize=(6, 4))
    plt.bar(methods, counts, color=['blue', 'green'])
    plt.title('Review Count Comparison')
    plt.ylabel('Number of Reviews')
    plt.savefig('review_count_comparison.png')
    plt.show()

# Graph 2: Text Length Distribution (in words)
def plot_text_length_distribution(baseline, filtered):
    baseline['text_length'] = baseline['processed_text'].fillna('').apply(lambda x: len(x.split()))
    filtered['text_length'] = filtered['filtered_text'].fillna('').apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 5))
    plt.hist(baseline['text_length'], bins=30, alpha=0.5, label='Baseline', color='blue')
    plt.hist(filtered['text_length'], bins=30, alpha=0.5, label='Bloom Filter', color='green')
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('text_length_distribution.png')
    plt.show()

# Graph 3: Top 10 Words Frequency Comparison
def plot_top_words(baseline, filtered):
    # Combine all texts into one string for each method
    baseline_text = ' '.join(baseline['processed_text'].fillna(''))
    filtered_text = ' '.join(filtered['filtered_text'].fillna(''))
    
    baseline_counter = Counter(baseline_text.split())
    filtered_counter = Counter(filtered_text.split())
    
    baseline_top = baseline_counter.most_common(10)
    filtered_top = filtered_counter.most_common(10)
    
    # Unzip the tuples for plotting
    baseline_words, baseline_counts = zip(*baseline_top)
    filtered_words, filtered_counts = zip(*filtered_top)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(baseline_words, baseline_counts, color='blue')
    axes[0].set_title('Top 10 Words - Baseline')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(filtered_words, filtered_counts, color='green')
    axes[1].set_title('Top 10 Words - Bloom Filter')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('top_words_comparison.png')
    plt.show()

# Graph 4: Word Cloud Comparison
def plot_word_clouds(baseline, filtered):
    baseline_text = ' '.join(baseline['processed_text'].fillna(''))
    filtered_text = ' '.join(filtered['filtered_text'].fillna(''))
    
    baseline_wc = WordCloud(width=800, height=400, background_color='white').generate(baseline_text)
    filtered_wc = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(baseline_wc, interpolation='bilinear')
    axes[0].set_title('Word Cloud - Baseline')
    axes[0].axis('off')
    
    axes[1].imshow(filtered_wc, interpolation='bilinear')
    axes[1].set_title('Word Cloud - Bloom Filter')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('word_clouds_comparison.png')
    plt.show()

# Graph 5: Rating Distribution Comparison
def plot_rating_distribution(baseline, filtered):
    plt.figure(figsize=(10, 5))
    plt.hist(baseline['rating'].dropna(), bins=20, alpha=0.5, label='Baseline', color='blue')
    plt.hist(filtered['rating'].dropna(), bins=20, alpha=0.5, label='Bloom Filter', color='green')
    plt.title('Rating Distribution Comparison')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('rating_distribution_comparison.png')
    plt.show()

def plot_sentiment_distribution(baseline, filtered):
    # Drop NaNs first
    baseline_nonan = baseline['processed_text'].dropna()
    filtered_nonan = filtered['filtered_text'].dropna()
    
    # Sample based on the actual non-NaN counts
    baseline_sample = baseline_nonan.sample(n=min(1000, len(baseline_nonan)), random_state=42)
    filtered_sample = filtered_nonan.sample(n=min(1000, len(filtered_nonan)), random_state=42)
    
    # Compute sentiment polarity using TextBlob
    baseline_sentiment = baseline_sample.apply(lambda x: TextBlob(x).sentiment.polarity)
    filtered_sentiment = filtered_sample.apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Plotting the sentiment distribution
    plt.figure(figsize=(10, 5))
    plt.hist(baseline_sentiment, bins=20, alpha=0.5, label='Baseline', color='blue')
    plt.hist(filtered_sentiment, bins=20, alpha=0.5, label='Bloom Filter', color='green')
    plt.title('Sentiment Distribution Comparison')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('sentiment_distribution_comparison.png')
    plt.show()


# Graph 7: PCA on TF-IDF Features
def plot_pca_tf_idf(baseline, filtered):
    sample_size = min(500, len(baseline), len(filtered))
    baseline_sample = baseline['processed_text'].dropna().sample(n=sample_size, random_state=42)
    filtered_sample = filtered['filtered_text'].dropna().sample(n=sample_size, random_state=42)
    
    combined_text = pd.concat([baseline_sample, filtered_sample])
    labels = ['Baseline'] * sample_size + ['Bloom Filter'] * sample_size
    
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result[:sample_size, 0], pca_result[:sample_size, 1], alpha=0.5, label='Baseline', color='blue')
    plt.scatter(pca_result[sample_size:, 0], pca_result[sample_size:, 1], alpha=0.5, label='Bloom Filter', color='green')
    plt.title('PCA of TF-IDF Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig('pca_tf_idf_comparison.png')
    plt.show()

if __name__ == '__main__':
    baseline_df, filtered_df = load_data()
    
    # Generate Graph 1: Review Count Comparison
    #plot_review_counts(baseline_df, filtered_df)
    
    # Generate Graph 2: Text Length Distribution
    #plot_text_length_distribution(baseline_df, filtered_df)
    
    # Generate Graph 3: Top 10 Words Frequency Comparison
    #plot_top_words(baseline_df, filtered_df)
    
    # Generate Graph 4: Word Cloud Comparison
    #plot_word_clouds(baseline_df, filtered_df)
    
    # Generate Graph 5: Rating Distribution Comparison
    #plot_rating_distribution(baseline_df, filtered_df)
    
    # Generate Graph 6: Sentiment Distribution Comparison
    #plot_sentiment_distribution(baseline_df, filtered_df)
    
    # Generate Graph 7: PCA on TF-IDF Features
    plot_pca_tf_idf(baseline_df, filtered_df)
