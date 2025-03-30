from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_json('../data/raw/Video_Games_subset.jsonl', lines=True)
cleaned_data = pd.read_csv('../data/processed/processed_reviews_baseline.csv')  
filtered_data = pd.read_csv('../data/processed/filtered_reviews_bloom.csv')  

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(cleaned_data['processed_text']).toarray()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_train_vec)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', label='Cleaned Data')
plt.title('PCA of Cleaned Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
