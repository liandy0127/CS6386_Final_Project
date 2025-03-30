from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Correct the file reading to use `read_json` for .jsonl files
raw_data = pd.read_json('../data/raw/Video_Games_subset.jsonl', lines=True)
cleaned_data = pd.read_csv('../data/processed/processed_reviews_baseline.csv')  # Baseline data
filtered_data = pd.read_csv('../data/processed/filtered_reviews_bloom.csv')  # Filtered data

# Continue with your preprocessing and model as before

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(cleaned_data['processed_text'], cleaned_data['rating'], test_size=0.2, random_state=42)

# Vectorize text data (TF-IDF or other methods)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a simple model (e.g., Nearest Neighbors)
model = NearestNeighbors(n_neighbors=5)
model.fit(X_train_vec)

# Predictions and evaluation
predictions = model.kneighbors(X_test_vec, return_distance=False)
# Evaluate using a relevant metric (e.g., mean squared error for ratings)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
