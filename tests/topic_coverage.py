import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
cleaned_data = pd.read_csv('../data/processed/processed_reviews_baseline.csv')  # Example path
cleaned_data['processed_text'] = cleaned_data['processed_text'].fillna('')
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(cleaned_data['processed_text'])
print("TF-IDF transformation complete!")
