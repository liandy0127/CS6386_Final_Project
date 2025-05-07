import pandas as pd
import re
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import json

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # Added this line
nltk.download('wordnet')

lemmatizer = nltk.WordNetLemmatizer()

def read_jsonl_in_chunks(file_path, max_lines=10000):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {file_path}")):
            if i >= max_lines:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid line {i+1} in {file_path}")
    return pd.DataFrame(data)

review_df = read_jsonl_in_chunks('data/raw/Video_Games_subset.jsonl')
meta_df = read_jsonl_in_chunks('data/raw/meta_Video_Games_subset.jsonl')

print("Review columns:", review_df.columns.tolist())
print("Meta columns:", meta_df.columns.tolist())

if 'text' in review_df.columns and 'title' in review_df.columns:
    review_df['combined_text'] = review_df['title'] + ' ' + review_df['text'].fillna('')
elif 'description' in review_df.columns and 'title' in review_df.columns:
    review_df['combined_text'] = review_df['title'] + ' ' + review_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
else:
    raise ValueError("review_df lacks required columns (title and either text or description)")

meta_df['combined_text'] = meta_df['main_category'] + ' ' + meta_df['categories'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

tqdm.pandas(desc="Processing reviews")
review_df['processed_text'] = review_df['combined_text'].progress_apply(preprocess_text)
meta_df['processed_text'] = meta_df['combined_text'].progress_apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=2500)

X_review = vectorizer.fit_transform(review_df['processed_text']).toarray()
X_meta = vectorizer.fit_transform(meta_df['processed_text']).toarray()

review_df.to_csv('../data/processed/processed_reviews_baseline.csv', index=False)
meta_df.to_csv('../data/processed/processed_meta_baseline.csv', index=False)

print("Processing complete. Processed files saved.")