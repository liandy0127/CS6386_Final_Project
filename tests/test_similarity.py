from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(text1, text2):
    vec1 = vectorizer.transform([text1]).toarray()
    vec2 = vectorizer.transform([text2]).toarray()
    return cosine_similarity(vec1, vec2)[0][0]
original_similarity = calculate_similarity(raw_data['text'][0], raw_data['text'][1])
cleaned_similarity = calculate_similarity(cleaned_data['processed_text'][0], cleaned_data['processed_text'][1])

print(f'Original Similarity: {original_similarity}')
print(f'Cleaned Similarity: {cleaned_similarity}')
