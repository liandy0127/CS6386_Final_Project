import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
from sentence_transformers import util

# ─── Dependencies ───────────────────────────────────────────────────────────────
# pip install pandas numpy torch transformers sentence-transformers scikit-learn tqdm nltk

# Ensure NLTK data is available
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ─── 1. Data-Cleaning Function ──────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)  # Strip punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]
    return " ".join(tokens)

# ─── 2. Load & Merge ────────────────────────────────────────────────────────────
reviews = pd.read_csv("../data/processed/reviews_denoised_combined.csv")
meta = pd.read_csv("../data/processed/meta_denoised_combined.csv")

# Merge so every row has one text field
df = pd.merge(
    reviews[['user_id', 'parent_asin', 'rating', 'text']],
    meta[['parent_asin', 'description']],
    on='parent_asin',
    how='left'
)

# Create combined_text, prioritizing review text
df['combined_text'] = df['text'].fillna('').astype(str)
df['combined_text'] = df['combined_text'].where(df['combined_text'] != '', df['description'].fillna(''))

# Drop rows with empty combined_text
df = df[df['combined_text'] != ''].reset_index(drop=True)

# ─── 3. Encode & Split ─────────────────────────────────────────────────────────
user_enc = LabelEncoder().fit(df['user_id'])
item_enc = LabelEncoder().fit(df['parent_asin'])
df['user'] = user_enc.transform(df['user_id'])
df['item'] = item_enc.transform(df['parent_asin'])
df['label'] = (df['rating'] >= 4).astype(int)  # Implicit ground truth

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ─── 4. Device Setup ───────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ─── 5. Load BLaIR Model ────────────────────────────────────────────────────────
model_name = "hyp1231/blair-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def embed_texts(texts, clean=False):
    """Batch-encode a list of texts with optional cleaning."""
    if clean:
        texts = [clean_text(t) for t in texts]
    embeddings = []
    for i in tqdm(range(0, len(texts), 16), desc="Embedding batches"):
        batch = texts[i:i+16]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state[:, 0]
            out = out / out.norm(dim=1, keepdim=True)  # Normalize
        embeddings.append(out)
    return torch.cat(embeddings, dim=0)

# ─── 6. Prepare Item Embeddings ───────────────────────────────────────────────
print("Building item text list...")
item_texts_raw = []
for i in tqdm(range(len(item_enc.classes_)), desc="Item Texts"):
    item_text = df.loc[df['item'] == i, 'combined_text'].iloc[0] if not df.loc[df['item'] == i, 'combined_text'].empty else "No description available"
    item_texts_raw.append(item_text)

print("Computing raw embeddings...")
emb_raw = embed_texts(item_texts_raw, clean=False)
print("Computing cleaned embeddings...")
emb_clean = embed_texts(item_texts_raw, clean=True)

# ─── 7. Evaluate Function ──────────────────────────────────────────────────────
def evaluate(embeddings, desc):
    # Build user profiles
    profiles = {}
    for u, grp in tqdm(train_df.groupby('user'), desc=f"Building profiles ({desc})"):
        pos = grp.loc[grp['label'] == 1, 'item'].values
        profiles[u] = embeddings[pos].mean(dim=0) if len(pos) else torch.zeros(embeddings.size(1), device=device)

    # Metrics collection
    y_true, y_score, y_pred = [], [], []
    hit, users = 0, test_df['user'].nunique()

    for u, grp in tqdm(test_df.groupby('user'), total=users, desc=f"Evaluating {desc}"):
        true_set = set(grp.loc[grp['label'] == 1, 'item'])
        prof = profiles.get(u, torch.zeros(embeddings.size(1), device=device))
        sims = util.cos_sim(prof, embeddings)[0]
        topk = torch.topk(sims, k=10).indices.cpu().tolist()
        if any(i in true_set for i in topk):
            hit += 1

        for _, row in grp.iterrows():
            y_true.append(row['label'])
            sc = sims[row['item']].item()
            y_score.append(sc)
            y_pred.append(1 if sc >= 0.5 else 0)

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_score))
    mae = mean_absolute_error(y_true, y_score)
    hit_rate = hit / users if users > 0 else 0

    # Print results
    print(f"\n--- {desc} Text Evaluation Results ---")
    print(f"Relevance Threshold: >= 0.5")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"Hit Rate:  {hit_rate:.4f}")
    print("-------------------------------------------")

# ─── 8. Run & Compare ──────────────────────────────────────────────────────────
evaluate(emb_raw, "Raw")
evaluate(emb_clean, "Cleaned")