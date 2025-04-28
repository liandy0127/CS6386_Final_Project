import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util

# ─── 0. Dependencies ──────────────────────────────────────────────────────────
# pip install pandas numpy torch sentence-transformers scikit-learn tqdm

# ─── 1. Load & Merge with Renamed Columns ────────────────────────────────────
reviews = pd.read_csv("../data/processed/reviews_denoised_combined.csv")
meta = pd.read_csv("../data/processed/meta_denoised_combined.csv")

# Merge so every row has exactly one text field
df = pd.merge(
    reviews[['user_id', 'parent_asin', 'rating', 'text']],
    meta[['parent_asin', 'description']],
    on='parent_asin', how='left'
)

# Create combined_text, prioritizing review text
df['combined_text'] = df['text'].fillna('').astype(str)
df['combined_text'] = df['combined_text'].where(df['combined_text'] != '', df['description'].fillna(''))

# Drop rows where combined_text is empty
df = df[df['combined_text'] != ''].reset_index(drop=True)

# ─── 2. Encode Users, Items & Labels ──────────────────────────────────────────
user_enc = LabelEncoder().fit(df['user_id'])
item_enc = LabelEncoder().fit(df['parent_asin'])
df['user'] = user_enc.transform(df['user_id'])
df['item'] = item_enc.transform(df['parent_asin'])
df['label'] = (df['rating'] >= 4).astype(int)

# ─── 3. Train/Test Split ──────────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ─── 4. Device Setup (CPU or MPS on M1 Pro) ───────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ─── 5. Compute Item Embeddings ───────────────────────────────────────────────
sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Build a list of texts in item_index order
print("Building item text list...")
item_texts = []
for i in tqdm(range(len(item_enc.classes_)), desc="Item Texts"):
    item_text = df.loc[df['item'] == i, 'combined_text'].iloc[0] if not df.loc[df['item'] == i, 'combined_text'].empty else "No description available"
    item_texts.append(item_text)

print("Computing item embeddings...")
item_embeddings = sbert.encode(
    item_texts, convert_to_tensor=True, show_progress_bar=True
).to(device)  # shape: [num_items, emb_dim]

# ─── 6. Build User Profiles ───────────────────────────────────────────────────
print("Building user profiles...")
user_profiles = {}
for u, grp in tqdm(train_df.groupby('user'), desc="Users"):
    pos_items = grp.loc[grp['label'] == 1, 'item'].values
    if len(pos_items):
        user_profiles[u] = item_embeddings[pos_items].mean(dim=0)
    else:
        user_profiles[u] = torch.zeros(item_embeddings.size(1), device=device)

# ─── 7. Recommendation & Metrics Collection ──────────────────────────────────
print("Scoring and collecting metrics...")
y_true, y_score, y_pred = [], [], []
hit_count = 0
num_users = test_df['user'].nunique()

for u, grp in tqdm(test_df.groupby('user'), total=num_users, desc="Evaluating"):
    true_set = set(grp.loc[grp['label'] == 1, 'item'])
    profile = user_profiles.get(u, torch.zeros(item_embeddings.size(1), device=device))

    sims = util.cos_sim(profile, item_embeddings)[0]  # [num_items]
    topk = torch.topk(sims, k=10).indices.cpu().tolist()

    if any(i in true_set for i in topk):
        hit_count += 1

    for _, row in grp.iterrows():
        y_true.append(row['label'])
        score = sims[row['item']].item()
        y_score.append(score)
        y_pred.append(1 if score >= 0.5 else 0)

# ─── 8. Compute & Print Metrics ───────────────────────────────────────────────
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_score))
mae = mean_absolute_error(y_true, y_score)
hit_rate = hit_count / num_users if num_users > 0 else 0

print("\n--- Evaluation Results (Hold-out Set) ---")
print(f"Relevance Threshold: >= 0.5")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"Hit Rate:  {hit_rate:.4f}")
print("-------------------------------------------")