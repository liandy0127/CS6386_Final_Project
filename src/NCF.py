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

# ─── 1. Load & Merge ──────────────────────────────────────────────────────────
reviews = pd.read_csv("../data/processed/reviews_denoised_combined.csv")
meta = pd.read_csv("../data/processed/meta_denoised_combined.csv")

# Merge so every row has a text field
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

# ─── 2. Encode Users, Items & Labels ──────────────────────────────────────────
user_enc = LabelEncoder().fit(df['user_id'])
item_enc = LabelEncoder().fit(df['parent_asin'])
df['user'] = user_enc.transform(df['user_id'])
df['item'] = item_enc.transform(df['parent_asin'])
df['label'] = (df['rating'] >= 4).astype(int)  # Implicit ground truth

# ─── 3. Train/Test Split ──────────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ─── 4. Device Setup ──────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ─── 5. Compute Item Text Embeddings (Optional) ───────────────────────────────
USE_TEXT_EMBEDDINGS = True  # Set to False to use pure NCF without text
if USE_TEXT_EMBEDDINGS:
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Building item text list...")
    item_texts = []
    for i in tqdm(range(len(item_enc.classes_)), desc="Item Texts"):
        item_text = df.loc[df['item'] == i, 'combined_text'].iloc[0] if not df.loc[df['item'] == i, 'combined_text'].empty else "No description available"
        item_texts.append(item_text)

    print("Computing item embeddings...")
    item_text_embeddings = sbert.encode(
        item_texts, convert_to_tensor=True, show_progress_bar=True
    ).to(device)  # shape: [num_items, emb_dim]
else:
    item_text_embeddings = None

# ─── 6. Custom Dataset ────────────────────────────────────────────────────────
class NCFDataset(Dataset):
    def __init__(self, df):
        self.users = df['user'].values
        self.items = df['item'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# ─── 7. NCF Model ─────────────────────────────────────────────────────────────
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, text_emb_dim=384 if USE_TEXT_EMBEDDINGS else 0):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2 + text_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids, text_embeddings=None):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        if text_embeddings is not None:
            item_text_emb = text_embeddings[item_ids]
            x = torch.cat([user_emb, item_emb, item_text_emb], dim=1)
        else:
            x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc_layers(x).squeeze()

# ─── 8. Training Function ─────────────────────────────────────────────────────
def train_ncf(model, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids, item_text_embeddings if USE_TEXT_EMBEDDINGS else None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")

# ─── 9. Evaluation Function ───────────────────────────────────────────────────
def evaluate_ncf(model, test_df, num_items):
    model.eval()
    y_true, y_score, y_pred = [], [], []
    hit_count = 0
    num_users = test_df['user'].nunique()

    with torch.no_grad():
        for u, grp in tqdm(test_df.groupby('user'), total=num_users, desc="Evaluating"):
            true_set = set(grp.loc[grp['label'] == 1, 'item'])
            user_id = torch.tensor([u] * num_items, device=device)
            item_ids = torch.arange(num_items, device=device)
            scores = model(user_id, item_ids, item_text_embeddings if USE_TEXT_EMBEDDINGS else None)
            topk = torch.topk(scores, k=10).indices.cpu().tolist()
            if any(i in true_set for i in topk):
                hit_count += 1

            for _, row in grp.iterrows():
                y_true.append(row['label'])
                score = scores[row['item']].item()
                y_score.append(score)
                y_pred.append(1 if score >= 0.5 else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_score))
    mae = mean_absolute_error(y_true, y_score)
    hit_rate = hit_count / num_users if num_users > 0 else 0

    print("\n--- NCF Evaluation Results (Hold-out Set) ---")
    print(f"Relevance Threshold: >= 0.5")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"Hit Rate:  {hit_rate:.4f}")
    print("-------------------------------------------")

# ─── 10. Main Execution ───────────────────────────────────────────────────────
# Prepare data
train_dataset = NCFDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize model
num_users = len(user_enc.classes_)
num_items = len(item_enc.classes_)
model = NCF(num_users, num_items).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Train model
print("Training NCF model...")
train_ncf(model, train_loader, optimizer, criterion, num_epochs=5)

# Evaluate model
print("Evaluating NCF model...")
evaluate_ncf(model, test_df, num_items)