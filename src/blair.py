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
from sentence_transformers import util  # for cos_sim :contentReference[oaicite:5]{index=5}

# ─── Dependencies ───────────────────────────────────────────────────────────────
# pip install pandas numpy torch transformers sentence-transformers scikit-learn tqdm nltk

# Ensure NLTK data is available
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ─── 1. Data-Cleaning Function ──────────────────────────────────────────────────
stop_words  = set(stopwords.words('english'))                             # :contentReference[oaicite:6]{index=6}
lemmatizer  = WordNetLemmatizer()                                        # 

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)                                 # remove HTML :contentReference[oaicite:7]{index=7}
    text = text.lower()                                                  # lowercase :contentReference[oaicite:8]{index=8}
    text = re.sub(r'[^a-z\s]', ' ', text)                                # strip punctuation :contentReference[oaicite:9]{index=9}
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)                                        # lemmatize 
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]
    return " ".join(tokens)

# ─── 2. Load & Merge ────────────────────────────────────────────────────────────
reviews = pd.read_csv("../data/processed/reviews_noisy.csv")
meta    = pd.read_csv("../data/processed/meta_noisy.csv")

# Rename metadata processed_text → 'text'
meta.rename(columns={'processed_text': 'text'}, inplace=True)            # :contentReference[oaicite:10]{index=10}

# Merge so every row has one 'text' field
df = pd.merge(
    reviews[['user_id','parent_asin','rating','processed_text']],
    meta [['parent_asin','text']],
    on='parent_asin', how='left'
)
df['text'] = df['text'].fillna(df['processed_text'])

# ─── 3. Encode & Split ─────────────────────────────────────────────────────────
user_enc = LabelEncoder().fit(df['user_id'])
item_enc = LabelEncoder().fit(df['parent_asin'])
df['user']  = user_enc.transform(df['user_id'])                           # :contentReference[oaicite:11]{index=11}
df['item']  = item_enc.transform(df['parent_asin'])
df['label'] = (df['rating'] >= 4).astype(int)                             # implicit ground truth :contentReference[oaicite:12]{index=12}

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ─── 4. Device Setup ───────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # :contentReference[oaicite:13]{index=13}
print("Using device:", device)

# ─── 5. Load BLaIR Model ────────────────────────────────────────────────────────
model_name = "hyp1231/blair-roberta-base"                                # :contentReference[oaicite:14]{index=14}
tokenizer  = AutoTokenizer.from_pretrained(model_name)                   # :contentReference[oaicite:15]{index=15}
model      = AutoModel.from_pretrained(model_name).to(device)

def embed_texts(texts, clean=False):
    """Batch-encode a list of texts with optional cleaning."""
    if clean:
        texts = [clean_text(t) for t in texts]
    embeddings = []
    for i in tqdm(range(0, len(texts), 16), desc="Embedding batches"):
        batch = texts[i:i+16]
        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        max_length=128,         # shorter sequences :contentReference[oaicite:16]{index=16}
                        return_tensors="pt")
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state[:,0]
            out = out / out.norm(dim=1, keepdim=True)
        embeddings.append(out)
    return torch.cat(embeddings, dim=0)  

# ─── 6. Prepare Item Embeddings ───────────────────────────────────────────────
item_texts_raw   = [df.loc[df['item']==i, 'text'].iloc[0] for i in range(len(item_enc.classes_))]
print("Computing raw embeddings...")
emb_raw   = embed_texts(item_texts_raw, clean=False)
print("Computing cleaned embeddings...")
emb_clean = embed_texts(item_texts_raw, clean=True)

# ─── 7. Evaluate Function ──────────────────────────────────────────────────────
def evaluate(embeddings, desc):
    # Build user profiles
    profiles = {}
    for u, grp in train_df.groupby('user'):
        pos = grp.loc[grp['label']==1, 'item'].values
        profiles[u] = embeddings[pos].mean(dim=0) if len(pos) else torch.zeros(embeddings.size(1), device=device)

    # Metrics collection
    y_true, y_score, y_pred = [], [], []
    hit, users = 0, test_df['user'].nunique()

    for u, grp in tqdm(test_df.groupby('user'), total=users, desc=f"Evaluating {desc}"):
        true_set = set(grp.loc[grp['label']==1, 'item'])
        prof     = profiles.get(u, torch.zeros(embeddings.size(1), device=device))
        sims     = util.cos_sim(prof, embeddings)[0]                         # :contentReference[oaicite:17]{index=17}
        topk     = torch.topk(sims, k=10).indices.cpu().tolist()
        if any(i in true_set for i in topk): hit += 1

        for _, row in grp.iterrows():
            y_true.append(row['label'])
            sc = sims[row['item']].item()
            y_score.append(sc)
            y_pred.append(1 if sc >= 0.5 else 0)

    # Compute metrics
    precision = precision_score(y_true, y_pred)                            # :contentReference[oaicite:18]{index=18}
    recall    = recall_score(y_true, y_pred)                               # :contentReference[oaicite:19]{index=19}
    f1        = f1_score(y_true, y_pred)                                   # :contentReference[oaicite:20]{index=20}
    rmse      = np.sqrt(mean_squared_error(y_true, y_score))              # :contentReference[oaicite:21]{index=21}
    mae       = mean_absolute_error(y_true, y_score)                       # :contentReference[oaicite:22]{index=22}
    hit_rate  = hit / users

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
evaluate(emb_raw,   "Raw")
evaluate(emb_clean, "Cleaned")
