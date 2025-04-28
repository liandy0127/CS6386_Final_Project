# KNN.py ──────────────────────────────────────────────────────────────
import pandas as pd, json
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from tqdm.auto import tqdm

DATA      = "../data/processed/raw_reviews_baseline.csv"
GROUND_TR = "ground_truth.json"
K_NEIGH   = 40
TOP_K     = 10
TEST_SIZE = 0.20
SIM_OPTS  = {"name": "cosine", "user_based": True}

# --------------------------------------------------------------------
def load_surprise(path):
    """
    Read CSV → pandas → Surprise Dataset.
    • Drop rows with rating ≤ 0 or NaN (Surprise expects 1–5).
    """
    df = pd.read_csv(path, usecols=["user_id", "parent_asin", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # remove invalid ratings (<=0) or missing
    df = df[df["rating"] > 0].dropna(subset=["user_id", "parent_asin", "rating"])

    reader = Reader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(df[["user_id", "parent_asin", "rating"]], reader)
    return df, data

def build_user_gt(df, gt_path):
    with open(gt_path) as f:
        item_gt = json.load(f)
    user_gt = {}
    for u, grp in df.groupby("user_id"):
        bought = set(grp["parent_asin"])
        # union all co-purchased sets, then remove items already bought
        recs = set().union(*(item_gt.get(i, []) for i in bought)) - bought
        if recs:
            user_gt[u] = recs
    return user_gt

# --------------------------------------------------------------------
def evaluate(df, data, user_gt):
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    algo = KNNBasic(k=K_NEIGH, sim_options=SIM_OPTS)
    algo.fit(train)

    # rating-prediction accuracy on hold-out test set
    preds_test = algo.test(test)
    rmse = accuracy.rmse(preds_test, verbose=False)
    mae  = accuracy.mae (preds_test, verbose=False)

    # Top-K eval against ground truth
    precisions, recalls = [], []
    hits = n_users = 0
    for user, truths in tqdm(user_gt.items(), desc="Users"):
        try:
            u_inner = train.to_inner_uid(user)
        except ValueError:          # user not in training split
            continue
        neighbors  = algo.get_neighbors(u_inner, K_NEIGH)
        cand_iids  = {iid for nb in neighbors for iid, _ in train.ur[nb]}
        candidates = [train.to_raw_iid(iid) for iid in cand_iids]

        scores = [(iid, algo.predict(user, iid).est) for iid in candidates]
        topk   = [iid for iid, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]]

        hit = len(set(topk) & truths)
        precisions.append(hit / TOP_K)
        recalls.append(hit / min(len(truths), TOP_K))
        if hit: hits += 1
        n_users += 1

    p = sum(precisions)/n_users
    r = sum(recalls)/n_users
    f = 2*p*r/(p+r) if p+r else 0
    hr= hits/n_users

    print("\n── KNN Evaluation (denoised data) ──")
    print(f"Users evaluated        : {n_users}")
    print(f"Precision  @{TOP_K:2d}     : {p:.4f}")
    print(f"Recall     @{TOP_K:2d}     : {r:.4f}")
    print(f"F1-Score   @{TOP_K:2d}     : {f:.4f}")
    print(f"Hit-Rate   @{TOP_K:2d}     : {hr:.4f}")
    print(f"RMSE (ratings)         : {rmse:.4f}")
    print(f"MAE  (ratings)         : {mae:.4f}")
    print("────────────────────────────────────\n")

# --------------------------------------------------------------------
if __name__ == "__main__":
    df, surprise_data = load_surprise(DATA)
    user_gt = build_user_gt(df, GROUND_TR)
    evaluate(df, surprise_data, user_gt)
