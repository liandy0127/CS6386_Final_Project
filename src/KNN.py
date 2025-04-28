# KNN.py
import pandas as pd
import json
# import ast # Not needed if JSON is loaded directly
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
from tqdm.auto import tqdm # Use auto for compatibility

# --- Data Loading Function (Robust Version) ---
def load_review_data(path):
    """
    Load reviews into DataFrame & Surprise Dataset from CSV, cleaning column names.
    Includes checks for required columns and handles missing values.
    """
    print(f"\n--- Loading review data from: {path} ---")
    df = None # Initialize df to None
    try:
        # Read all columns first to handle potential whitespace in headers
        df = pd.read_csv(path)
        # print(f"  Original columns: {df.columns.tolist()}") # Too verbose

        # Clean column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()
        # print(f"  Cleaned columns: {df.columns.tolist()}") # Too verbose


        # Now select the required columns for Surprise: user_id, item_id (parent_asin), rating
        # Ensure parent_asin is also selected as it's used as the item ID
        required_cols = ['user_id', 'parent_asin', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  Available columns after cleaning names: {df.columns.tolist()}")
            raise ValueError(f"Required columns for Surprise not found after cleaning names: {missing_cols}")

        # Select only the required columns and create a copy to avoid SettingWithCopyWarning
        df = df[required_cols].copy()
        print(f"  Successfully loaded and selected columns: {required_cols}")


        # Handle missing values in the required columns BEFORE loading into Surprise
        initial_rows = len(df)
        df.dropna(subset=required_cols, inplace=True)
        rows_after_dropna = len(df)
        if initial_rows != rows_after_dropna:
             print(f"  Dropped {initial_rows - rows_after_dropna} rows with missing values in essential columns ({required_cols}).")

        # Ensure rating is numeric, coercing errors to NaN, and drop resulting NaNs
        initial_rows_rating_convert = len(df)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(subset=['rating'], inplace=True)
        rows_after_dropna_rating = len(df)
        if initial_rows_rating_convert != rows_after_dropna_rating:
             print(f"  Dropped {initial_rows_rating_convert - rows_after_dropna_rating} rows where rating conversion failed.")


        print(f"  Final number of rows for Surprise: {len(df)}")

        if df.empty:
             print("  Warning: DataFrame is empty after cleaning and dropping NaNs. Cannot load into Surprise Dataset.")
             # Return empty df and None for data to handle gracefully later
             return df, None


        # Load data into Surprise Dataset format
        # Adjust rating_scale if your ratings are not strictly 1-5 (e.g., if outliers were capped)
        reader = Reader(rating_scale=(1,5)) # Assuming ratings are still within 1-5 after cleaning/capping
        data = Dataset.load_from_df(df[required_cols], reader)
        print("  Data loaded into Surprise Dataset.")

        return df, data # Return both the DataFrame and the Surprise data object

    except FileNotFoundError:
        print(f"Error: Review data file not found at {path}")
        return None, None # Return None, None if file not found
    except Exception as e:
         print(f"An unexpected error occurred during load_review_data: {e}")
         return None, None # Return None, None on other errors


# --- Ground Truth Building (from item-level JSON based on reviews) - Based on your provided logic ---
def build_user_gt(reviews_df, item_gt_json_path):
    """
    Build user->set(recommended items) based on their reviews and an item-level ground truth JSON.
    Assumes item_gt_json_path points to a JSON file like {item_id: [recommended_item1, ...]}
    where recommended items are typically 'bought_together'.
    """
    print(f"\n--- Building user ground truth based on reviews and item GT from: {item_gt_json_path} ---")
    item_gt = {}
    user_gt = {}

    # Load the item-level ground truth JSON
    try:
        with open(item_gt_json_path, 'r') as f:
            item_gt_raw = json.load(f)

        # Process item_gt - ensure keys/values are strings and strip whitespace
        for item_id_raw, recommended_items in tqdm(item_gt_raw.items(), desc="Loading Item GT JSON"):
             cleaned_item_id = str(item_id_raw).strip()
             if isinstance(recommended_items, list):
                 # Filter out any non-string/empty items and clean item IDs
                 cleaned_recs = {str(item).strip() for item in recommended_items if isinstance(item, (str, int, float)) and str(item).strip()}
                 if cleaned_recs:
                     item_gt[cleaned_item_id] = cleaned_recs

        print(f"  Loaded item -> recommendations map for {len(item_gt)} items.")

    except FileNotFoundError:
        print(f"Error: Item ground truth JSON file not found at {item_gt_json_path}. Cannot build user GT.")
        # Return empty user_gt
        return user_gt
    except json.JSONDecodeError as e:
        print(f"Error decoding item ground truth JSON file at {item_gt_json_path}: {e}. Cannot build user GT.")
         # Return empty user_gt on error
        return user_gt
    except Exception as e:
         print(f"An unexpected error occurred during item ground truth loading: {e}. Cannot build user GT.")
          # Return empty user_gt on error
         return user_gt


    # Build user -> recommended items based on their reviews and the item_gt map
    # Only use users from the reviews_df that have a valid 'user_id'
    review_users_df = reviews_df.dropna(subset=['user_id'])
    review_users = review_users_df['user_id'].unique()

    print(f"Building user -> ground truth recommendations map for {len(review_users)} users from this reviews dataset...")

    # Group by user_id from the reviews_df that has valid user_ids
    for user_raw, grp in tqdm(review_users_df.groupby('user_id'), desc="Building User GT"):
        # Get items this user bought/rated from this specific reviews_df
        # Ensure parent_asin is not null
        bought_items_by_user = set(grp['parent_asin'].dropna().unique()) # Use parent_asin as item ID

        # Items recommended based on their bought items (from item_gt)
        recommended_items_for_user = set()
        for bought_item in bought_items_by_user:
            # Get recommended items for each item the user bought/rated
            # Use item_gt.get(item, set()) to safely handle items not found in item_gt
            recommended_items_for_user.update(item_gt.get(bought_item, set()))

        # Remove items the user already bought/rated from the recommendations
        recommended_items_for_user -= bought_items_by_user

        if recommended_items_for_user:
            user_gt[user_raw] = recommended_items_for_user

    print(f"  Built user -> ground truth map for {len(user_gt)} users.")
    return user_gt


# --- Evaluation Function (Based on your provided logic) ---
def evaluate_knn(reviews_df, data, user_gt, # Note: reviews_df is passed but not strictly needed in this version
                 k_neighbors=40, k_eval=10, test_size=0.2):
    """Train KNN, recommend Top-K, compute metrics using a single train/test split."""

    if data is None:
         print("\n--- Evaluation Skipped ---")
         print("Surprise Dataset is empty or could not be loaded.")
         print("--------------------------")
         return

    print("\n--- Training KNN model for Evaluation (Train/Test Split) ---")
    # 1) Split into train/test for RMSE/MAE and Recommendation Evaluation
    # The trainset from this split will be used to train the model
    # for BOTH RMSE/MAE and recommendation candidate generation/scoring.
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)

    algo = KNNBasic(k=k_neighbors, sim_options={'name':'cosine','user_based':True})
    print(f"Training KNNBasic (user_based, k={k_neighbors}, cosine similarity) on train/test split ({1-test_size:.0%}/{test_size:.0%})...")
    algo.fit(trainset)
    print("Training complete.")

    # 2) Hold-out rating accuracy on the testset split
    print("\n--- Evaluating Rating Accuracy (RMSE/MAE) on Hold-Out Test Set ---")
    preds_test = algo.test(testset)
    rmse = accuracy.rmse(preds_test, verbose=False)
    mae  = accuracy.mae(preds_test,  verbose=False)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")


    print("\n--- Generating Top-K Recommendations and Computing Precision/Recall ---")
    if not user_gt:
         print("  User ground truth is empty. Skipping recommendation evaluation.")
         print("-----------------------------------------")
         return # Skip recommendation evaluation if no ground truth

    precisions, recalls = [], []
    hits = eval_users = 0

    # Iterate through users who have ground truth recommendations
    # Only evaluate users who are also present in the *trainset split*
    users_to_evaluate = [user_raw for user_raw in user_gt.keys()] # Get all users with GT
    print(f"Attempting to evaluate {len(users_to_evaluate)} users with ground truth...")

    # Get mapping from inner to raw item IDs from the trainset split
    # Build this using trainset.all_items() and trainset.to_raw_iid()
    print("Building inner -> raw item ID mapping from trainset...")
    inner_to_raw_iid = {}
    # iterate over all inner item ids in the trainset
    for inner_iid in tqdm(trainset.all_items(), desc="Mapping inner to raw iids"):
         raw_iid = trainset.to_raw_iid(inner_iid)
         inner_to_raw_iid[inner_iid] = raw_iid
    print(f"  Built mapping for {len(inner_to_raw_iid)} items.")


    for user_raw in tqdm(users_to_evaluate, desc="Evaluating Users"):
        try:
            # Check if user exists in the trainset split and get inner ID
            u_inner = trainset.to_inner_uid(user_raw)
        except ValueError:
            # print(f"  User {user_raw} not in trainset split. Skipping.") # Too verbose
            continue # Skip users not in this trainset split

        truths_raw_set = user_gt[user_raw] # Get ground truth items for this user (already a set)

        # Get items the user has already rated in the trainset split (using inner IDs)
        user_rated_items_inner = {iid for (iid, _) in trainset.ur[u_inner]}


        # Build candidate pool from items rated by neighbors in the trainset split
        # Exclude items the target user has already rated (using inner IDs)
        candidates_iid_inner = set()
        try:
            # Get neighbors for the user from the model trained on the trainset split
            # algo.get_neighbors works with inner UIDs
            # Handle case where get_neighbors might return empty list or raise error
            nbrs_inner = []
            try:
                nbrs_inner = algo.get_neighbors(u_inner, k=k_neighbors)
            except ValueError: # e.g., if k_neighbors is too large for this user's data
                 pass # nbrs_inner remains empty

            if not nbrs_inner:
                 # print(f"  No neighbors found in trainset split for user {user_raw}.") # Too verbose
                 continue # Skip if no neighbors

            for nb_inner in nbrs_inner:
                 # Add items rated by neighbor, excluding items the target user already rated (using inner IDs)
                 candidates_iid_inner.update({iid for (iid, _) in trainset.ur[nb_inner] if iid not in user_rated_items_inner})

        except Exception as e:
             # print(f"  Unexpected error getting neighbors or candidates for user {user_raw}: {e}. Skipping user.") # Too verbose
             continue # Skip user on unexpected error


        if not candidates_iid_inner:
            # print(f"  No unseen candidate items found through neighbors in trainset split for user {user_raw}.") # Too verbose
            continue # Skip if no unseen candidates


        # Convert candidate inner item IDs back to raw item IDs for scoring and comparison with ground truth
        candidates_raw = [inner_to_raw_iid[iid] for iid in candidates_iid_inner]

        # Score candidates using the model trained on the trainset split
        # algo.predict expects raw user and item IDs
        scores = [(it_raw, algo.predict(user_raw, it_raw, verbose=False).est) for it_raw in candidates_raw]

        # Select Top-K recommended items based on scores
        # Ensure we have enough candidates to get K recommendations
        # Sort by score (descending) and take the top k_eval items
        topk_raw = [it_raw for it_raw, score in sorted(scores, key=lambda x: x[1], reverse=True)[:k_eval]]

        if not topk_raw:
             # print(f"  Could not generate Top-{k_eval} recommendations for user {user_raw}.") # Too verbose
             continue # Skip if no recommendations generated


        # Compute hits by comparing Top-K recommendations (raw IDs) to ground truth items (raw IDs)
        hit_count = len(set(topk_raw) & truths_raw_set) # truths_raw_set is already a set of raw IDs

        # Compute Precision and Recall for this user
        precision_u = hit_count / k_eval if k_eval else 0.0
        # Recall denominator should be the number of ground truth items for the user
        # Ensure denominator is not zero
        recall_denominator = len(truths_raw_set)
        recall_u = hit_count / recall_denominator if recall_denominator else 0.0 # Recall relative to the user's GT size


        precisions.append(precision_u)
        recalls.append(recall_u)

        # Only count a user as evaluated if we successfully generated recommendations for them
        # and they were in the ground truth and the trainset split.
        eval_users += 1


    # 7) Aggregate metrics across all evaluated users
    # Ensure division by zero is handled if eval_users is 0
    avg_prec = sum(precisions) / eval_users if eval_users > 0 else 0.0
    avg_rec  = sum(recalls)    / eval_users if eval_users > 0 else 0.0
    # Calculate F1 score - handle the case where avg_prec + avg_rec is zero or near zero
    avg_f1   = (2 * avg_prec * avg_rec / (avg_prec + avg_rec)) if (avg_prec + avg_rec) > 1e-9 else 0.0
    hit_rate = sum(p > 1e-9 for p in precisions) / eval_users if eval_users > 0 else 0.0 # Proportion of users with at least one hit (precision > 0)

    # 8) Print final results
    print("\n--- Recommendation Evaluation Results ---")
    print(f"Users in Ground Truth File: {len(user_gt)}") # Total users found in the GT file
    print(f"Users Evaluated for Recommendations: {eval_users}") # Users from GT who were successfully processed
    print(f"Precision @{k_eval}: {avg_prec:.4f}")
    print(f"Recall    @{k_eval}: {avg_rec:.4f}")
    print(f"F1-Score  @{k_eval}: {avg_f1:.4f}")
    print(f"Hit Rate @{k_eval}:  {hit_rate:.4f}")
    # RMSE and MAE were printed earlier after the rating accuracy evaluation block
    print("-----------------------------------------")


# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Evaluate KNN recommendation model performance on different data cleaning levels.")
    p.add_argument("--ground_truth", default="../data/processed/ground_truth.json",
                   help="Path to the item-level ground truth JSON file ({item_id: [recommended_item1, ...]}).")
    p.add_argument("--noisy_reviews", default="../data/processed/reviews_noisy.csv",
                   help="Path to the noisy review data CSV file.")
    p.add_argument("--text_cleaned_reviews", default="../data/processed/reviews_text_cleaned.csv",
                   help="Path to the text cleaned review data CSV file.")
    p.add_argument("--missing_handled_reviews", default="../data/processed/reviews_missing_handled.csv",
                   help="Path to the missing handled review data CSV file.")
    p.add_argument("--outliers_handled_reviews", default="../data/processed/reviews_outliers_handled.csv",
                   help="Path to the outliers handled review data CSV file.")
    p.add_argument("--denoised_reviews", default="../data/processed/reviews_denoised_combined.csv",
                   help="Path to the fully denoised review data CSV file.")


    p.add_argument("--k_neighbors",  type=int,   default=40,
                   help="Number of neighbors to use for KNN.")
    p.add_argument("--k_eval",       type=int,   default=10,
                   help="Number of top items to recommend for evaluation.")
    p.add_argument("--test_size",    type=float, default=0.2,
                   help="Proportion of data to use for the rating accuracy test set (RMSE/MAE).")
    args = p.parse_args()

    # --- Load Item-Level Ground Truth ---
    # Load item-level ground truth once
    item_ground_truth_map = {} # Initialize as empty dict
    try:
        print(f"\n--- Loading item-level ground truth from: {args.ground_truth} ---")
        with open(args.ground_truth, 'r') as f:
             item_ground_truth_raw = json.load(f)
        # Ensure keys and list elements are strings and stripped
        for item_id_raw, recommended_items in tqdm(item_ground_truth_raw.items(), desc="Loading Item GT JSON"):
             cleaned_item_id = str(item_id_raw).strip()
             if isinstance(recommended_items, list):
                 cleaned_recs = {str(item).strip() for item in recommended_items if isinstance(item, (str, int, float)) and str(item).strip()}
                 if cleaned_recs:
                     item_ground_truth_map[cleaned_item_id] = cleaned_recs

        print(f"  Loaded item -> recommendations map for {len(item_ground_truth_map)} items.")

    except FileNotFoundError:
        print(f"Error: Item ground truth JSON file not found at {args.ground_truth}. Cannot perform recommendation evaluation.")
        item_ground_truth_map = {} # Ensure it's empty
    except json.JSONDecodeError as e:
        print(f"Error decoding item ground truth JSON file at {args.ground_truth}: {e}. Cannot perform recommendation evaluation.")
        item_ground_truth_map = {} # Ensure it's empty
    except Exception as e:
         print(f"An unexpected error occurred during item ground truth loading: {e}. Cannot perform recommendation evaluation.")
         item_ground_truth_map = {} # Ensure it's empty


    # --- Define datasets to evaluate ---
    datasets_to_evaluate = [
        ('Noisy Data', args.noisy_reviews),
        ('Text Cleaned Data', args.text_cleaned_reviews),
        ('Missing Handled Data', args.missing_handled_reviews),
        ('Outliers Handled Data', args.outliers_handled_reviews),
        ('Fully Denoised Data', args.denoised_reviews),
    ]

    # --- Run Evaluation for Each Dataset ---
    print("\n*** Starting KNN Evaluation on Different Data Cleaning Levels ***")
    for description, reviews_filepath in datasets_to_evaluate:
        print(f"\n--- Evaluating KNN on: {description} ---")

        # Load the review data for the current cleaning level
        reviews_df, surprise_data = load_review_data(reviews_filepath)

        # Check if data loading was successful
        if reviews_df is None or surprise_data is None:
             print(f"Skipping evaluation for {description} due to data loading error.")
             print(f"--- Finished Evaluation on: {description} ---")
             continue # Skip to the next dataset

        # Build user ground truth using the loaded reviews_df and the item_ground_truth_map
        # This rebuilds user GT for each dataset based on which items users reviewed in *that* dataset
        user_ground_truth_for_dataset = {}
        # Only attempt to build user GT if item GT was loaded and there are reviews
        if item_ground_truth_map and not reviews_df.empty:
             print("\n--- Building user ground truth for this dataset ---")
             # Based on your provided build_user_gt logic:
             # Ensure reviews_df has 'user_id' and 'parent_asin' for groupby
             if 'user_id' in reviews_df.columns and 'parent_asin' in reviews_df.columns:
                 review_users_df = reviews_df.dropna(subset=['user_id', 'parent_asin'])
                 review_users = review_users_df['user_id'].unique()
                 print(f"Building user -> ground truth recommendations map for {len(review_users)} users from this reviews dataset...")

                 for user_raw, grp in tqdm(review_users_df.groupby('user_id'), desc="Building User GT"):
                    bought_items_by_user = set(grp['parent_asin'].dropna().unique())
                    recommended_items_for_user = set()
                    for bought_item in bought_items_by_user:
                        # Get recommended items for each item the user bought/rated
                        # Use item_ground_truth_map.get(item, set()) to safely handle items not found in item_gt
                        recommended_items_for_user.update(item_ground_truth_map.get(bought_item, set()))
                    recommended_items_for_user -= bought_items_by_user # Remove items user already reviewed
                    if recommended_items_for_user:
                        user_ground_truth_for_dataset[user_raw] = recommended_items_for_user
                 print(f"  Built user -> ground truth map for {len(user_ground_truth_for_dataset)} users.")
             else:
                 print("  Skipping user ground truth building: 'user_id' or 'parent_asin' column missing in reviews data.")


        # Evaluate KNN model performance using the user_ground_truth_for_dataset
        # Pass reviews_df as the original evaluate_knn expects it (though it might not use it directly)
        evaluate_knn(
            reviews_df, # reviews_df from the current dataset
            surprise_data, # Surprise Dataset object for the current dataset
            user_ground_truth_for_dataset, # User ground truth derived from this dataset
            k_neighbors=args.k_neighbors,
            k_eval=args.k_eval,
            test_size=args.test_size
        )
        print(f"\n--- Finished Evaluation on: {description} ---")

    print("\n*** KNN evaluation script finished. ***")