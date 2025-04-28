import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import implicit
import numpy as np
from collections import defaultdict
import time # To time the training
import warnings # To potentially suppress the OpenBLAS warning

# Suppress the OpenBLAS warning from the 'implicit' library if it's not relevant to you
# from implicit.cpu.als import check_blas_config
# warnings.filterwarnings("ignore", message="OpenBLAS is configured", category=RuntimeWarning)
# You might need to set environment variables as suggested by the warning for best performance

# --- Configuration ---
REVIEW_DATA_PATH = "../data/processed/reviews_denoised.csv" # Adjust path as needed

# --- Check your actual review CSV column names and adjust these! ---
REVIEW_USER_COL = 'user_id'
REVIEW_ITEM_COL = 'parent_asin' # Assuming reviews are linked via parent_asin
REVIEW_RATING_COL = 'rating'

# --- Implicit ALS Configuration ---
ALS_FACTORS = 50          # Number of latent factors (dimensionality of the embedding)
ALS_REGULARIZATION = 0.01 # Regularization parameter to prevent overfitting
ALS_ITERATIONS = 20       # Number of training iterations
ALPHA_CONFIDENCE = 40     # Factor for confidence (1 + alpha * rating). Higher ratings mean higher confidence.

# --- Evaluation Configuration ---
K_EVAL = 10         # Evaluate Top-K recommendations
TEST_SET_SIZE = 0.20 # Hold out 20% of the data for testing
RELEVANCE_THRESHOLD = 4.0 # Rating threshold for relevant items (ground truth positives)

# --- Helper Functions ---

def load_and_prepare_implicit_data(review_path):
    """
    Loads review data, maps IDs to integers, and prepares for implicit model.
    Returns DataFrame, user_map, item_map, user_inv_map, item_inv_map.
    """
    print(f"Loading reviews from: {review_path}")
    try:
        # Load only necessary columns from reviews
        reviews_df = pd.read_csv(review_path, usecols=[REVIEW_USER_COL, REVIEW_ITEM_COL, REVIEW_RATING_COL])
        # Drop rows with missing values in key columns
        reviews_df.dropna(subset=[REVIEW_USER_COL, REVIEW_ITEM_COL, REVIEW_RATING_COL], inplace=True)
        # Ensure rating is numeric and drop rows where conversion failed
        reviews_df[REVIEW_RATING_COL] = pd.to_numeric(reviews_df[REVIEW_RATING_COL], errors='coerce')
        reviews_df.dropna(subset=[REVIEW_RATING_COL], inplace=True)
        print(f"Loaded {len(reviews_df)} valid reviews.")

    except FileNotFoundError:
        print(f"ERROR: Review file not found at {review_path}")
        return None, None, None, None, None
    except ValueError as e:
        print(f"ERROR loading reviews: {e}. Check column names and data types.")
        return None, None, None, None, None

    # Map user_id and item_id to contiguous integers
    unique_users = reviews_df[REVIEW_USER_COL].unique()
    unique_items = reviews_df[REVIEW_ITEM_COL].unique()

    user_map = {user: i for i, user in enumerate(unique_users)}
    item_map = {item: i for i, item in enumerate(unique_items)}

    # Create inverse maps (useful for converting integer IDs back to original IDs if needed)
    user_inv_map = {i: user for user, i in user_map.items()}
    item_inv_map = {i: item for item, i in item_map.items()}

    reviews_df['user_id_int'] = reviews_df[REVIEW_USER_COL].map(user_map)
    reviews_df['item_id_int'] = reviews_df[REVIEW_ITEM_COL].map(item_map)

    print(f"Mapped {len(unique_users)} users and {len(unique_items)} items to integers.")

    return reviews_df, user_map, item_map, user_inv_map, item_inv_map

def create_sparse_matrix(df, num_users, num_items, use_confidence=True):
    """
    Creates a sparse matrix (user x item) from DataFrame.
    Optionally applies confidence weighting derived from ratings.
    """
    # Ensure integer IDs are correct (they should be after mapping)
    user_ids = df['user_id_int'].values
    item_ids = df['item_id_int'].values

    # Calculate values for the matrix
    if use_confidence:
        # Apply the confidence weighting: 1 + alpha * rating
        # This treats higher ratings as stronger indications of preference (higher confidence)
        confidence_values = 1 + ALPHA_CONFIDENCE * df[REVIEW_RATING_COL].values
        # Ensure confidence values are positive
        confidence_values[confidence_values < 0] = 0 # Should not happen with standard ratings

        print(f"Using confidence = 1 + {ALPHA_CONFIDENCE} * rating")
    else:
        # Binary interaction (value = 1) - useful for test matrix if only checking presence
        confidence_values = np.ones(len(df))
        print("Using binary interaction (value=1)")

    # Create the sparse matrix (CSR format is efficient for row-wise operations like user recommendations)
    sparse_matrix = csr_matrix((confidence_values, (user_ids, item_ids)),
                               shape=(num_users, num_items))

    print(f"Created sparse matrix with shape: {sparse_matrix.shape}")
    return sparse_matrix

def evaluate_implicit_ranking(model, train_matrix, test_df, user_map, item_map, user_inv_map, k=10, threshold=4.0):
    """
    Evaluates the implicit ALS model using Precision@K, Recall@K, F1@K, Hit Rate@K
    on the hold-out test set.
    """
    print(f"\nEvaluating ranking metrics @{k} (Relevance Threshold: >= {threshold})...")

    # Build a dictionary of relevant items for each user in the test set (using integer IDs)
    relevant_test_items_dict = defaultdict(set)
    # Filter test_df for relevant items based on the threshold
    relevant_test_df = test_df[test_df[REVIEW_RATING_COL] >= threshold]

    for index, row in relevant_test_df.iterrows():
        relevant_test_items_dict[row['user_id_int']].add(row['item_id_int'])

    # Get the set of users present in the training data
    # Find unique user indices that have at least one non-zero entry in the training matrix
    train_user_ints_set = set(np.unique(train_matrix.nonzero()[0]))

    evaluated_users_count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    hit_count = 0

    # Iterate through users who are present in the training set AND have relevant items in the test set
    # We only evaluate users for whom we have both training data (to learn from) and relevant test data (as ground truth)
    users_to_evaluate = [uid for uid in relevant_test_items_dict.keys() if uid in train_user_ints_set]

    if not users_to_evaluate:
        print("No users found in the test set with relevant items that are also present in the training set.")
        # Return zero for all metrics if no users can be evaluated
        return 0, 0, 0, 0, 0

    print(f"Evaluating {len(users_to_evaluate)} users with relevant items in test set...")

    # Get the full set of all item integer IDs for filtering recommendations if needed (optional)
    all_item_ints = set(item_map.values())


    for user_int_id in users_to_evaluate:
        # Increment count assuming we will successfully evaluate this user
        # Decrement later if an error occurs
        evaluated_users_count += 1

        true_relevant_item_ints = relevant_test_items_dict[user_int_id]
        num_relevant_in_test = len(true_relevant_item_ints)

        # --- Get recommendations from the model ---
        try:
            # The model.recommend method takes the user ID (int),
            # the user's row from the *training* matrix (to filter seen items),
            # and the number of recommendations N.
            # It returns a tuple: (array of item IDs, array of scores)
            recommended_items_with_scores_tuple = model.recommend(
                user_int_id,
                train_matrix.getrow(user_int_id), # Pass user's row from the training matrix for filtering
                N=k
            )

            # --- Debugging Print ---
            # Print the type and value returned by model.recommend for diagnosis
            print(f"DEBUG: User {user_int_id} (orig: {user_inv_map.get(user_int_id, 'N/A')}): Recommend returned type {type(recommended_items_with_scores_tuple)}")
            # Ensure the output is a tuple before trying to print its elements
            if isinstance(recommended_items_with_scores_tuple, tuple) and len(recommended_items_with_scores_tuple) == 2:
                 print(f"DEBUG: User {user_int_id}: Recommend returned value (first few): ({recommended_items_with_scores_tuple[0][:5] if recommended_items_with_scores_tuple[0] is not None else 'None'}, {recommended_items_with_scores_tuple[1][:5] if recommended_items_with_scores_tuple[1] is not None else 'None'})")
            else:
                 print(f"DEBUG: User {user_int_id}: Recommend returned value (unexpected format): {recommended_items_with_scores_tuple}")

            # --- End Debugging Print ---


            # --- Correctly unpack the tuple output ---
            if recommended_items_with_scores_tuple is None or not isinstance(recommended_items_with_scores_tuple, tuple) or len(recommended_items_with_scores_tuple) != 2:
                 print(f"Warning: recommend() returned unexpected format for user {user_int_id}. Skipping user. Output: {recommended_items_with_scores_tuple}")
                 evaluated_users_count -= 1 # Decrement as we are skipping this user's evaluation
                 continue # Skip to the next user

            recommended_item_ids_array, recommended_scores_array = recommended_items_with_scores_tuple

            # Ensure arrays are not None and have the same length
            if recommended_item_ids_array is None or recommended_scores_array is None or len(recommended_item_ids_array) != len(recommended_scores_array):
                 print(f"Warning: recommend() returned mismatched or None arrays for user {user_int_id}. Skipping user. Output: {recommended_items_with_scores_tuple}")
                 evaluated_users_count -= 1 # Decrement as we are skipping this user's evaluation
                 continue


            # Extract recommended item integer IDs by zipping the arrays
            # Use a list comprehension for clarity before converting to set
            recommended_items_list = list(zip(recommended_item_ids_array, recommended_scores_array))
            recommended_item_ints = {item_int for item_int, score in recommended_items_list}

            # The number of valid recommended items is simply the number of items returned (should be N=k or less)
            valid_recommended_count = len(recommended_items_list)

            # If no items were recommended (empty arrays), skip evaluation for this user
            if valid_recommended_count == 0:
                 print(f"Info: No items recommended for user {user_int_id}. Skipping user.")
                 evaluated_users_count -= 1 # Decrement as we are skipping this user's evaluation
                 continue


        except Exception as e:
            # Catch potential errors during the recommendation process (like the IndexError)
            print(f"Error getting recommendations for user {user_int_id}: {e}")
            evaluated_users_count -= 1 # Decrement as we are skipping this user's evaluation
            continue # Skip to the next user


        # --- Calculate metrics for this user ---
        # Hits are the intersection of recommended items and the true relevant items from the test set
        hits = recommended_item_ints.intersection(true_relevant_item_ints)
        num_hits = len(hits)

        # Calculate metrics for this user
        # Precision@K: Proportion of recommended items that are relevant
        # Use the count of successfully extracted valid recommended items as the denominator
        precision = num_hits / valid_recommended_count if valid_recommended_count > 0 else 0

        # Recall@K: Proportion of relevant items in the test set that were recommended
        recall = num_hits / num_relevant_in_test if num_relevant_in_test > 0 else 0 # num_relevant_in_test should be > 0 for evaluated users

        # F1-Score@K: Harmonic mean of Precision and Recall
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Hit Rate@K: Did we recommend at least one relevant item in the top K?
        is_hit = 1 if num_hits > 0 else 0

        # Accumulate metrics across users
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        hit_count += is_hit

    # Handle case where evaluated_users_count might become 0 due to errors/warnings
    if evaluated_users_count <= 0: # Use <= 0 just in case it started at 0
        print("Evaluation finished, but no users were successfully evaluated due to errors or unexpected recommendation output.")
        return 0, 0, 0, 0, 0

    # --- Average metrics over all successfully evaluated users ---
    avg_precision = total_precision / evaluated_users_count
    avg_recall = total_recall / evaluated_users_count
    avg_f1 = total_f1 / evaluated_users_count
    avg_hit_rate = hit_count / evaluated_users_count

    return avg_precision, avg_recall, avg_f1, avg_hit_rate, evaluated_users_count


def run_implicit_als_evaluation():
    """Loads data, trains Implicit ALS, and evaluates using a hold-out set."""
    print("--- Starting Implicit ALS Evaluation ---")

    # 1. Load and prepare data
    reviews_df, user_map, item_map, user_inv_map, item_inv_map = load_and_prepare_implicit_data(REVIEW_DATA_PATH)

    if reviews_df is None:
        print("\nExiting due to errors during data loading.")
        return

    num_users = len(user_map)
    num_items = len(item_map)
    print(f"Total unique users: {num_users}, Total unique items: {num_items}")


    # 2. Split the DataFrame into training and test sets
    print(f"\nSplitting DataFrame into training and test sets (test_size={TEST_SET_SIZE:.0%})...")
    # Use a simple random split of rows. We removed 'stratify' as it fails for users with single reviews.
    # Evaluation logic will only consider users present in *both* train and test sets.
    train_df, test_df = train_test_split(reviews_df, test_size=TEST_SET_SIZE, random_state=42) # Removed stratify

    print(f"Train set size: {len(train_df)} rows")
    print(f"Test set size:  {len(test_df)} rows")

    # 3. Create sparse matrices from the split DataFrames
    print("\nCreating training sparse matrix with confidence...")
    train_matrix = create_sparse_matrix(train_df, num_users, num_items, use_confidence=True)

    # The implicit ALS model trains on the transposed matrix (item x user)
    train_matrix_transpose = train_matrix.T.tocsr()
    print("Created transposed training sparse matrix (item x user) for training.")


    # 4. Train the Implicit ALS model
    print("\nTraining Implicit ALS model...")
    # Initialize the ALS model
    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=42 # for reproducibility
        # use_gpu=True # Uncomment and set to True if you have a compatible GPU and implicit is built with CUDA support
    )

    # Train the model
    start_time = time.time()
    # The .fit() method requires the item-user matrix
    model.fit(train_matrix_transpose)
    end_time = time.time()
    print(f"Model training complete in {end_time - start_time:.2f} seconds.")

    # 5. Evaluate the model using ranking metrics
    avg_precision, avg_recall, avg_f1, avg_hit_rate, evaluated_users = evaluate_implicit_ranking(
        model,
        train_matrix, # Pass the training matrix to the evaluation function for filtering
        test_df,
        user_map,
        item_map,
        user_inv_map, # <-- Pass user_inv_map here to the evaluation function
        k=K_EVAL,
        threshold=RELEVANCE_THRESHOLD
    )

    # 6. Print evaluation results
    print("\n--- Evaluation Results (Hold-out Set - Implicit ALS) ---")
    print(f"Model: Implicit ALS (factors={ALS_FACTORS}, reg={ALS_REGULARIZATION}, iterations={ALS_ITERATIONS}, alpha={ALPHA_CONFIDENCE})")
    print(f"Relevance Threshold: >= {RELEVANCE_THRESHOLD}")
    print(f"Evaluated Users (present in train & test with relevant items): {evaluated_users}")
    print(f"Precision @{K_EVAL}: {avg_precision:.4f}")
    print(f"Recall @{K_EVAL}:    {avg_recall:.4f}")
    print(f"F1-Score @{K_EVAL}:  {avg_f1:.4f}") # Print the calculated F1 from the evaluation function
    print(f"Hit Rate @{K_EVAL}:  {avg_hit_rate:.4f}")
    print("-------------------------------------------------------")

    # Note: Implicit models typically don't predict explicit rating values directly,
    # they predict preference strength or a ranking score. Therefore, RMSE/MAE metrics
    # for rating prediction accuracy are not applicable or standard for this type of model.
    print("\nRMSE/MAE metrics are not applicable for this Implicit ALS model.")


# --- Main Execution ---
if __name__ == "__main__":
    run_implicit_als_evaluation()