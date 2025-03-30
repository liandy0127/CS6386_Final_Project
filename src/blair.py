import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Model name for the large BLaIR checkpoint
MODEL_NAME = "hyp1231/blair-roberta-large"

# Paths to your CSV files
BASELINE_METADATA_PATH = "../data/processed/processed_meta_baseline.csv"
BLOOM_FILTERED_PATH = "../data/processed/filtered_reviews_bloom.csv"

# Example context and item metadata for the "Video Games" category
language_context = (
    "I'm looking for a video game with an immersive storyline, "
    "engaging gameplay, and robust multiplayer features."
)

# Example item metadata (or short descriptions). In real usage, you'd likely
# extract these from your metadata file, or from user inputs, etc.
item_metadata = [
    "Nintendo Switch - The Legend of Zelda: Breath of the Wild, "
    "an expansive open-world adventure with puzzles, exploration, and action.",
    "PlayStation 5 - God of War Ragnarök, a narrative-driven action game "
    "featuring deep mythology, cinematic combat, and rich visuals."
]


# ------------------------------------------------------------------------------
# Helper Function: Generate Embeddings
# ------------------------------------------------------------------------------
def generate_embeddings(text_list, model, tokenizer):
    """
    Given a list of text items and a loaded model/tokenizer,
    return normalized embeddings.
    """
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        # We take the [CLS] token (first token) representation: outputs.last_hidden_state[:, 0]
        embeddings = outputs.last_hidden_state[:, 0]
        # Normalize each embedding vector
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings


# ------------------------------------------------------------------------------
# Recommendation from Baseline Metadata
# ------------------------------------------------------------------------------
def recommend_from_baseline(language_context, item_metadata):
    print("===== Baseline Metadata Recommendations =====")

    # Load baseline metadata (must contain columns like 'title', 'average_rating', 'price')
    baseline_df = pd.read_csv(BASELINE_METADATA_PATH)

    # Prepare text list: first is user context, followed by item metadata
    text_list = [language_context] + item_metadata

    # Generate embeddings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embeddings = generate_embeddings(text_list, model, tokenizer)

    # Compute similarity (context vs. each item)
    # embeddings[0] is the context, embeddings[1:] are the items
    similarity_scores = torch.mm(embeddings[0:1], embeddings[1:].T).squeeze().cpu().numpy()

    # Sort items by similarity
    top_indices = similarity_scores.argsort()[::-1]  # highest to lowest

    # In this example, we just pick the top 5 if available
    top_indices = top_indices[:5]

    # Retrieve recommended rows from the DataFrame
    # If you have fewer item_metadata lines than 5, just adapt accordingly
    recommended_products = baseline_df.iloc[top_indices % len(baseline_df)]

    # Print columns that exist in your baseline CSV
    # Adjust these column names to match your actual data
    # e.g., 'title', 'average_rating', 'price', 'asin', etc.
    columns_to_show = []
    for col_candidate in ["title", "average_rating", "price", "asin"]:
        if col_candidate in baseline_df.columns:
            columns_to_show.append(col_candidate)

    print("Recommended Products (Baseline):")
    print(recommended_products[columns_to_show])


# ------------------------------------------------------------------------------
# Recommendation from Bloom-Filtered Data
# ------------------------------------------------------------------------------
def recommend_from_bloom(language_context, item_metadata):
    print("===== Bloom-Filtered Recommendations =====")

    # Load bloom-filtered data (usually contains review-level info)
    bloom_df = pd.read_csv(BLOOM_FILTERED_PATH)

    # Prepare text list
    text_list = [language_context] + item_metadata

    # Generate embeddings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embeddings = generate_embeddings(text_list, model, tokenizer)

    # Compute similarity
    similarity_scores = torch.mm(embeddings[0:1], embeddings[1:].T).squeeze().cpu().numpy()
    top_indices = similarity_scores.argsort()[::-1]
    top_indices = top_indices[:5]

    # Retrieve recommended rows from the bloom DataFrame
    # If you want to map these reviews back to product-level info, you'd need a common key
    recommended_reviews = bloom_df.iloc[top_indices % len(bloom_df)]

    # Print columns that exist in your bloom-filtered CSV
    # Typically includes 'title' (often the review title), 'rating', etc.
    columns_to_show = []
    for col_candidate in ["title", "rating", "asin", "filtered_text"]:
        if col_candidate in bloom_df.columns:
            columns_to_show.append(col_candidate)

    print("Recommended Reviews (Bloom Filter):")
    print(recommended_reviews[columns_to_show])


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Baseline Metadata Approach
    recommend_from_baseline(language_context, item_metadata)

    print("\n-------------------------------------------\n")

    # Bloom Filtered Approach
    recommend_from_bloom(language_context, item_metadata)
