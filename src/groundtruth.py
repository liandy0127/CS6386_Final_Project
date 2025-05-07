import pandas as pd
import json

def build_ground_truth(review_csv_path, output_json_path):
    """
    Builds ground truth recommendations based on co-purchases by the same user.

    For each item, the ground truth is the set of other items bought by the same users.
    Saves the result as a JSON mapping item_id -> list of co-purchased item_ids.
    """
    # Load processed reviews
    df = pd.read_csv(review_csv_path)

    # Ensure necessary columns exist
    required_cols = {'user_id', 'parent_asin'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Group by user and collect unique items per user
    user_groups = df.groupby('user_id')['parent_asin'].apply(lambda items: list(items.unique()))

    # Build ground truth: for each item, collect co-purchased items
    ground_truth = {}
    for items in user_groups:
        for item in items:
            # Initialize set if first encounter
            co_items = ground_truth.setdefault(item, set())
            # Add all other items in this user's basket
            for co in items:
                if co != item:
                    co_items.add(co)

    # Convert sets to lists for JSON serialization
    ground_truth = {item: list(co_items) for item, co_items in ground_truth.items()}

    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Ground truth saved to '{output_json_path}' ({len(ground_truth)} items)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ground truth from processed reviews data."
    )
    parser.add_argument(
        "--input",
        default="../data/processed/raw_reviews_baseline.csv",
        help="Path to the processed reviews CSV file"
    )
    parser.add_argument(
        "--output",
        default="ground_truth.json",
        help="Path to save the output JSON ground truth mapping"
    )
    args = parser.parse_args()

    build_ground_truth(args.input, args.output)
