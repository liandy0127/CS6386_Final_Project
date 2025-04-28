import pandas as pd
import json

def build_ground_truth(review_csv_path, meta_csv_path, ground_truth_path, output_json_path):
    """
    Builds filtered ground truth based on co-purchases and filters both reviews and metadata to only include items in ground truth.

    Args:
        review_csv_path (str): Path to processed reviews CSV.
        meta_csv_path (str): Path to processed metadata CSV.
        ground_truth_path (str): Path to JSON file mapping each item to its co-purchased items.
        output_json_path (str): Path to write filtered ground truth JSON.
    """
    # Load ground truth mapping
    with open(ground_truth_path, 'r') as gt_file:
        ground_truth = json.load(gt_file)

    # Determine set of items to keep: keys and all co-purchased values
    keep_items = set(ground_truth.keys())
    for co_list in ground_truth.values():
        keep_items.update(co_list)

    # Load reviews and metadata
    reviews = pd.read_csv(review_csv_path)
    meta    = pd.read_csv(meta_csv_path)

    # Filter both DataFrames to only the keep_items
    reviews_filtered = reviews[reviews['parent_asin'].isin(keep_items)].copy()
    meta_filtered    = meta[meta['parent_asin'].isin(keep_items)].copy()

    # Rebuild ground truth mapping to only include filtered items
    filtered_gt = {
        item: [co for co in co_list if co in keep_items]
        for item, co_list in ground_truth.items()
        if item in keep_items
    }

    # Save filtered ground truth JSON
    with open(output_json_path, 'w') as out_file:
        json.dump(filtered_gt, out_file, indent=2)

    # Optionally save filtered CSVs
    reviews_filtered.to_csv('filtered_reviews.csv', index=False)
    meta_filtered.to_csv('filtered_meta.csv', index=False)

    print(f"Filtered reviews: {len(reviews_filtered)} rows")
    print(f"Filtered meta:    {len(meta_filtered)} rows")
    print(f"Filtered ground truth saved to '{output_json_path}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter reviews, metadata, and ground truth to the co-purchase subset."
    )
    parser.add_argument(
        "--reviews", default="../../data/processed/processed_reviews.csv",
        help="Path to reviews CSV"
    )
    parser.add_argument(
        "--meta", default="../../data/processed/processed_meta.csv",
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--ground_truth", default="../../data/processed/ground_truth.json",
        help="Path to input ground truth JSON"
    )
    parser.add_argument(
        "--output", default="../../data/processed/filtered_ground_truth.json",
        help="Path to output filtered ground truth JSON"
    )
    args = parser.parse_args()

    build_ground_truth(
        args.reviews,
        args.meta,
        args.ground_truth,
        args.output
    )
