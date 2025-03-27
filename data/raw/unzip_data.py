import gzip
import shutil
import os

review_input_path = 'Video_Games.jsonl.gz'  # Path to the gzipped reviews file
meta_input_path = 'meta_Video_Games.jsonl.gz'  # Path to the gzipped meta file
review_output_path = 'Video_Games.jsonl'  # Path where the unzipped review file will be saved
meta_output_path = 'meta_Video_Games.jsonl'  # Path where the unzipped meta file will be saved

if not os.path.exists(review_input_path):
    print(f"Error: The file {review_input_path} does not exist.")
else:
    with gzip.open(review_input_path, 'rb') as f_in:
        with open(review_output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Review file successfully unzipped to {review_output_path}")

if not os.path.exists(meta_input_path):
    print(f"Error: The file {meta_input_path} does not exist.")
else:
    with gzip.open(meta_input_path, 'rb') as f_in:
        with open(meta_output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Meta file successfully unzipped to {meta_output_path}")
