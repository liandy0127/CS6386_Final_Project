# NCF Evaluation Pipeline Across Cleaning Levels

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm # Use auto for compatibility

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- PyTorch Dataset Class ---
class ReviewDataset(Dataset):
    """Custom Dataset for review data."""
    def __init__(self, data):
        # Ensure data is a pandas DataFrame with expected columns
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not all(col in data.columns for col in ['user', 'item', 'interaction']):
             raise ValueError("Input DataFrame must contain 'user', 'item', and 'interaction' columns.")

        self.users = torch.tensor(data['user'].values, dtype=torch.long)
        self.items = torch.tensor(data['item'].values, dtype=torch.long)
        self.labels = torch.tensor(data['interaction'].values, dtype=torch.float) # Use float for BCELoss target

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# --- NCF Model Definition ---
class NCF(nn.Module):
    """Neural Collaborative Filtering model."""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        # Ensure num_users and num_items are positive
        if num_users <= 0 or num_items <= 0:
            raise ValueError("Number of users and items must be positive.")

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # Using a simple MLP structure after concatenation
        self.fc1 = nn.Linear(embedding_dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # Output a single value for binary prediction
        self.relu = nn.ReLU() # Activation function

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        # Concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)
        # Pass through MLP layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # Output layer with Sigmoid for binary classification probability
        x = torch.sigmoid(self.fc3(x))
        # Squeeze to remove the last dimension if batch size is 1
        return x.squeeze()

# --- Data Loading and Preparation Function ---
def load_and_prepare_data(path, test_size=0.2, relevance_threshold=4):
    """
    Loads data, cleans column names, encodes IDs, converts to implicit feedback,
    splits data, and creates DataLoaders.
    Returns train_loader, test_loader, user_encoder, item_encoder, num_users, num_items.
    Returns None, None, None, None, None, None if loading/preparation fails.
    """
    print(f"\n--- Loading and preparing data from: {path} ---")
    df = None # Initialize df to None
    try:
        # Read all columns first to handle potential whitespace in headers
        df = pd.read_csv(path)
        # print(f"  Original columns: {df.columns.tolist()}") # Too verbose

        # Clean column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()
        # print(f"  Cleaned columns: {df.columns.tolist()}") # Too verbose

        # Required columns for NCF input
        required_cols = ['user_id', 'parent_asin', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  Available columns after cleaning names: {df.columns.tolist()}")
            print(f"Error: Required columns for NCF not found after cleaning names: {missing_cols}")
            return None, None, None, None, None, None # Indicate failure

        # Select only the required columns and create a copy
        df = df[required_cols].copy()
        print(f"  Successfully loaded and selected columns: {required_cols}")

        # Handle missing values in the required columns BEFORE encoding or converting to implicit
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

        if df.empty:
             print("  Warning: DataFrame is empty after cleaning and dropping NaNs. Cannot proceed with NCF.")
             return None, None, None, None, None, None # Indicate failure

        print(f"  Final number of rows for NCF: {len(df)}")

        # Encode user and item IDs
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        df['user'] = user_encoder.fit_transform(df['user_id'])
        df['item'] = item_encoder.fit_transform(df['parent_asin'])

        num_users = len(user_encoder.classes_)
        num_items = len(item_encoder.classes_)
        print(f"  Encoded {num_users} users and {num_items} items.")

        # Convert ratings to implicit feedback (1 for interaction >= threshold, 0 otherwise)
        # Explicitly handle potential non-numeric ratings before comparison
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(subset=['rating'], inplace=True) # Drop rows where rating is NaN after coercion
        df['interaction'] = (df['rating'] >= relevance_threshold).astype(int)
        print(f"  Converted ratings to implicit feedback (interaction >= {relevance_threshold}).")


        # Split into train and test sets for the NCF model
        # Use the encoded 'user', 'item', and 'interaction' columns
        train_df, test_df = train_test_split(df[['user', 'item', 'interaction']], test_size=test_size, random_state=42)
        print(f"  Split data into {len(train_df)} training and {len(test_df)} testing samples.")

        # Create PyTorch datasets and DataLoaders
        train_dataset = ReviewDataset(train_df)
        test_dataset = ReviewDataset(test_df)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) # Increased batch size
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False) # Increased batch size
        print("  Created DataLoaders.")

        return train_loader, test_loader, user_encoder, item_encoder, num_users, num_items

    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None, None, None, None, None, None # Indicate failure
    except Exception as e:
         print(f"An unexpected error occurred during data loading and preparation: {e}")
         return None, None, None, None, None, None # Indicate failure

# --- Training Function ---
def train_ncf(model, train_loader, epochs, criterion, optimizer, device):
    """Trains the NCF model."""
    print(f"\n--- Training NCF Model for {epochs} epochs ---")
    model.train() # Set model to training mode
    model.to(device) # Move model to device

    for epoch in range(epochs):
        total_loss = 0
        # Wrap train_loader with tqdm for progress bar
        for user, item, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to the same device as the model
            user, item, label = user.to(device), item.to(device), label.to(device)

            optimizer.zero_grad() # Zero the gradients
            output = model(user, item) # Forward pass
            loss = criterion(output, label) # Compute loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights
            total_loss += loss.item() # Accumulate loss

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} finished. Average Loss: {avg_loss:.4f}")

# --- Evaluation Function ---
def evaluate_ncf(model, test_loader, relevance_threshold=0.5, device='cpu'):
    """Evaluates the NCF model and prints metrics."""
    print("\n--- Evaluating NCF Model ---")
    model.eval() # Set model to evaluation mode
    model.to(device) # Move model to device

    y_true = []
    y_pred_scores = [] # Store raw scores for RMSE/MAE
    y_pred_binary = [] # Store binary predictions for Precision/Recall/F1/Hit Rate

    with torch.no_grad(): # Disable gradient computation
        for user, item, label in tqdm(test_loader, desc="Evaluating"):
            # Move data to the same device as the model
            user, item, label = user.to(device), item.to(device), label.to(device)

            output = model(user, item) # Forward pass
            y_true.extend(label.cpu().numpy()) # Move labels back to CPU for numpy conversion
            y_pred_scores.extend(output.cpu().numpy()) # Move scores back to CPU

            # Convert scores to binary predictions based on the relevance threshold
            binary_preds = (output >= relevance_threshold).cpu().numpy().astype(int)
            y_pred_binary.extend(binary_preds)


    # Convert lists to numpy arrays for metric calculations
    y_true = np.array(y_true)
    y_pred_scores = np.array(y_pred_scores)
    y_pred_binary = np.array(y_pred_binary)


    # --- Calculate Metrics ---
    # Precision, Recall, F1-Score (using binary predictions)
    # Handle cases where there are no positive predictions or true positives
    try:
        precision = precision_score(y_true, y_pred_binary)
    except ValueError:
        precision = 0.0 # No positive predictions

    try:
        recall = recall_score(y_true, y_pred_binary)
    except ValueError:
        recall = 0.0 # No true positives

    # Calculate F1 score - handle the case where precision + recall is zero or near zero
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 1e-9 else 0.0


    # RMSE, MAE (using raw prediction scores against binary true labels)
    # Note: RMSE/MAE are more typically used for explicit rating prediction,
    # but can be calculated here against the 0/1 implicit labels.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_scores))
    mae = mean_absolute_error(y_true, y_pred_scores)

    # Hit Rate (Accuracy for binary classification)
    hit_rate = accuracy_score(y_true, y_pred_binary)


    # --- Print Results ---
    print("\n--- Evaluation Results (Hold-out Test Set) ---")
    print(f"Relevance Threshold for Implicit Feedback: >= {relevance_threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"Hit Rate:  {hit_rate:.4f}") # This is binary accuracy
    print("-------------------------------------------")


# --- Main Execution ---
if __name__ == "__main__":
    # Define paths to the different review datasets
    datasets_to_evaluate = [
        ('Noisy Data', "../data/processed/reviews_noisy.csv"),
        ('Text Cleaned Data', "../data/processed/reviews_text_cleaned.csv"),
        ('Missing Handled Data', "../data/processed/reviews_missing_handled.csv"),
        ('Outliers Handled Data', "../data/processed/reviews_outliers_handled.csv"),
        ('Fully Denoised Data', "../data/processed/reviews_denoised_combined.csv"),
    ]

    # --- Hyperparameters ---
    EMBEDDING_DIM = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256 # Used in DataLoader
    TEST_SIZE = 0.2
    RELEVANCE_THRESHOLD = 4 # Rating >= 4 considered an interaction
    PREDICTION_THRESHOLD = 0.5 # Model output >= 0.5 considered a positive prediction

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n*** Starting NCF Evaluation on Different Data Cleaning Levels ***")

    # Loop through each dataset and perform training and evaluation
    for description, reviews_filepath in datasets_to_evaluate:
        print(f"\n{'='*60}") # Separator for clarity
        print(f"Evaluating NCF on: {description}")
        print(f"Dataset path: {reviews_filepath}")
        print(f"{'='*60}")


        # --- Load and Prepare Data ---
        train_loader, test_loader, user_encoder, item_encoder, num_users, num_items = \
            load_and_prepare_data(reviews_filepath, test_size=TEST_SIZE, relevance_threshold=RELEVANCE_THRESHOLD)

        # Check if data was loaded successfully
        if train_loader is None or test_loader is None:
            print(f"Skipping evaluation for {description} due to data loading or preparation error.")
            continue # Skip to the next dataset

        # --- Initialize Model, Loss, and Optimizer ---
        try:
            model = NCF(num_users, num_items, embedding_dim=EMBEDDING_DIM)
            criterion = nn.BCELoss() # Binary Cross-Entropy Loss
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            print("Model, criterion, and optimizer initialized.")
        except ValueError as e:
             print(f"Error initializing NCF model for {description}: {e}. Skipping evaluation.")
             continue # Skip to the next dataset


        # --- Train the Model ---
        train_ncf(model, train_loader, EPOCHS, criterion, optimizer, device)

        # --- Evaluate the Model ---
        evaluate_ncf(model, test_loader, relevance_threshold=PREDICTION_THRESHOLD, device=device)

        print(f"\n--- Finished Evaluation on: {description} ---")
        print(f"{'='*60}\n")

    print("\n*** NCF evaluation script finished. ***")
