import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_metadata(df):
    """
    Impute missing metadata:
      - price: numeric median
      - main_category: most frequent
      - store: most frequent
    """
    df = df.copy()
    # Price
    if 'price' in df.columns:
        imp_price = SimpleImputer(strategy='median')
        df['price'] = imp_price.fit_transform(df[['price']])
    # Categorical
    for col in ('main_category', 'store'):
        if col in df.columns:
            imp_cat = SimpleImputer(strategy='most_frequent')
            df[col] = imp_cat.fit_transform(df[[col]])
    return df

def handle_missing_reviews(df):
    """
    Impute missing reviews fields:
      - rating: median
      - text: placeholder
      - user_id / parent_asin: drop rows if missing
    """
    df = df.copy()
    # Rating
    if 'rating' in df.columns:
        imp_rating = SimpleImputer(strategy='median')
        df['rating'] = imp_rating.fit_transform(df[['rating']])
    # Text
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('no review provided')
    # Drop critical missing
    df.dropna(subset=['user_id', 'parent_asin'], inplace=True)
    return df
