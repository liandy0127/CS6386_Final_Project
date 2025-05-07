import streamlit as st
import os
import json
from blair import recommend_items


# --- Page config ---
st.set_page_config(page_title="Product Recommender", layout="wide")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("We clean data using the following method:")
#    st.subheader("Data Cleaning Statistics")
    stats_files = [
        ("Missing Value Handling", "../data/processed/missing_stats.json"),
        ("Outlier Removal", "../data/processed/outlier_stats.json"),
        ("Text Normalization", "../data/processed/clean_text_stats.json")
    ]
    for section_title, file_path in stats_files:
        st.markdown(f"#### {section_title}")
        if not os.path.exists(file_path):
            st.warning(f"Statistics file not found: `{file_path}`")
            continue
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            st.markdown("```json\n" + json.dumps(data, indent=2) + "\n```")
        except Exception as e:
            st.warning(f"Could not load or parse `{file_path}`: {e}")

# --- Input Section ---
with col1:
    st.title("Review-Based Product Recommender")

    subcol1, subcol2 = st.columns([3, 1])
    with subcol1:
        st.subheader("Do you need to standardize table headers?")
    with subcol2:
        st.checkbox("Yes", key="standardize_checkbox")

    st.markdown("Enter a product name or ID to get recommended items.")
    product_input = st.text_input("🔎 Enter Product Name or ID:")
    top_k = st.slider("How many recommendations?", min_value=1, max_value=20, value=10)
    if st.button("Recommend"):
        if product_input.strip() == "":
            st.warning("Please enter a product name or ID.")
        else:
            recommended_products = recommend_items(product_input, top_k)
            st.success(f"Top-{top_k} recommendations for '{product_input}':")
            for i, prod in enumerate(recommended_products, 1):
                title = prod.get('title', '').strip()
                if not title or title == "[Missing Value]":
                    continue
                st.markdown(f"**{i}. {title}**")