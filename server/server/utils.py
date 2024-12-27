import json
import pandas as pd
import requests
import os
from openai import OpenAI
import numpy as np

SEARCH_API_URL = f"{os.environ.get('ZILLIZ_HOST')}/v2/vectordb/entities/search"
SEARCH_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('ZILLIZ_API_KEY')}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

COLLECTION_NAME = "FindTheSub"
LIMIT = 10
TOP_N = 5

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(txts):
    response = client.embeddings.create(
        input=txts,
        model="text-embedding-3-small"
    )
    return response.data

def process_results(df, top_n=1000):
    if df.empty:
        return df
    
    # Create a raw rank column (lower index => higher rank)
    df['raw_rank'] = df.index

    # Convert raw rank to normalized rank in [0, 1]
    df['rank_norm'] = (
        (df['raw_rank'] - df['raw_rank'].min()) / 
        (df['raw_rank'].max() - df['raw_rank'].min()) 
        if len(df) > 1 else 0
    )

    # Group and compute raw statistics + count
    grouped = df.groupby('sub').agg(
        mean_distance=('distance', 'mean'),
        max_distance=('distance', 'max'),
        avg_rank_norm=('rank_norm', 'mean'),
        count=('distance', 'size')
    ).reset_index()

    # Normalize columns in [0, 1]
    for col in ['mean_distance', 'max_distance', 'avg_rank_norm']:
        col_min = grouped[col].min()
        col_max = grouped[col].max()
        if col_max != col_min:
            grouped[col] = (grouped[col] - col_min) / (col_max - col_min)
        else:
            grouped[col] = 0

    # Alternatively, use log scaling to reduce the advantage of very large subreddits
    grouped['count_log'] = np.log1p(grouped['count'])  # log(1 + count)

    alpha = 0.3  # weight for max_distance
    beta = 0.3   # weight for mean_distance
    gamma = 0.2  # weight for avg_rank_norm
    delta = 0.2  # weight for count (or count_log)

    grouped['score'] = (
        alpha * grouped['max_distance'] +
        beta * grouped['mean_distance'] +
        gamma * (1 - grouped['avg_rank_norm']) +
        delta * grouped['count_log']
    )

    # Sort by the score in descending order and return top_n
    results = grouped.sort_values('score', ascending=False).head(top_n)
    return results

def query_closest_subs(txt, limit=LIMIT, top_n=TOP_N):
    # Get embeddings from OpenAI
    embeddings = get_embedding([txt])
    if not embeddings or not embeddings[0].embedding:
        raise ValueError("Failed to obtain embedding for the input text.")
    
    # Send query to Zilliz
    payload = {
        "collectionName": COLLECTION_NAME,
        "data": [embeddings[0].embedding],
        "limit": limit,
        "outputFields": ["sub","distance","title", "post_id"],
        "annsField": "vector"
    }
    resp = requests.post(SEARCH_API_URL, data=json.dumps(payload), headers=SEARCH_HEADERS)
    resp.raise_for_status()
    data = resp.json().get('data', [])
    
    # Convert results to DataFrame
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    
    # Apply the new aggregation strategy
    return {"posts": df, "results": process_results(df, top_n=top_n)}