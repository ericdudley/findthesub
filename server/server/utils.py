import json
import pandas as pd
import requests
import os
from openai import OpenAI

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

def process_results(df, top_n=TOP_N, strategy='mean'):
    if df.empty:
        return df
    if strategy not in ['mean', 'sum', 'median']:
        strategy = 'mean'
    if strategy == 'mean':
        grouped = df.groupby('sub')['distance'].mean().reset_index()
    elif strategy == 'sum':
        grouped = df.groupby('sub')['distance'].sum().reset_index()
    else:  # strategy == 'median'
        grouped = df.groupby('sub')['distance'].median().reset_index()
    return grouped.sort_values('distance', ascending=False).head(top_n)

def first_come_first_serve(df, top_n=TOP_N):
    if df.empty:
        return df
    df_sorted = df.sort_values('distance', ascending=False)
    chosen = []
    used_subs = set()
    for _, row in df_sorted.iterrows():
        if row['sub'] not in used_subs:
            chosen.append(row)
            used_subs.add(row['sub'])
        if len(chosen) >= top_n:
            break
    return pd.DataFrame(chosen)

def query_closest_subs(txt, limit=LIMIT, top_n=TOP_N, strategy='first_come_first_serve'):
    embeddings = get_embedding([txt])
    if not embeddings or not embeddings[0].embedding:
        raise ValueError("Failed to obtain embedding for the input text.")

    payload = {
        "collectionName": COLLECTION_NAME,
        "data": [embeddings[0].embedding],
        "limit": limit,
        "outputFields": ["sub", "distance"]
    }
    resp = requests.post(SEARCH_API_URL, data=json.dumps(payload), headers=SEARCH_HEADERS)
    resp.raise_for_status()
    data = resp.json().get('data', [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # If you want to use one of the standard grouping strategies:
    if strategy in ['mean', 'sum', 'median']:
        return process_results(df, top_n, strategy)

    # If you specifically want to use the "first_come_first_serve" strategy:
    if strategy == 'first_come_first_serve':
        return first_come_first_serve(df, top_n)

    return df