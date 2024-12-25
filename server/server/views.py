from django.shortcuts import render
from django.http import JsonResponse
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import ast
import numpy as np
import boto3
from server.embeddings import get_df

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(txts):
    response = client.embeddings.create(
        input=txts,
        model="text-embedding-3-small"
    )
    return response.data

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_subs_by_similarity(df, txt, n=15):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames if no data is available
    
    embedding = np.array(get_embedding([txt])[0].embedding)
    df['similarities'] = df.Embedding.apply(lambda x: cosine_similarity(x, embedding))
    grouped = df.groupby('Sub')['similarities'].mean().reset_index()
    sorted_subs = grouped.sort_values('similarities', ascending=False)
    top_subs = sorted_subs.head(n)
    bottom_subs = sorted_subs.tail(n).sort_values('similarities', ascending=True)
    return top_subs, bottom_subs

def index(request):
    query = request.GET.get('q')
    if not query:
        return JsonResponse({'error': 'Missing required "q" query param. Example: /api?q=\'Test query\''}, status=400)
    
    top_subs, bottom_subs = get_subs_by_similarity(get_df(), query)
    return JsonResponse({
        'topSubs': top_subs.to_dict(orient='records'),
        'bottomSubs': bottom_subs.to_dict(orient='records')
    })