from django.shortcuts import render
from django.http import JsonResponse
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import boto3

# Load environment variables
load_dotenv()

# Global DataFrame
_df = pd.DataFrame()

def get_df():
    """Getter for the global DataFrame."""
    return _df

def set_df(new_df):
    """Setter for the global DataFrame."""
    global _df
    _df = new_df

def init_embeddings():
    """Initialize the embeddings DataFrame."""
    global _df

    print("Initializing embeddings...")
    
    # Check if file already exists
    if os.path.exists("embeddings.pickle"):
        print("File already exists locally.")
        _df = pd.read_pickle("embeddings.pickle")
        return
    
    print("Downloading embeddings...")
    
    # Initialize boto3 S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-west-1'
    )

    # S3 bucket and file details
    BUCKET_NAME = "findthesub"
    FILE_NAME = "embeddings.pickle"
    LOCAL_FILE_PATH = "embeddings.pickle"

    # Download the embeddings.pickle file from S3
    try:
        s3_client.download_file(BUCKET_NAME, FILE_NAME, LOCAL_FILE_PATH)
        print(f"Downloaded {FILE_NAME} from S3 bucket {BUCKET_NAME} successfully.")
    except Exception as e:
        print(f"Error downloading {FILE_NAME} from S3 bucket {BUCKET_NAME}: {e}")
        return

    # Load the embeddings DataFrame
    try:
        _df = pd.read_pickle(LOCAL_FILE_PATH)
    except FileNotFoundError:
        _df = pd.DataFrame()  # Handle missing file by creating an empty DataFrame
        print(f"Error: {LOCAL_FILE_PATH} not found. Proceeding with an empty DataFrame.")