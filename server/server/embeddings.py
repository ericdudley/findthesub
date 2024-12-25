import os
import threading
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

# Global DataFrame
_df = pd.DataFrame()

# Lock to prevent simultaneous initializations
_init_lock = threading.Lock()
_initialized = False

def get_df():
    """Getter for the global DataFrame."""
    return _df

def set_df(new_df):
    """Setter for the global DataFrame."""
    global _df
    _df = new_df

def init_embeddings():
    """Initialize the embeddings DataFrame in a thread-safe manner."""
    global _df, _initialized

    if _initialized:
        print("Embeddings already initialized.")
        return

    with _init_lock:
        # Double-checked locking
        if _initialized:
            print("Embeddings already initialized by another thread.")
            return

        print("Initializing embeddings...")

        # Check if file already exists locally
        if os.path.exists("embeddings.pickle"):
            print("File already exists locally. Loading embeddings.")
            try:
                _df = pd.read_pickle("embeddings.pickle")
                _initialized = True
                print("Embeddings loaded successfully.")
                return
            except Exception as e:
                print(f"Error loading local embeddings.pickle: {e}")
                # Proceed to download if local load fails

        print("Downloading embeddings from S3...")

        # Initialize boto3 S3 client
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name='us-west-1'
            )
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            return

        # S3 bucket and file details
        BUCKET_NAME = "findthesub"
        FILE_NAME = "embeddings.pickle"
        LOCAL_FILE_PATH = "embeddings.pickle"
        TEMP_FILE_PATH = "embeddings_temp.pickle"

        # Download the embeddings.pickle file from S3 to a temporary file
        try:
            s3_client.download_file(BUCKET_NAME, FILE_NAME, TEMP_FILE_PATH)
            print(f"Downloaded {FILE_NAME} from S3 bucket {BUCKET_NAME} successfully.")
        except Exception as e:
            print(f"Error downloading {FILE_NAME} from S3 bucket {BUCKET_NAME}: {e}")
            return

        # Atomically rename the temporary file to the local file path
        try:
            os.replace(TEMP_FILE_PATH, LOCAL_FILE_PATH)
            print(f"Renamed {TEMP_FILE_PATH} to {LOCAL_FILE_PATH} atomically.")
        except Exception as e:
            print(f"Error renaming {TEMP_FILE_PATH} to {LOCAL_FILE_PATH}: {e}")
            return

        # Load the embeddings DataFrame
        try:
            print("Loading embeddings from local file...")
            _df = pd.read_pickle(LOCAL_FILE_PATH)
            _initialized = True
            print("Embeddings loaded successfully from S3.")
        except Exception as e:
            _df = pd.DataFrame()  # Handle missing or corrupted file by creating an empty DataFrame
            print(f"Error loading {LOCAL_FILE_PATH}: {e}. Proceeding with an empty DataFrame.")

# Example Django view that ensures embeddings are initialized
def embeddings_view(request):
    try:
        init_embeddings()
        data = get_df().to_dict(orient='records')  # Convert DataFrame to list of dicts
        return JsonResponse({"embeddings": data}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)