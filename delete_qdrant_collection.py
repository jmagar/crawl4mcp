from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # This might be None if not set
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "crawl4ai_mcp") # Default if not in .env

if not QDRANT_URL:
    print("Error: QDRANT_URL not found in environment variables. Please set it in your .env file.")
    exit(1)

if not COLLECTION_NAME:
    print("Error: QDRANT_COLLECTION not found in environment variables and no default was suitable. Please set it in your .env file.")
    exit(1)

try:
    print(f"Attempting to connect to Qdrant at: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None)

    print(f"Checking for collection: {COLLECTION_NAME}...")
    collections_response = client.get_collections()
    existing_collection_names = [col.name for col in collections_response.collections]

    if COLLECTION_NAME in existing_collection_names:
        print(f"Deleting existing collection: {COLLECTION_NAME}...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist. No need to delete.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure Qdrant is running and accessible, and your .env file has the correct QDRANT_URL and optionally QDRANT_API_KEY.") 