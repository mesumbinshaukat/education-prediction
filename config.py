import os
import atexit
from dotenv import load_dotenv
from pymongo import MongoClient
import logging

# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton MongoDB client
_mongo_client = None
_db = None

def get_mongo_client():
    """
    Get or create a MongoDB client using a singleton pattern.
    """
    global _mongo_client
    
    if _mongo_client is None:
        try:
            # Connection pooling parameters
            _mongo_client = MongoClient(
                MONGO_URI,
                maxPoolSize=50,                 # Maximum number of connections in the pool
                minPoolSize=10,                 # Minimum number of connections in the pool
                maxIdleTimeMS=60000,            # Max time a connection can be idle (1 minute)
                socketTimeoutMS=30000,          # Socket timeout (30 seconds)
                connectTimeoutMS=5000,          # Connection timeout (5 seconds)
                serverSelectionTimeoutMS=5000,  # Server selection timeout (5 seconds)
                retryWrites=True,               # Enable retryable writes
                w='majority'                    # Write concern
            )
            logger.info("MongoDB client initialized")
            
            # Register cleanup function to run on application exit
            atexit.register(close_mongo_client)
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    return _mongo_client

def close_mongo_client():
    """
    Close the MongoDB client connection
    """
    global _mongo_client
    
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB client connection closed")
            _mongo_client = None
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

def get_db():
    """
    Get the MongoDB database instance
    """
    global _db
    
    if _db is None:
        client = get_mongo_client()
        _db = client['edupredict']
        logger.info("MongoDB database connection established")
    
    return _db
