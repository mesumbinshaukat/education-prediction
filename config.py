import os
import logging
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mongodb_connection.log')
    ]
)

# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Store MongoDB client as a global variable to avoid multiple connections
_mongodb_client = None

def get_db():
    global _mongodb_client
    
    try:
        # If we already have a connection, reuse it
        if _mongodb_client is not None:
            return _mongodb_client['edupredict']
            
        # Log connection attempt
        logging.info(f"Attempting to connect to MongoDB Atlas...")
        
        # Configure client with shorter timeout for faster error detection
        _mongodb_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000,
            socketTimeoutMS=10000
        )
        
        # Force a connection to verify it works
        # This will raise an exception if it fails
        _mongodb_client.admin.command('ping')
        
        # Get database names to verify access
        db_names = _mongodb_client.list_database_names()
        logging.info(f"Successfully connected to MongoDB Atlas. Available databases: {db_names}")
        
        return _mongodb_client['edupredict']
        
    except ServerSelectionTimeoutError as e:
        logging.error(f"MongoDB Atlas connection error - Server selection timeout: {e}")
        logging.error("This usually indicates network connectivity issues or IP whitelist restrictions.")
        logging.error("Make sure your IP address is whitelisted in MongoDB Atlas.")
        raise
        
    except ConnectionFailure as e:
        logging.error(f"MongoDB Atlas connection error - Could not connect: {e}")
        logging.error("Check your network connection and MongoDB Atlas status.")
        raise
        
    except OperationFailure as e:
        if "Authentication failed" in str(e):
            logging.error(f"MongoDB Atlas authentication error: {e}")
            logging.error("Check your username and password in the connection string.")
        else:
            logging.error(f"MongoDB Atlas operation error: {e}")
            logging.error("Check your MongoDB Atlas user permissions.")
        raise
        
    except Exception as e:
        logging.error(f"Unexpected error connecting to MongoDB Atlas: {e}")
        logging.error("Check your MONGO_URI in the .env file.")
        raise
