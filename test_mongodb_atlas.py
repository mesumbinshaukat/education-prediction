import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mongodb_atlas_connection():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB URI
        MONGO_URI = os.getenv("MONGO_URI")
        logger.info(f"Using MongoDB URI (masked): {MONGO_URI[:15]}...{MONGO_URI[-15:]}")
        
        # Try to connect with explicit timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ping')
        logger.info("Connected to MongoDB Atlas successfully!")
        
        # Get the database
        db = client['edupredict']
        
        # List collections
        collections = db.list_collection_names()
        logger.info(f"Available collections: {collections}")
        
        # Check for predictionHistory collection
        if 'predictionHistory' in collections:
            count = db.predictionHistory.count_documents({})
            logger.info(f"predictionHistory collection has {count} documents")
        else:
            logger.warning("predictionHistory collection does not exist!")
        
        # Test data
        test_data = {
            'test_id': f'TEST_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'name': 'Test Student',
            'timestamp': datetime.now(),
            'test_type': 'atlas_connection_test'
        }
        
        # Try to insert
        logger.info("Attempting to insert test document...")
        result = db.predictionHistory.insert_one(test_data)
        logger.info(f"Test document inserted with ID: {result.inserted_id}")
        
        # Verify insertion
        logger.info("Verifying document was inserted...")
        inserted = db.predictionHistory.find_one({'_id': result.inserted_id})
        if inserted:
            logger.info("Test document verified in database!")
            
            # Clean up test data
            db.predictionHistory.delete_one({'_id': result.inserted_id})
            logger.info("Test document cleaned up")
            
            print("\n✅ SUCCESS: MongoDB Atlas connection and insertion test passed!")
            return True
        else:
            logger.error("Could not verify inserted document!")
            print("\n❌ ERROR: Could not verify document after insertion!")
            return False
            
    except Exception as e:
        logger.error(f"MongoDB Atlas test failed: {str(e)}")
        print(f"\n❌ ERROR: MongoDB Atlas test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n===== TESTING MONGODB ATLAS CONNECTION =====\n")
    test_mongodb_atlas_connection()

