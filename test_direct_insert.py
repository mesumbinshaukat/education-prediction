import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
import json
from bson import ObjectId

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder for MongoDB types
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(MongoJSONEncoder, self).default(obj)

def test_direct_insert():
    try:
        # Load environment variables
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_URI")
        logger.info(f"MongoDB URI (masked): {MONGO_URI[:15]}...{MONGO_URI[-15:]}")
        
        # Connect to MongoDB Atlas
        logger.info("Connecting to MongoDB Atlas...")
        client = MongoClient(MONGO_URI)
        
        # Test connection
        client.admin.command('ping')
        logger.info("Connection to MongoDB Atlas is working!")
        
        # Access database
        db = client['edupredict']
        
        # List collections
        collections = db.list_collection_names()
        logger.info(f"Available collections: {collections}")
        
        # Test data with unique ID
        test_id = f'TEST-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        test_data = {
            'student_id': test_id,
            'name': 'Test Student',
            'email': 'test@example.com',
            'attendance': 85.0,
            'homework_completion': 90.0,
            'test_scores': 88.0,
            'prediction': 'Good',
            'prediction_score': 87.5,
            'binary_prediction': 1,
            'confidence': 0.875,
            'probability': 87.5,
            'username': 'test_script',
            'created_at': datetime.now(),
            'test_type': 'direct_insert_test'
        }
        
        # Direct insert without transaction
        logger.info("Attempting direct insert...")
        result = db.predictionHistory.insert_one(test_data)
        
        if result.inserted_id:
            logger.info(f"Success! Document inserted with ID: {result.inserted_id}")
            
            # Verify the insert
            doc = db.predictionHistory.find_one({'_id': result.inserted_id})
            if doc:
                logger.info("Document verified in database")
                
                # Convert to a serializable format
                doc_serializable = json.loads(
                    json.dumps(doc, cls=MongoJSONEncoder)
                )
                logger.info(f"Retrieved document: {json.dumps(doc_serializable, indent=2)}")
                
                # Count documents
                count = db.predictionHistory.count_documents({})
                logger.info(f"Total documents in collection: {count}")
                
                # Clean up test data
                db.predictionHistory.delete_one({'_id': result.inserted_id})
                logger.info("Test document cleaned up")
                
                # Verify document is gone
                doc_after_delete = db.predictionHistory.find_one({'_id': result.inserted_id})
                if doc_after_delete:
                    logger.warning("Document still exists after deletion!")
                else:
                    logger.info("Document deletion verified")
                
                return True
            else:
                logger.error("Document not found after insertion - verification failed")
                return False
        else:
            logger.error("Insert failed - no inserted_id returned")
            return False
            
    except Exception as e:
        logger.error(f"Error during direct insert test: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

def test_without_transactions():
    """Simpler test without transactions to isolate the issue"""
    try:
        # Load environment variables
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_URI")
        
        # Connect directly to MongoDB Atlas
        client = MongoClient(MONGO_URI)
        db = client['edupredict']
        
        # Create test document with current timestamp for uniqueness
        test_id = f'SIMPLE-TEST-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        test_data = {
            'test_id': test_id,
            'timestamp': datetime.now(),
            'simple_test': True
        }
        
        # Insert directly
        print(f"Inserting simple test document with ID: {test_id}")
        result = db.predictionHistory.insert_one(test_data)
        print(f"Inserted ID: {result.inserted_id}")
        
        # Verify
        doc = db.predictionHistory.find_one({'test_id': test_id})
        if doc:
            print("✅ Simple test document verified")
            
            # Clean up
            db.predictionHistory.delete_one({'_id': result.inserted_id})
            print("Simple test document cleaned up")
            return True
        else:
            print("❌ Simple test failed - document not found after insertion")
            return False
            
    except Exception as e:
        print(f"❌ Simple test error: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Testing Direct MongoDB Insert ===\n")
    
    # First try the simpler test without transactions
    print("\n=== Testing Simple Insert ===")
    simple_success = test_without_transactions()
    print(f"Simple test {'succeeded' if simple_success else 'failed'}\n")
    
    # Then try the more complex test with document verification
    success = test_direct_insert()
    print(f"\nMain test {'succeeded' if success else 'failed'}")
    
    # Overall result
    if simple_success and success:
        print("\n✅ OVERALL RESULT: All MongoDB tests succeeded. Direct insertion is working correctly.")
        print("If form submissions still aren't working, the issue is in the form submission process, not MongoDB.")
    else:
        print("\n❌ OVERALL RESULT: MongoDB tests failed. There appears to be an issue with the database connection or permissions.")

