import os
import sys
import logging
import json
from datetime import datetime
from bson import ObjectId, json_util
from config import get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mongodb_write_test.log')
    ]
)

def test_mongodb_write():
    """Test writing to the predictionHistory collection in MongoDB Atlas"""
    print("\n===== MongoDB Atlas Write Test =====\n")
    
    try:
        # Step 1: Connect to MongoDB Atlas
        logging.info("Connecting to MongoDB Atlas...")
        db = get_db()
        
        # Step 2: Prepare test document
        test_id = f"TEST-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        test_document = {
            'student_id': test_id,
            'name': f'Test Student {test_id}',
            'email': 'test@example.com',
            'attendance': 85.0,
            'homework_completion': 90.0,
            'test_scores': 88.0,
            'prediction_score': 87.5,
            'prediction': 'Excellent',
            'binary_prediction': 1,
            'confidence': 0.875,
            'probability': 87.5,
            'username': 'test_script',
            'created_at': datetime.now(),
            'test_flag': True  # Flag to identify test documents
        }
        
        # Step 3: Write the test document
        logging.info(f"Attempting to write test document with ID: {test_id}")
        result = db.predictionHistory.insert_one(test_document)
        
        # Step 4: Verify the write was successful
        if result.inserted_id:
            logging.info(f"✅ Write SUCCESS: Document inserted with ID: {result.inserted_id}")
            
            # Step 5: Retrieve the document to verify it was saved
            retrieved = db.predictionHistory.find_one({'student_id': test_id})
            
            if retrieved:
                logging.info("✅ Read SUCCESS: Document retrieved successfully")
                logging.info(f"Retrieved document: {json.dumps(json.loads(json_util.dumps(retrieved)), indent=2)}")
                
                # Step 6: List recent documents
                logging.info("Recent documents in predictionHistory collection:")
                recent_docs = list(db.predictionHistory.find().sort('created_at', -1).limit(5))
                
                for i, doc in enumerate(recent_docs):
                    doc_id = doc['_id']
                    student_name = doc.get('name', 'Unknown')
                    created_at = doc.get('created_at', 'Unknown time')
                    
                    logging.info(f"{i+1}. ID: {doc_id}, Name: {student_name}, Created: {created_at}")
                
                # Step 7: Clean up by removing the test document
                db.predictionHistory.delete_one({'student_id': test_id})
                logging.info(f"Removed test document with student_id: {test_id}")
                
                return True
            else:
                logging.error("❌ Read FAILURE: Could not retrieve the document after writing")
                logging.error("This may indicate a replication lag or read consistency issue")
                return False
        else:
            logging.error("❌ Write FAILURE: Document was not inserted")
            return False
            
    except Exception as e:
        logging.error(f"❌ ERROR: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error("This may indicate connection issues, authentication problems, or insufficient permissions")
        return False

def check_permissions():
    """Check if the current user has write permissions on the predictionHistory collection"""
    try:
        db = get_db()
        
        # Check if collection exists
        collections = db.list_collection_names()
        if 'predictionHistory' not in collections:
            logging.warning("⚠️ Collection 'predictionHistory' does not exist yet. It will be created on first write.")
        
        # Check if we can insert and then delete a document (tests write permissions)
        logging.info("Testing write permissions...")
        result = db.predictionHistory.insert_one({
            'test': True,
            'timestamp': datetime.now()
        })
        
        if result.inserted_id:
            logging.info("✅ Write permissions confirmed")
            db.predictionHistory.delete_one({'_id': result.inserted_id})
            return True
        else:
            logging.error("❌ Write permission test failed")
            return False
            
    except Exception as e:
        logging.error(f"❌ Permission check failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # First check permissions
        if check_permissions():
            # Then test writing a full document
            success = test_mongodb_write()
            
            if success:
                print("\n✅ SUCCESS: MongoDB Atlas write test completed successfully")
                print("Your connection to MongoDB Atlas is working properly and you have write permissions")
                print("If form submissions are still not working, the issue is likely with the form submission process")
            else:
                print("\n❌ FAILURE: MongoDB Atlas write test failed")
                print("See logs above for details on what went wrong")
        else:
            print("\n❌ FAILURE: You don't have write permissions on the predictionHistory collection")
            print("Check your MongoDB Atlas user permissions")
    except Exception as e:
        print(f"\n❌ FAILURE: Unexpected error: {str(e)}")

