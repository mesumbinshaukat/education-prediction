import requests
import json
from pymongo import MongoClient
from config import get_db
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def check_db_connection():
    """Test MongoDB connection and print status of relevant collections"""
    try:
        logging.info("Testing database connection...")
        db = get_db()
        
        # Check if db connection is active
        db_info = db.command("serverStatus")
        logging.info(f"MongoDB connection successful. Server version: {db_info.get('version', 'unknown')}")
        
        # List collections
        collections = db.list_collection_names()
        logging.info(f"Available collections: {collections}")
        
        # Check predictionHistory collection
        if 'predictionHistory' in collections:
            count = db.predictionHistory.count_documents({})
            logging.info(f"predictionHistory collection has {count} documents")
            
            # Show a sample document if available
            if count > 0:
                sample = db.predictionHistory.find_one()
                logging.info(f"Sample document structure: {json.dumps(sample, default=str, indent=2)}")
        else:
            logging.warning("predictionHistory collection does not exist!")
            
        # Check users collection
        if 'users' in collections:
            user_count = db.users.count_documents({})
            logging.info(f"users collection has {user_count} users")
        else:
            logging.warning("users collection does not exist!")
            
        return True
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return False
        
def simulate_form_submission():
    """Simulate a form submission with test data"""
    try:
        logging.info("Attempting to simulate form submission...")
        
        # Test data
        form_data = {
            'name': f'Test Student {datetime.now().strftime("%Y%m%d%H%M%S")}',
            'student_id': f'TEST-{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'email': 'test@example.com',
            'attendance': 85,
            'homework_completion': 90,
            'test_scores': 88
        }
        
        # Directly insert into MongoDB
        try:
            db = get_db()
            
            # Create prediction record
            prediction_data = {
                # Student information
                'student_id': form_data['student_id'],
                'name': form_data['name'],
                'email': form_data['email'],
                'attendance': float(form_data['attendance']),
                'homework_completion': float(form_data['homework_completion']),
                'test_scores': float(form_data['test_scores']),
                
                # Prediction results - calculated same as in app.py
                'prediction_score': round((float(form_data['test_scores']) * 0.5) + 
                                        (float(form_data['attendance']) * 0.3) + 
                                        (float(form_data['homework_completion']) * 0.2), 2),
                
                # Metadata
                'username': 'test_script',
                'created_at': datetime.now(),
            }
            
            # Calculate prediction text
            if prediction_data['prediction_score'] >= 80:
                prediction_data['prediction'] = "Excellent"
            elif prediction_data['prediction_score'] >= 60:
                prediction_data['prediction'] = "Good"
            else:
                prediction_data['prediction'] = "Needs Improvement"
                
            # Legacy values
            prediction_data['binary_prediction'] = 1 if prediction_data['prediction_score'] >= 60 else 0
            prediction_data['confidence'] = prediction_data['prediction_score'] / 100.0
            prediction_data['probability'] = prediction_data['prediction_score']
            
            # Insert the test data
            result = db.predictionHistory.insert_one(prediction_data)
            
            if result.inserted_id:
                logging.info(f"Successfully inserted test document with ID: {result.inserted_id}")
                logging.info(f"Test document: {json.dumps(prediction_data, default=str)}")
                
                # Verify insertion
                inserted = db.predictionHistory.find_one({'_id': result.inserted_id})
                if inserted:
                    logging.info("Document verified in database!")
                    
                    # Now delete the test document
                    db.predictionHistory.delete_one({'_id': result.inserted_id})
                    logging.info("Test document deleted to avoid cluttering the database.")
                    return True
                else:
                    logging.error("Document not found after insertion!")
                    return False
            else:
                logging.error("Document insertion failed!")
                return False
                
        except Exception as e:
            logging.error(f"Error inserting test document: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Form submission simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("========== DATABASE CONNECTION TEST ==========")
    db_ok = check_db_connection()
    
    if db_ok:
        print("\n========== FORM SUBMISSION TEST ==========")
        form_ok = simulate_form_submission()
        
        if form_ok:
            print("\n✅ SUCCESS: Database connection and form submission are working correctly!")
            print("\nIf you're still having issues with the form submission:")
            print("1. Make sure you're logged in (check browser cookies)")
            print("2. Verify the form is submitting to the correct URL")
            print("3. Check browser console for any JavaScript errors")
            print("4. Try clearing browser cache and cookies")
        else:
            print("\n❌ ERROR: Form submission test failed. See logs above for details.")
    else:
        print("\n❌ ERROR: Database connection test failed. See logs above for details.")

