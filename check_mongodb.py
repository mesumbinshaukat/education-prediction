import os
import json
from bson import json_util
from config import get_db
from pymongo import MongoClient
from datetime import datetime

def check_prediction_history():
    try:
        # Connect to the database
        db = get_db()
        
        # Get collection
        collection = db.predictionHistory
        
        # Count documents in the collection
        count = collection.count_documents({})
        print(f"Total records in predictionHistory collection: {count}")
        
        if count > 0:
            # Get a sample document to show schema
            sample = collection.find_one({})
            
            # Prettify the output
            print("\nCollection Schema (based on a sample document):")
            print(json.dumps(json.loads(json_util.dumps(sample)), indent=4))
            
            # Show top 5 most recent entries
            print("\nMost recent entries:")
            recent = list(collection.find().sort("created_at", -1).limit(5))
            
            for i, doc in enumerate(recent):
                print(f"\nRecord {i+1}:")
                print(f"Student: {doc.get('name', 'N/A')}")
                print(f"Student ID: {doc.get('student_id', 'N/A')}")
                print(f"Email: {doc.get('email', 'N/A')}")
                print(f"Prediction: {doc.get('prediction', 'N/A')}")
                print(f"Score: {doc.get('prediction_score', 'N/A')}%")
                print(f"Created: {doc.get('created_at', 'N/A')}")
        else:
            print("\nNo records found in the collection.")
            print("The collection exists but is empty. Form submissions would be stored here.")
            
        # Show collection stats
        stats = db.command("collstats", "predictionHistory")
        print(f"\nCollection size: {stats.get('size', 0) / 1024:.2f} KB")
        print(f"Storage size: {stats.get('storageSize', 0) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("\nPossible issues:")
        print("1. MongoDB connection string may be incorrect")
        print("2. MongoDB server may not be running")
        print("3. Database or collection may not exist yet")

if __name__ == "__main__":
    check_prediction_history()

