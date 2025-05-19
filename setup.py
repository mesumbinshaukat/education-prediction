import os
import sys
import shutil
import logging
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        sys.exit(1)
    logger.info(f"Python version {sys.version.split()[0]} is compatible")

def create_directory_structure():
    """Create necessary directories if they don't exist"""
    directories = [
        'models',
        'data',
        'knowledge',
        'knowledge/conversations',
        'logs',
        'logs/chatbot',
        'static',
        'static/js',
        'static/css',
        'static/images',
        'templates',
        'templates/components',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_mongodb_connection():
    """Check MongoDB connection and create initial collections"""
    try:
        load_dotenv()
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in .env file")
            
        client = MongoClient(mongo_uri)
        db = client['edupredict']
        
        # Create collections if they don't exist
        collections = ['users', 'predictionHistory', 'chatHistory']
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
        
        # Create indexes
        db.users.create_index('username', unique=True)
        db.predictionHistory.create_index([('username', 1), ('created_at', -1)])
        db.chatHistory.create_index([('session_id', 1), ('timestamp', -1)])
        
        logger.info("MongoDB connection successful")
        return True
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return False

def create_sample_model():
    """Create a sample ML model for testing"""
    try:
        # Generate sample data
        np.random.seed(42)
        X = np.random.rand(100, 3) * 100  # 100 samples, 3 features
        y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 >= 60).astype(int)
        
        # Create and train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # Save model and scaler
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info("Sample ML model created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample model: {str(e)}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path('.env')
    if not env_path.exists():
        env_content = """# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/edupredict

# Flask Configuration
SECRET_KEY=your-secret-key-here

# Optional: HuggingFace API Key for enhanced chatbot
# HUGGINGFACE_API_KEY=your-api-key-here
"""
        env_path.write_text(env_content)
        logger.info("Created .env file with default configuration")
        logger.warning("Please update the SECRET_KEY in .env file before deployment")

def cleanup_unused_files():
    """Remove unused files and directories"""
    to_remove = [
        'train_model.py',
        'routes',
        'middleware',
        'streaming',
        'myenv'
    ]
    
    for item in to_remove:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            logger.info(f"Removed unused item: {item}")

def main():
    """Main setup function"""
    logger.info("Starting project setup...")
    
    # Check Python version
    check_python_version()
    
    # Create directory structure
    create_directory_structure()
    
    # Check MongoDB connection
    if not check_mongodb_connection():
        logger.error("MongoDB setup failed. Please check your MongoDB installation and connection settings.")
        sys.exit(1)
    
    # Create sample model
    create_sample_model()
    
    # Create .env file
    create_env_file()
    
    # Cleanup unused files
    cleanup_unused_files()
    
    logger.info("""
Setup completed successfully! Next steps:
1. Update the SECRET_KEY in .env file
2. (Optional) Add your HuggingFace API key for enhanced chatbot
3. Run 'python app.py' to start the application
    """)

if __name__ == '__main__':
    main()
