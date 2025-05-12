        if session_id in self.conversations:
            self.conversations[session_id] = []
        
        return {
            'response': "Conversation history cleared.",
            'type': 'success',
            'success': True
        }
    
    def provide_feedback(self, session_id, message_id, feedback, is_authenticated=False):
        """Store feedback for a message"""
        if message_id not in self.feedbacks:
            self.feedbacks[message_id] = {
                'feedback': feedback,
                'session_id': session_id,
                'timestamp': datetime.now(),
                'is_authenticated': is_authenticated
            }
            
        return {
            'response': "Thank you for your feedback!",
            'type': 'success',
            'success': True
        }
    
    def get_chatbot_stats(self):
        """Get usage statistics for the chatbot"""
        total_conversations = len(self.conversations)
        total_messages = sum(len(messages) for messages in self.conversations.values())
        total_feedbacks = len(self.feedbacks)
        
        return {
            'success': True,
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'total_feedbacks': total_feedbacks
        }

# Singleton chatbot instance
_chatbot = None

def get_chatbot():
    """Get or create the chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = Chatbot()
    return _chatbot
"""
        with open(chatbot_path, 'w') as f:
            f.write(chatbot_content)
        print_success("Created utils/chatbot.py")
    else:
        print_info("utils/chatbot.py already exists, skipping")
    
    # Create __init__.py to make utils a proper package
    init_path = utils_dir / '__init__.py'
    if not init_path.exists():
        with open(init_path, 'w') as f:
            f.write("# Utils package\n")
        print_success("Created utils/__init__.py")
    else:
        print_info("utils/__init__.py already exists, skipping")
    
    return True

def install_requirements():
    """Install required Python packages"""
    print_header("Installing Required Packages")
    
    # Create requirements file if it doesn't exist
    req_path = Path('requirements.txt')
    if not req_path.exists():
        requirements = """Flask==2.0.1
Flask-SocketIO==5.1.1
pymongo==4.0.1
numpy==1.21.4
pandas==1.3.4
scikit-learn==1.0.1
werkzeug==2.0.2
python-socketio==5.4.0
python-engineio==4.3.0
eventlet==0.33.0
fpdf==1.7.2
"""
        with open(req_path, 'w') as f:
            f.write(requirements)
        print_success("Created requirements.txt")
    else:
        print_info("requirements.txt already exists, using existing

#!/usr/bin/env python
import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("setup")

# Define required directories
REQUIRED_DIRS = [
    "models",
    "data",
    "knowledge",
    "logs",
    "static",
    "templates",
    "utils",
    "middleware",
    "streaming"
]

def check_python_version():
    """Check if Python version is compatible"""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        logger.error(f"Python {min_version[0]}.{min_version[1]} or higher is required.")
        logger.error(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    else:
        logger.info(f"Python version check passed: {current_version[0]}.{current_version[1]}")
        return True

def create_directories():
    """Create all required directories if they don't exist"""
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    
    logger.info("Creating required directories...")
    for directory in REQUIRED_DIRS:
        dir_path = project_root / directory
        if not dir_path.exists():
            logger.info(f"Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")
    
    # Create subdirectories
    (project_root / "knowledge" / "conversations").mkdir(exist_ok=True)
    (project_root / "logs" / "chatbot").mkdir(exist_ok=True)
    
    logger.info("Directory structure created successfully!")
    return True

def verify_mongodb_connection():
    """Verify connection to MongoDB"""
    try:
        from pymongo import MongoClient
        
        # Try to connect to MongoDB
        logger.info("Attempting to connect to MongoDB...")
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        
        # The ismaster command is cheap and does not require auth
        client.admin.command('ismaster')
        
        logger.info("MongoDB connection successful!")
        
        # Check if our database exists or create it
        db_name = "education_prediction"
        db = client[db_name]
        
        # List of all collections we need
        required_collections = [
            "users", 
            "predictionHistory", 
            "chatbotConversations"
        ]
        
        # Check existing collections
        existing_collections = db.list_collection_names()
        
        # Create any missing collections with a sample document
        for collection in required_collections:
            if collection not in existing_collections:
                logger.info(f"Creating collection: {collection}")
                # Add a sample document to initialize the collection
                if collection == "users":
                    db[collection].insert_one({
                        "username": "admin",
                        "email": "admin@example.com",
                        "password": "pbkdf2:sha256:150000$GHDXkUYD$9e12d8ab45d9b2c29a3a73919e3c2c70e14d2a971a72c0d0deecddf0c77a77e5",  # Password: admin123
                        "created_at": datetime.now(),
                        "is_test_account": True
                    })
                elif collection == "predictionHistory":
                    db[collection].insert_one({
                        "student_id": "S12345",
                        "name": "Test Student",
                        "email": "test@example.com",
                        "attendance": 80,
                        "homework_completion": 85,
                        "test_scores": 75,
                        "prediction": "Good",
                        "prediction_score": 78.5,
                        "binary_prediction": 1,
                        "confidence": 78.5,
                        "probability": 0.785,
                        "username": "admin",
                        "created_at": datetime.now(),
                        "is_test_data": True
                    })
                elif collection == "chatbotConversations":
                    db[collection].insert_one({
                        "session_id": "test-session-123",
                        "user_query": "How does the prediction system work?",
                        "bot_response": "The prediction system evaluates student performance based on attendance, homework completion, and test scores.",
                        "is_authenticated": False,
                        "username": None,
                        "timestamp": datetime.now(),
                        "is_test_data": True
                    })
            else:
                logger.info(f"Collection already exists: {collection}")
                
        return True
    
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        logger.error("Please make sure MongoDB is installed and running")
        return False

def create_sample_model():
    """Create and save a sample prediction model"""
    try:
        logger.info("Creating sample ML model...")
        project_root = Path(os.path.dirname(os.path.abspath(__file__)))
        models_dir = project_root / "models"
        
        # Create a simple logistic regression model
        # Sample data: attendance, homework_completion, test_scores
        X = np.array([
            [95, 90, 88],  # Excellent
            [85, 82, 80],  # Good
            [75, 70, 72],  # Good
            [60, 55, 50],  # Needs Improvement
            [90, 88, 85],  # Excellent
            [80, 75, 73],  # Good
            [65, 60, 62],  # Needs Improvement
            [50, 45, 40],  # Needs Improvement
            [92, 90, 95],  # Excellent
            [88, 85, 87]   # Good
        ])
        
        # Labels: 1 for Good or Excellent, 0 for Needs Improvement
        y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and fit model
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # Save the model and scaler
        with open(models_dir / "best_model.pkl", 'wb') as f:
            pickle.dump(model, f)
            
        with open(models_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info("Sample ML model created and saved successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error creating sample model: {e}")
        return False

def create_config_file():
    """Create a sample .env file if it doesn't exist"""
    try:
        project_root = Path(os.path.dirname(os.path.abspath(__file__)))
        env_file = project_root / ".env"
        
        if not env_file.exists():
            logger.info("Creating .env configuration file...")
            with open(env_file, 'w') as f:
                f.write("MONGO_URI=mongodb://localhost:27017/education_prediction\n")
                f.write("SECRET_KEY=your_secret_key_here\n")
                f.write("HUGGINGFACE_API_KEY=your_huggingface_api_key_here\n")
            
            logger.info(".env file created successfully!")
        else:
            logger.info(".env file already exists.")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating config file: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Education Prediction System setup...")
    
    # Steps to perform
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Verifying MongoDB connection", verify_mongodb_connection),
        ("Creating sample ML model", create_sample_model),
        ("Creating configuration file", create_config_file)
    ]
    
    # Track success
    all_successful = True
    
    for step_name, step_func in steps:
        logger.info(f"STEP: {step_name}")
        try:
            success = step_func()
            if not success:
                all_successful = False
                logger.warning(f"Step '{step_name}' did not complete successfully.")
        except Exception as e:
            all_successful = False
            logger.error(f"Error during '{step_name}': {e}")
    
    if all_successful:
        logger.info("✓ Setup completed successfully!")
        logger.info("✓ You can now run the application with: python app.py")
    else:
        logger.warning("⚠ Setup completed with warnings or errors.")
        logger.warning("⚠ Please check the logs and fix any issues before running the application.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

