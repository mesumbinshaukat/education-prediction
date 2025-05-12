# Education Prediction System

The Education Prediction System is an intelligent platform that analyzes student performance metrics to predict educational outcomes. By leveraging machine learning techniques, the system evaluates attendance records, homework completion rates, and test scores to classify students into performance categories and provide actionable insights for educators.

<!-- ![Education Prediction System](https://via.placeholder.com/800x400?text=Education+Prediction+System) -->

## 🚀 Key Features

- **Personalized Student Performance Prediction**: Generate predictions based on attendance, homework completion, and test scores
- **User Authentication**: Secure login system with personalized dashboards
- **Prediction History Tracking**: View and analyze past predictions with detailed metrics
- **PDF Report Generation**: Create professional PDF reports for student assessments
- **Interactive AI Chatbot**: Dual-mode chatbot system:
  - Public chatbot for general inquiries
  - Enhanced authenticated chatbot with additional capabilities
- **Real-time Updates**: WebSocket integration for instant message exchange
- **Machine Learning Integration**: Continuously improving model that learns from new prediction data
- **Responsive Design**: Modern UI that works across devices of all sizes

## 🔧 Tech Stack

### Backend
- **Flask**: Web framework for the application
- **PyMongo**: MongoDB integration
- **SocketIO**: Real-time WebSocket communication
- **Scikit-learn**: Machine learning models for prediction
- **LangChain**: Framework for creating chatbot capabilities
- **HuggingFace**: Models and embeddings for natural language processing
- **FPDF**: PDF generation for reports

### Frontend
- **Bootstrap**: Responsive UI framework
- **JavaScript/jQuery**: Frontend interactivity
- **WebSockets**: Real-time chat functionality

### Database
- **MongoDB**: NoSQL database for storing user data and predictions

### Machine Learning
- **Scikit-learn**: Logistic regression model for predictions
- **Transformers**: NLP capabilities for the chatbot
- **FAISS**: Vector database for semantic search

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- MongoDB (local or remote instance)
- pip (Python package manager)
- Git (for cloning the repository)

## 🔌 Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/mesumbinshaukat/education-prediction
cd education-prediction
```

### Step 2: Set up a virtual environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set up MongoDB
- Install MongoDB locally or use a cloud service like MongoDB Atlas
- Create a database named `education_prediction`

### Step 5: Automated Setup (Recommended)

The project includes a setup script that automates the initialization process:

```bash
python setup.py
```

This script will:
- Verify Python version compatibility
- Create all required directories
- Check MongoDB connection and create initial collections
- Generate a sample ML model for testing
- Create a default configuration file

### Step 6: Manual Configuration (Alternative)

If you prefer manual setup, create a `.env` file in the root directory with the following content:
```
MONGO_URI=mongodb://localhost:27017/education_prediction
SECRET_KEY=your_secret_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here (optional for enhanced chatbot)
```

And initialize the directory structure:
```bash
mkdir -p models data knowledge logs static templates utils middleware streaming
```

## 🚀 Running the Application

### Start the application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Testing the Installation

After running the setup script, you can verify your installation:

1. **Check the MongoDB Connection:**
   - The setup script creates a test user with the credentials:
     - Username: `admin`
     - Password: `admin123`
   - You can log in with these credentials to verify authentication works

2. **Verify Prediction Functionality:**
   - Enter student metrics on the Dashboard page
   - The system should generate a prediction using the sample model

3. **Test the Chatbot:**
   - Try both the public and authenticated chatbot options
   - The public chatbot is available without login
   - The authenticated chatbot requires login and offers enhanced features

## 📱 Usage Guide

### 1. Home Page
The landing page provides an overview of the system's capabilities and access to other features.

### 2. Registration and Login
- New users can register by providing username, email, and password
- Existing users can log in using their credentials

### 3. Making Predictions
1. Navigate to the Dashboard
2. Enter student details:
   - Name
   - Student ID
   - Email
   - Attendance percentage (0-100)
   - Homework completion rate (0-100)
   - Test scores (0-100)
3. Click "Make Prediction"
4. View the prediction result showing:
   - Performance category (Excellent, Good, or Needs Improvement)
   - Probability score
   - Performance breakdown

### 4. Viewing Prediction History
1. Navigate to the History page
2. View all past predictions in a tabular format
3. Filter by date, student name, or performance category
4. Generate PDF reports for any prediction

### 5. Chatbot Interaction
- **Public Chatbot**: Available without login for general inquiries
- **Authenticated Chatbot**: Enhanced capabilities for logged-in users

## 🤖 Chatbot Functionality

The system features a dual-mode chatbot:

### Public Chatbot
- Available to all visitors
- Can answer general questions about the system
- Provides basic information about educational metrics
- Accessible via the "Chatbot" link in the navbar for non-logged in users

### Authenticated Chatbot
- Available only to logged-in users
- All capabilities of the public chatbot plus:
- Can make predictions directly via chat
- Can access user's prediction history
- Can generate reports
- Provides personalized recommendations
- Accessible via the "Chatbot" link in the navbar after login

### Example Chatbot Commands

```
# General inquiry
How does the prediction system work?

# Making a prediction (authenticated)
Make a prediction for student John with 85% attendance, 90% homework completion, and 78% test scores

# Viewing history (authenticated)
Show my prediction history for the last month
```

## ⚙️ Configuration

### Database Configuration
The system uses MongoDB. Configure your connection in `config.py`:

```python
def get_db():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.education_prediction
    return db
```

### Model Configuration
ML models are stored in the `models` directory:
- `best_model.pkl`: The main prediction model
- `scaler.pkl`: StandardScaler for preprocessing input data

Models are automatically retrained as new prediction data is collected.

### Chatbot Configuration
Chatbot settings can be modified in `utils/chatbot.py`, including:
- Response patterns
- Security settings
- Knowledge base update frequency

### Middleware Configuration
The middleware directory contains components that process requests and responses:
- **Authentication**: Controls user access to protected routes
- **Rate Limiting**: Prevents API abuse by limiting request frequency
- **Error Handling**: Standardizes error responses across the application

### Streaming Configuration
The streaming directory handles real-time data communication:
- **WebSocket Events**: Manages real-time chat functionality
- **Data Broadcasts**: Handles pushing updates to connected clients
- **Connection Management**: Maintains WebSocket session state

## 🛠️ Development Setup

### Directory Structure
```
education-prediction/
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── setup.py                # Automated setup script
├── requirements.txt        # Package dependencies
├── .env                    # Environment configuration
├── README.md               # Project documentation
├── models/                 # ML model files
│   ├── best_model.pkl      # Trained prediction model
│   └── scaler.pkl          # Data preprocessing scaler
├── data/                   # Data sources and files
├── knowledge/              # Chatbot knowledge base
│   └── conversations/      # Stored chat dialogues
├── logs/                   # Application logs
│   └── chatbot/            # Chatbot specific logs
├── static/                 # CSS, JS, images
│   ├── style.css           # Main stylesheet
│   └── js/                 # JavaScript files
├── templates/              # HTML templates
│   ├── components/         # Reusable UI components
│   │   └── navbar.html     # Navigation component
│   └── ...                 # Page templates
├── utils/                  # Utility functions
│   ├── chatbot.py          # Chatbot implementation
│   └── ...                 # Other utilities
├── middleware/             # Request/response middleware
│   └── auth.py             # Authentication middleware
└── streaming/              # Real-time data streaming
    └── events.py           # WebSocket event handlers
```

### Development Workflow
1. Create a new branch for your feature
2. Make changes and test locally
3. Run with debug mode for hot-reloading: `python app.py`
4. Submit a pull request for review

### Testing
Run unit tests with:
```bash
python -m unittest discover tests
```

## 📚 API Documentation

The system provides RESTful API endpoints for integration with other systems.

### Authentication
```http
POST /login
Content-Type: application/json

{
  "username": "user",
  "password": "pass"
}
```

### Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "name": "Student Name",
  "student_id": "ST12345",
  "email": "student@example.com",
  "attendance": 85,
  "homework_completion": 90,
  "test_scores": 78
}
```

### Get Prediction History
```http
GET /api/predictions?skip=0&limit=20
Authorization: Bearer {token}
```

### Chat API
```http
POST /api/chat
Content-Type: application/json

{
  "query": "What is my prediction history?",
  "session_id": "unique-session-id"
}
```

For a complete API reference, see the [API Documentation](docs/api.md).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- [Mesum Bin Shaukat](https://github.com/mesumbinshaukat/)
- [Abdul Rafay Khan](https://github.com/abdulrafayKhan-10)
- [Huzaifa Irfan](https://github.com/Huzaifa1509)
- [Sarim Saleem](https://github.com/sarimkhan515)

## 🙏 Acknowledgments

- Special thanks to all contributors and testers
- Inspired by modern educational assessment systems
- Built with a focus on helping educators identify students needing additional support

---

*The thirst for learning, upgrading technical skills and applying the concepts in real life environment at a fast pace is what the industry demands from IT professionals today.*
