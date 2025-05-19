# Education Prediction System

A comprehensive web application that uses machine learning to predict student performance and provides an AI-powered chatbot for educational support.

## Features

- **Student Performance Prediction**
  - Predicts student performance based on attendance, homework completion, and test scores
  - Provides detailed analytics and insights
  - Generates performance reports with actionable recommendations

- **AI-Powered Chatbot**
  - Educational support through natural language conversations
  - Context-aware responses using LangChain and HuggingFace
  - Conversation history tracking and analysis

- **User Management**
  - Secure user authentication with Flask-Login
  - User registration and profile management
  - Role-based access control

- **Analytics Dashboard**
  - Visual representation of prediction results
  - Performance trends and statistics
  - Exportable reports in PDF format

## Technology Stack

- **Backend**: Python 3.8+, Flask 2.3.3
- **Database**: MongoDB 4.5.0
- **Machine Learning**: scikit-learn 1.3.1, numpy 2.2.0
- **AI/ML**: LangChain, HuggingFace Transformers
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Real-time Communication**: Flask-SocketIO, Eventlet

## Prerequisites

- Python 3.8 or higher
- MongoDB 4.0 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/education-prediction.git
   cd education-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Run the setup script:
   ```bash
   python setup.py
   ```
   This will:
   - Check Python version compatibility
   - Create necessary directories
   - Set up MongoDB connection
   - Create a sample ML model
   - Generate .env file
   - Clean up unused files

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update the following variables:
     - `MONGO_URI`: MongoDB connection string
     - `SECRET_KEY`: Flask secret key
     - `HUGGINGFACE_API_KEY`: (Optional) For enhanced chatbot

## Usage

1. Start MongoDB service:
   ```bash
   # On Windows
   net start MongoDB
   # On Unix or MacOS
   sudo service mongod start
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Access the application:
   - Open your web browser
   - Navigate to `http://localhost:5000`

## Project Structure

```
education-prediction/
├── app.py                 # Main application file
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
├── setup.py             # Setup script
├── models/              # ML model files
│   ├── best_model.pkl
│   └── scaler.pkl
├── knowledge/           # Chatbot knowledge base
│   └── conversations/
├── logs/               # Application logs
│   └── chatbot/
├── static/            # Static files
│   ├── css/
│   ├── js/
│   └── images/
├── templates/         # HTML templates
│   └── components/
└── utils/            # Utility modules
    └── chatbot.py
```

## API Endpoints

- `POST /api/predict`: Generate student performance prediction
- `GET /api/predictions`: Retrieve prediction history
- `POST /api/chat`: Chatbot interaction endpoint
- `GET /api/analytics`: Get analytics data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- Flask-Login for authentication
- LangChain for AI/ML integration
- MongoDB for database management
- Bootstrap for frontend design
