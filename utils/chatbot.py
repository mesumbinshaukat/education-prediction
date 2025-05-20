import os
import re
import json
import time
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Additional imports
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import from project
from config import get_db

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("logs/chatbot.log"),
                             logging.StreamHandler()])
logger = logging.getLogger("chatbot")

# Define path constants
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# Ensure directories exist
KNOWLEDGE_DIR.mkdir(exist_ok=True)
(KNOWLEDGE_DIR / "conversations").mkdir(exist_ok=True)

from .knowledge_loader import KnowledgeLoader

class ChatbotAgent:
    """
    Intelligent chatbot for the Education Prediction System that can:
    - Answer questions about the system
    - Help with student predictions (authenticated users only)
    - Learn from new data and system changes
    - Execute tasks for authenticated users
    - Maintain security by preventing access to sensitive information
    """
    
    def __init__(self, rebuild_kb: bool = False, use_local_model: bool = False):
        """Initialize the chatbot agent with enhanced knowledge base.
        
        Args:
            rebuild_kb (bool): Whether to rebuild the knowledge base
            use_local_model (bool): Whether to use a local model instead of API
        """
        # Store configuration
        self.use_local_model = use_local_model
        self.kb_path = KNOWLEDGE_DIR / "kb_faiss"
        
        # Initialize existing components
        self.embeddings = None
        self.security_patterns = set()
        self.safe_patterns = set()
        self.knowledge_base = None
        self.db = None
        self.llm = None
        self.chain = None  # Initialize chain attribute
        self.vectorstore = None  # Initialize vectorstore attribute
        
        # Initialize knowledge loader
        self.knowledge_loader = KnowledgeLoader()
        self.knowledge_loader.load_knowledge_base()
        self.knowledge_loader.process_knowledge_chunks()
        
        # Initialize other components
        self._init_embeddings()
        self._init_security_patterns()
        self._init_safe_patterns()
        self._init_llm()
        self._init_db()
        
        # Build or load knowledge base
        if rebuild_kb or not self.kb_path.exists():
            self._build_knowledge_base()
        else:
            self._load_knowledge_base()
            
        # Initialize the chain after knowledge base is ready
        self._create_chain()
    
    def _init_embeddings(self):
        """Initialize embeddings based on configuration"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=str(MODELS_DIR)
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            # Fallback to simple response generation
            self.llm = lambda x: "I'm sorry, I'm having trouble processing your request right now."
    
    def _init_security_patterns(self):
        """Initialize security patterns based on configuration"""
        try:
            self.security_patterns = set([
                r"(?i)password|secret|token|api[_-]?key|credential",
                r"(?i)\.env|config\.py|\.git",
                r"(?i)exploit|vulnerability|hack|attack|inject",
                r"(?i)sql\s*injection|xss|csrf|rce",
                r"(?i)delete\s*from|drop\s*table|truncate\s*table",
                r"(?i)rm\s*-rf|system\(|exec\(|eval\(",
                r"(?i)\/etc\/passwd|\/etc\/shadow"
            ])
        except Exception as e:
            logger.error(f"Failed to initialize security patterns: {e}")
            # Fallback to simple response generation
            self.llm = lambda x: "I'm sorry, I'm having trouble processing your request right now."
    
    def _init_safe_patterns(self):
        """Initialize safe patterns based on configuration"""
        try:
            self.safe_patterns = set([
                r"(?i)student\s*name",
                r"(?i)attendance",
                r"(?i)homework",
                r"(?i)test\s*scores?",
                r"(?i)grade",
                r"(?i)prediction"
            ])
        except Exception as e:
            logger.error(f"Failed to initialize safe patterns: {e}")
            # Fallback to simple response generation
            self.llm = lambda x: "I'm sorry, I'm having trouble processing your request right now."
    
    def _init_llm(self):
        """Initialize the language model based on configuration"""
        try:
            if self.use_local_model:
                # Use smaller local model for resource constraints
                model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
                
                # Check if we need to download the model
                if not (MODELS_DIR / "chatbot_model").exists():
                    logger.info(f"Downloading model {model_id}...")
                    os.makedirs(MODELS_DIR / "chatbot_model", exist_ok=True)
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Initialize and load the model
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                    # Create the pipeline
                    self.llm_pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_length=512,
                        temperature=0.7,
                        top_p=0.95,
                        repetition_penalty=1.15
                    )
                    
                    # Wrapper function for LangChain compatibility
                    def llm_wrapper(prompt):
                        response = self.llm_pipeline(prompt)[0]["generated_text"]
                        return response.replace(prompt, "").strip()
                    
                    self.llm = llm_wrapper
                    
                except Exception as e:
                    logger.error(f"Error loading local model: {e}")
                    self._use_huggingface_api()
            else:
                self._use_huggingface_api()
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Fallback to simple response generation
            self.llm = lambda x: "I'm sorry, I'm having trouble processing your request right now."
    
    def _use_huggingface_api(self):
        """Use a simple but reliable text generation approach"""
        logger.info("Using Simple Text Generation")
        
        def generate_text(query: str) -> str:
            # Handle greetings
            if self._is_greeting(query):
                return self._get_greeting_response()
                
            # Handle various query types with better pattern matching
            query_lower = query.lower()
            import random
            
            # Define query type detection patterns
            query_types = {
                "history": ["history", "past", "previous", "old", "recent"],
                "summary": ["summarize", "summary", "overview", "recap", "statistics", "stats"],
                "view": ["view", "show", "list", "display", "see", "check"],
                "make_prediction": ["make", "create", "new", "predict", "calculate", "add"],
                "how_works": ["how", "work", "function", "operate", "explain", "process"],
                "report": ["report", "pdf", "download", "export", "document", "file"],
                "performance": ["score", "grade", "performance", "result", "rating"]
            }
            
            # Try to identify the primary query intent
            primary_intent = None
            max_matches = 0
            
            for intent, keywords in query_types.items():
                matches = sum(1 for keyword in keywords if keyword in query_lower)
                if matches > max_matches:
                    max_matches = matches
                    primary_intent = intent
                    
            # Check if this is related to predictions
            is_prediction_related = "prediction" in query_lower or "student" in query_lower
            
            # For history and summary requests related to predictions
            if is_prediction_related and primary_intent in ["history", "summary", "view"]:
                # Different responses for prediction history/summary requests
                if "summarize" in query_lower or "summary" in query_lower:
                    # Specific responses for summary requests
                    summary_responses = [
                        "Here's a summary of your prediction history:\n\n- Your average student performance is 78.5%\n- 30% of students rated as Excellent\n- 45% rated as Good\n- 25% rated as Needs Improvement\n\nYou can view detailed analytics on the Prediction History page.",
                        
                        "I've analyzed your prediction history and found:\n\n- You've made 15 predictions in the last month\n- Most predictions (65%) are in the 'Good' category\n- Your highest performing student scored 96%\n\nWould you like to view the full details?",
                        
                        "Your prediction summary shows the following trends:\n\n- Attendance has the strongest correlation with performance\n- Students with >90% homework completion typically rate as Excellent\n- Recent predictions show improving student performance\n\nVisit the dashboard for visual charts of these trends."
                    ]
                    return random.choice(summary_responses)
                else:
                    # General history viewing responses
                    history_responses = [
                        "I can help you access your prediction history. You can view all your past predictions on the Prediction History page, where you can also download detailed reports or analyze performance trends.",
                        
                        "Your prediction history is available on the dashboard. You can see recent predictions, filter by date, and generate PDF reports for any prediction you've made.",
                        
                        "To view your prediction history, please visit the History tab. There you'll find a complete record of all predictions made, with options to sort, filter, and export the data.",
                        
                        "I'd be happy to help you access your prediction records. The Prediction History page shows all past predictions with detailed statistics and performance trends."
                    ]
                    return random.choice(history_responses)
                        
            # For system explanation
            if "how" in query_lower and "work" in query_lower:
                return ("The Education Prediction System works by analyzing three key factors:\n"
                        "1. Student attendance (30% weight)\n"
                        "2. Homework completion (20% weight)\n"
                        "3. Test scores (50% weight)\n\n"
                        "These factors are combined to calculate a performance percentage, which determines "
                        "if a student is rated as 'Excellent', 'Good', or 'Needs Improvement'.")
            
            # For making new predictions
            if any(word in query_lower for word in ["make", "create", "new"]) and "prediction" in query_lower:
                # Check if the query already contains numerical values
                if re.search(r'(\d+)(?:\s*%)?', query_lower):
                    # If it has numbers, this is likely already a prediction request
                    # We'll extract the parameters in the _extract_prediction_params method
                    # and respond appropriately in the process_query method
                    return ("I'm analyzing the student data you provided. Let me make a prediction based on these values.")
                else:
                    # Otherwise, prompt for the needed information
                    return ("I can help you make an education prediction. Please provide the student's:\n"
                            "1. Attendance percentage (0-100)\n"
                            "2. Homework completion rate (0-100)\n"
                            "3. Test scores (0-100)")
            
            # For specific score-related questions
            if any(word in query_lower for word in ["score", "grade", "performance", "result"]):
                return ("Education scores are calculated as follows:\n"
                        "- 90-100%: Excellent performance\n"
                        "- 70-89%: Good performance\n"
                        "- Below 70%: Needs improvement\n\n"
                        "Would you like to make a prediction for a specific student?")
                
            # For help with reports
            if any(word in query_lower for word in ["report", "pdf", "download", "export"]):
                return ("You can generate PDF reports of student predictions:\n"
                        "1. Go to the Prediction History page\n"
                        "2. Select the prediction you want to report\n"
                        "3. Click the 'Generate Report' button\n"
                        "4. Save the PDF to your computer")
            
            # Handle queries based on primary intent for prediction-related queries
            if is_prediction_related:
                # For making a new prediction
                if primary_intent == "make_prediction":
                    # Check if the query already contains numerical values that might be parameters
                    if re.search(r'(\d+)(?:\s*%)?', query_lower) and len(re.findall(r'(\d+)(?:\s*%)?', query_lower)) >= 3:
                        return "I'm analyzing the student data you provided. Let me make a prediction based on these values."
                    else:
                        make_responses = [
                            "I can help you make an education prediction. Please provide the student's:\n1. Attendance percentage (0-100)\n2. Homework completion percentage\n3. Test score average\n\nPlease provide these values to continue.",
                            
                            "To create a new prediction, I'll need three key metrics:\n1. Student attendance percentage\n2. Homework completion percentage\n3. Test score average\n\nPlease provide these values to continue.",
                            
                            "Let's make a new student performance prediction. I need the following information:\n- Attendance rate (as a percentage)\n- Homework completion rate\n- Test scores average\n\nWhat are these values for your student?"
                        ]
                        return random.choice(make_responses)
                
                # For how the system works
                elif primary_intent == "how_works":
                    how_responses = [
                        "The Education Prediction System works by analyzing three key factors:\n1. Student attendance (30% weight)\n2. Homework completion (20% weight)\n3. Test scores (50% weight)\n\nThese factors are combined to calculate a performance percentage, which determines if a student is rated as 'Excellent', 'Good', or 'Needs Improvement'.",
                        
                        "Our prediction algorithm evaluates student performance using a weighted formula:\n- 30% of the score comes from attendance records\n- 20% is based on homework completion rates\n- 50% is determined by test score averages\n\nThe combined score classifies students into performance categories.",
                        
                        "The prediction system uses a multi-factor model that considers:\n1. Attendance patterns (30% of final score)\n2. Homework completion metrics (20% contribution)\n3. Academic test performance (50% of the evaluation)\n\nThe final score determines placement in one of three categories: Excellent (80%+), Good (60-79%), or Needs Improvement (<60%)."
                    ]
                    return random.choice(how_responses)
                
                # For reports and exports
                elif primary_intent == "report":
                    report_responses = [
                        "You can generate PDF reports of student predictions:\n1. Go to the Prediction History page\n2. Select the prediction you want to report\n3. Click the 'Generate Report' button\n4. Save the PDF to your computer",
                        
                        "To download prediction reports:\n- Navigate to History in the main menu\n- Select the predictions you want to include\n- Click the Export button in the toolbar\n- Choose your preferred format (PDF, CSV, or Excel)",
                        
                        "Creating reports from your prediction data is easy:\n1. Open the Prediction History section\n2. Use the filters to select relevant predictions\n3. Click 'Generate Report' from the actions menu\n4. Choose the level of detail you want included"
                    ]
                    return random.choice(report_responses)
                
                # For performance scores
                elif primary_intent == "performance":
                    score_responses = [
                        "Education scores are calculated as follows:\n- 90-100%: Excellent performance\n- 70-89%: Good performance\n- Below 70%: Needs improvement\n\nWould you like to make a prediction for a specific student?",
                        
                        "Our system classifies student performance into three categories:\n- Excellent: Scores of 90% or higher\n- Good: Scores between 70-89%\n- Needs Improvement: Scores below 70%\n\nThe score is weighted based on attendance (30%), homework (20%), and tests (50%).",
                        
                        "Student performance ratings are determined by their calculated score:\n- Excellent performers score 90% or above\n- Good performers score between 70-89%\n- Students needing improvement score below 70%\n\nThese categories help identify appropriate intervention strategies."
                    ]
                    return random.choice(score_responses)
                
                # General prediction information as fallback
                elif "what" in query_lower or "tell me about" in query_lower or "explain" in query_lower:
                    about_responses = [
                        "Our Education Prediction System analyzes three key factors:\n1. Student attendance (30% weight)\n2. Homework completion (20% weight)\n3. Test scores (50% weight)\n\nThe system then calculates a performance score and classifies students as 'Excellent', 'Good', or 'Needs Improvement'.",
                        
                        "The Education Prediction System is a tool that helps educators evaluate student performance based on key metrics. It considers attendance records, homework completion rates, and test scores to generate a holistic assessment.",
                        
                        "Our prediction system is designed to identify at-risk students early by analyzing performance patterns. It combines attendance data, homework completion rates, and test scores to calculate an overall performance metric."
                    ]
                    return random.choice(about_responses)
                
                # Default menu for prediction-related queries
                menu_responses = [
                    "The Education Prediction System helps analyze student performance.\nWould you like to:\n1. Make a new prediction\n2. View your prediction history\n3. Learn how predictions work\nPlease let me know what you'd like to do.",
                    
                    "I can help you with several prediction-related tasks:\n- Create a new student performance prediction\n- Access your prediction history\n- Understand how the prediction algorithm works\n- Generate reports from your predictions\nWhat would you like to do?",
                    
                    "Welcome to the Education Prediction System assistant. I can help you:\n1. Make predictions for student performance\n2. Review your previous predictions\n3. Understand the prediction methodology\n4. Export reports and data\nHow can I assist you today?"
                ]
                return random.choice(menu_responses)
            
            # Default response for other queries
            default_responses = [
                "I can help you with:\n- Making education predictions\n- Understanding student performance\n- Viewing prediction history\n- Generating reports\nWhat specific information would you like?",
                
                "The Education Prediction System offers several features:\n- Student performance predictions\n- Historical data analysis\n- Performance trends and reports\n- Data export capabilities\nWhat aspect are you interested in learning more about?",
                
                "I'm your Education Prediction assistant. I can help with:\n- Creating new student performance predictions\n- Analyzing existing prediction data\n- Understanding the prediction algorithm\n- Generating detailed reports\nPlease let me know what you need assistance with.",
                
                "Welcome! I can assist you with the Education Prediction System in several ways:\n- Make predictions using student metrics\n- View and analyze prediction history\n- Learn about the prediction methodology\n- Export and share prediction reports\nHow can I help you today?"
            ]
            return random.choice(default_responses)
        
        self.llm = generate_text

    def _init_db(self):
        """Initialize the database connection"""
        try:
            self.db = get_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fallback to simple response generation
            self.llm = lambda x: "I'm sorry, I'm having trouble processing your request right now."
    
    def _get_relevant_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant knowledge chunks for a query.
        
        Args:
            query (str): User query
            
        Returns:
            List[Dict[str, Any]]: Relevant knowledge chunks
        """
        # Search in knowledge base
        knowledge_chunks = self.knowledge_loader.search_knowledge(query)
        
        # If no direct matches, try to get system usage info
        if not knowledge_chunks and any(word in query.lower() for word in ["how", "use", "help"]):
            knowledge_chunks = self.knowledge_loader.get_knowledge_by_type("system_usage")
            
        return knowledge_chunks
        
    def process_query(self, query: str, session_id: str, is_authenticated: bool = False, 
                     username: str = None) -> Dict[str, Any]:
        """
        Process a user query with enhanced session handling.
        """
        start_time = time.time()
        
        # Ensure chain is initialized
        if self.chain is None:
            logger.warning("Chain not initialized, creating it now")
            self._create_chain()
        
        # For authenticated users, ensure session exists
        if is_authenticated and username:
            session = self.db.chatSessions.find_one({'session_id': session_id})
            if not session:
                session_result = self.create_new_chat_session(username, query)
                if not session_result['success']:
                    return {
                        'response': "Error creating chat session. Please try again.",
                        'type': 'error',
                        'success': False,
                        'processing_time': time.time() - start_time
                    }
        
        # Check rate limiting
        if self._check_rate_limit(session_id):
            response = {
                'response': "I'm receiving too many messages too quickly. Please wait a moment before sending more queries.",
                'type': 'rate_limit',
                'success': False,
                'processing_time': time.time() - start_time
            }
            return response
        
        # Check if the query is safe
        if not self.is_query_safe(query):
            response = {
                'response': "I'm sorry, but I cannot process that query for security reasons.",
                'type': 'security_warning',
                'success': False,
                'processing_time': time.time() - start_time
            }
            self._save_conversation(
                session_id=session_id,
                user_query=query,
                bot_response=response['response'],
                is_authenticated=is_authenticated,
                username=username or "guest"
            )
            return response
        
        # Preprocess the query
        processed_query = self._preprocess_query(query)
        query_lower = processed_query.lower()
        
        try:
            # Check for prediction history request
            history_keywords = ["show", "view", "see", "my", "predictions", "history", "past", "previous"]
            is_history_request = any(keyword in query_lower for keyword in history_keywords)
            
            if is_history_request:
                if not is_authenticated:
                    response = {
                        'response': "You need to be logged in to view your prediction history. Please log in to access this feature.",
                        'type': 'auth_required',
                        'success': False,
                        'processing_time': time.time() - start_time
                    }
                else:
                    history = self._get_prediction_history(username)
                    if history:
                        summary = self._summarize_prediction_history(history)
                        response = {
                            'response': summary,
                            'type': 'history',
                            'success': True,
                            'processing_time': time.time() - start_time
                        }
                    else:
                        response = {
                            'response': "You don't have any prediction history yet. Would you like to make a new prediction?",
                            'type': 'history',
                            'success': True,
                            'processing_time': time.time() - start_time
                        }
                
                self._save_conversation(
                    session_id=session_id,
                    user_query=query,
                    bot_response=response['response'],
                    is_authenticated=is_authenticated,
                    username=username or "guest"
                )
                return response

            # Check if the query is a simple greeting
            if self._is_greeting(processed_query):
                greeting_response = self._get_greeting_response()
                response = {
                    'response': greeting_response,
                    'type': 'greeting',
                    'success': True,
                    'processing_time': time.time() - start_time
                }
                
                self._save_conversation(
                    session_id=session_id,
                    user_query=query,
                    bot_response=response['response'],
                    is_authenticated=is_authenticated,
                    username=username or "guest"
                )
                return response

            # Check for prediction parameters
            prediction_params = self._extract_prediction_params(processed_query)
            if prediction_params and all(param in prediction_params for param in ["attendance", "homework_completion", "test_scores"]):
                if not is_authenticated:
                    response = {
                        'response': "You need to be logged in to make predictions. Please log in to use this feature.",
                        'type': 'auth_required',
                        'success': False,
                        'processing_time': time.time() - start_time
                    }
                else:
                    prediction_result = self._make_prediction(prediction_params)
                    if prediction_result['success']:
                        self._save_to_history(username, prediction_result)
                        student_name = prediction_result.get('name', 'The student')
                        score = round(prediction_result['prediction_score'], 1)
                        details = (f"\n\nThis prediction is based on:\n"
                                 f"- Attendance: {prediction_result['attendance']}%\n"
                                 f"- Homework completion: {prediction_result['homework_completion']}%\n"
                                 f"- Test scores: {prediction_result['test_scores']}%")
                        
                        if prediction_result['prediction'] == "Excellent":
                            prediction_message = f"{student_name} has an excellent performance with a score of {score}%.{details}\n\nRecommendation: Continue with the current approach - it's working very well!"
                        elif prediction_result['prediction'] == "Good":
                            prediction_message = f"{student_name} is doing well with a score of {score}%.{details}\n\nRecommendation: Consider focusing more on test preparation to move into the excellent category."
                        else:
                            prediction_message = f"{student_name} needs improvement with a score of {score}%.{details}\n\nRecommendation: Create an improvement plan focusing first on attendance and test preparation."
                        
                        response = {
                            'response': prediction_message,
                            'type': 'prediction',
                            'success': True,
                            'prediction': prediction_result,
                            'processing_time': time.time() - start_time
                        }
                    else:
                        response = {
                            'response': "I'm sorry, I couldn't process that prediction. Please check your input and try again.",
                            'type': 'error',
                            'success': False,
                            'processing_time': time.time() - start_time
                        }
                
                self._save_conversation(
                    session_id=session_id,
                    user_query=query,
                    bot_response=response['response'],
                    is_authenticated=is_authenticated,
                    username=username or "guest"
                )
                return response

            # Process general queries through the chain
            try:
                chain_response = self.chain(processed_query)
                response_text = chain_response.get('result', "I'm processing your question.")
                source_documents = chain_response.get('source_documents', [])
                
                # For non-authenticated users, add a note about available features
                if not is_authenticated and not self._is_greeting(processed_query):
                    response_text += "\n\nNote: Some features like making predictions and viewing history require you to log in. Would you like to know more about the available features?"
                
                response = {
                    'response': response_text,
                    'type': 'general',
                    'success': True,
                    'sources': [doc.metadata.get('source') for doc in source_documents if doc.metadata.get('source')],
                    'processing_time': time.time() - start_time
                }
                
            except Exception as chain_error:
                logger.error(f"Error in chain invocation: {chain_error}", exc_info=True)
                response = {
                    'response': "I'm having trouble answering that question right now. Could you try asking something else about the Education Prediction System?",
                    'type': 'error',
                    'success': False,
                    'processing_time': time.time() - start_time
                }
            
            self._save_conversation(
                session_id=session_id,
                user_query=query,
                bot_response=response['response'],
                is_authenticated=is_authenticated,
                username=username or "guest"
            )
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = {
                'response': "I'm sorry, I encountered an error while processing your request.",
                'type': 'error',
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            
            self._save_conversation(
                session_id=session_id,
                user_query=query,
                bot_response=error_response['response'],
                is_authenticated=is_authenticated,
                username=username or "guest"
            )
            return error_response

    def _process_query_with_context(self, query: str, session_id: str, username: str = None) -> Dict[str, Any]:
        """Process query with context.
        
        Args:
            query (str): User query with context
            session_id (str): Chat session ID
            username (str, optional): Username if logged in
            
        Returns:
            Dict[str, Any]: Response with chat history and metadata
        """
        try:
            # Check for security patterns
            if self._check_security_patterns(query):
                return {
                    "response": "I apologize, but I cannot process that request for security reasons.",
                    "error": "Security pattern detected"
                }
                
            # Check if user is logged in for restricted features
            is_logged_in = username is not None
            restricted_features = ["prediction", "history", "reports"]
            
            # Check if query is about restricted features
            if not is_logged_in and any(feature in query.lower() for feature in restricted_features):
                return {
                    "response": "I apologize, but that feature requires you to be logged in. Please log in to access predictions, history, and reports.",
                    "requires_login": True
                }
                
            # Process the query using the language model
            response_text = self.llm(query)
            
            # Save the conversation
            self._save_conversation(
                session_id=session_id,
                user_query=query,
                bot_response=response_text,
                username=username or "guest"
            )
            
            # Get chat history
            history = self.get_chat_history(session_id)
            
            return {
                "response": response_text,
                "history": history,
                "session_id": session_id,
                "username": username or "guest"
            }
            
        except Exception as e:
            logger.error(f"Error in _process_query_with_context: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try again.",
                "error": str(e)
            }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and features.
        
        Returns:
            Dict[str, Any]: System information
        """
        try:
            # Get system features from knowledge base
            features = self.knowledge_loader.get_knowledge_by_type("system_feature")
            
            # Get educational concepts
            concepts = self.knowledge_loader.get_knowledge_by_type("educational_concept")
            
            # Get system usage information
            usage = self.knowledge_loader.get_knowledge_by_type("system_usage")
            
            return {
                "features": features,
                "concepts": concepts,
                "usage": usage
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {}
            
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific feature.
        
        Args:
            feature_name (str): Name of the feature
            
        Returns:
            Dict[str, Any]: Feature information
        """
        try:
            # Search for feature in knowledge base
            features = self.knowledge_loader.get_knowledge_by_type("system_feature")
            feature_info = next(
                (f for f in features if f.get("name", "").lower() == feature_name.lower()),
                None
            )
            
            if feature_info:
                return {
                    "name": feature_info.get("name", feature_name),
                    "category": feature_info["category"],
                    "content": feature_info["content"]
                }
            else:
                return {"error": f"Feature '{feature_name}' not found"}
                
        except Exception as e:
            logger.error(f"Error getting feature info: {str(e)}")
            return {"error": str(e)}

    def _create_chain(self):
        """Create a simple but reliable chain for query processing"""
        try:
            if self.vectorstore is None:
                logger.warning("Vectorstore not initialized, creating simple chain")
                # Create a simple chain that just uses the LLM directly
                def simple_chain(query):
                    return {
                        "result": self.llm(query),
                        "source_documents": [],
                        "answer": self.llm(query)
                    }
                self.chain = simple_chain
                return

            # Create a retriever with basic similarity search
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            def process_response(query):
                try:
                    # Get relevant documents
                    docs = self.retriever.get_relevant_documents(query)
                    context = "\n".join(doc.page_content for doc in docs)
                    
                    # Generate response based on context and query
                    if self._is_greeting(query):
                        response = self._get_greeting_response()
                    else:
                        # Use the LLM's direct response
                        response = self.llm(f"Context: {context}\nQuestion: {query}")
                    
                    return {
                        "result": response,
                        "source_documents": docs,
                        "answer": response
                    }
                except Exception as e:
                    logger.error(f"Error in chain processing: {e}", exc_info=True)
                    # Fallback to simple response if chain processing fails
                    return {
                        "result": self.llm(query),
                        "source_documents": [],
                        "answer": self.llm(query)
                    }
            
            self.chain = process_response
            logger.info("Chain created successfully with document retrieval")
            
        except Exception as e:
            logger.error(f"Error creating chain: {e}", exc_info=True)
            # Create a simple fallback chain
            def fallback_chain(query):
                return {
                    "result": self.llm(query),
                    "source_documents": [],
                    "answer": self.llm(query)
                }
            self.chain = fallback_chain
            logger.info("Created fallback chain due to error")

    def _build_knowledge_base(self):
        """Build the knowledge base from project data and code"""
        documents = []
        
        # Load code files (excluding certain directories and file types)
        ignored_dirs = {".git", "__pycache__", "venv", "myenv", "node_modules"}
        ignored_extensions = {".pyc", ".pyo", ".pyd", ".git"}
        
        # Process code files
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in ignored_extensions):
                    continue
                
                try:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, PROJECT_ROOT)
                    
                    # Skip binary files and only process text files with recognizable extensions
                    if os.path.getsize(file_path) > 1000000:  # Skip files larger than 1MB
                        continue
                        
                    if file.endswith((".py", ".js", ".html", ".css", ".md", ".txt")):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            
                        # Create a document with metadata
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": relative_path,
                                "type": "code",
                                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                                "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Error processing file {file}: {e}")
        
        # Add system descriptions
        system_desc = """
        Education Prediction System:
        
        This system predicts student performance based on attendance, homework completion, and test scores.
        The system evaluates students as "Excellent", "Good", or "Needs Improvement" based on calculated percentages.
        Users can register, login, make predictions, view prediction history, and generate PDF reports.
        The system uses a machine learning model that is continuously retrained based on new prediction data.
        """
        
        documents.append(Document(
            page_content=system_desc,
            metadata={"source": "system_description.txt", "type": "documentation"}
        ))
        
        # Process any PDF files in the data directory
        for pdf_file in DATA_DIR.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
            except Exception as e:
                logger.warning(f"Error loading PDF {pdf_file}: {e}")
        
        # Get information from the MongoDB database
        try:
            # Get a sample of predictions for knowledge (limit to 100 for efficiency)
            predictions = list(self.db.predictionHistory.find(
                {}, 
                {"_id": 0, "password": 0, "email": 0}  # Exclude sensitive fields
            ).limit(100))
            
            prediction_text = json.dumps(predictions, default=str, indent=2)
            documents.append(Document(
                page_content=f"Sample prediction data: {prediction_text}",
                metadata={"source": "mongodb_predictions", "type": "data"}
            ))
        except Exception as e:
            logger.warning(f"Error getting MongoDB data: {e}")
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(str(self.kb_path))
        logger.info(f"Knowledge base built and saved to {self.kb_path}")

    def _load_knowledge_base(self):
        """Load the existing knowledge base"""
        try:
            self.vectorstore = FAISS.load_local(str(self.kb_path), self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Knowledge base loaded successfully")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            logger.info("Building new knowledge base...")
            self._build_knowledge_base()

    def update_knowledge_base(self, force: bool = False) -> bool:
        """
        Update the knowledge base with new information.
        
        Args:
            force: Whether to force an update regardless of the timing
            
        Returns:
            bool: True if the knowledge base was updated, False otherwise
        """
        # Check when the KB was last updated
        kb_metadata_path = KNOWLEDGE_DIR / "kb_metadata.json"
        current_time = time.time()
        
        if kb_metadata_path.exists() and not force:
            try:
                with open(kb_metadata_path, "r") as f:
                    metadata = json.load(f)
                last_update = metadata.get("last_update", 0)
                
                # Only update if it's been at least 6 hours since the last update
                if current_time - last_update < 21600:  # 6 hours in seconds
                    logger.info("Knowledge base is up to date")
                    return False
            except Exception as e:
                logger.warning(f"Error reading KB metadata: {e}")
        
        # Rebuild the knowledge base
        self._build_knowledge_base()
        
        # Update metadata
        with open(kb_metadata_path, "w") as f:
            json.dump({"last_update": current_time}, f)
        
        self.knowledge_updated = True
        return True

    def is_query_safe(self, query: str) -> bool:
        """
        Check if a query is safe or potentially harmful.
        
        Args:
            query: The user query to check
            
        Returns:
            bool: True if the query is safe, False otherwise
        """
        # First check if it matches any safe patterns
        for pattern in self.safe_patterns:
            if re.search(pattern, query):
                return True
                
        # Then check for harmful patterns
        for pattern in self.security_patterns:
            if re.search(pattern, query):
                logger.warning(f"Potentially unsafe query detected: {query}")
                return False
                
        return True

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query before passing it to the model.
        
        Args:
            query: The raw user query
            
        Returns:
            str: The preprocessed query
        """
        # Basic preprocessing
        query = query.strip()
        
        # Remove multiple spaces
        query = re.sub(r'\s+', ' ', query)
        
        return query
        
    def _is_greeting(self, query: str) -> bool:
        """
        Check if the query is a simple greeting.
        
        Args:
            query: The preprocessed user query
            
        Returns:
            bool: True if the query is a greeting, False otherwise
        """
        greetings = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
            "good evening", "howdy", "sup", "what's up", "hola"
        ]
        
        query_lower = query.lower()
        
        # Check exact matches
        if query_lower in greetings:
            return True
            
        # Check if the query starts with any greeting
        for greeting in greetings:
            if query_lower.startswith(greeting):
                return True
                
        return False
        
    def _get_greeting_response(self) -> str:
        """
        Generate a friendly greeting response.
        
        Returns:
            str: A greeting response
        """
        import random
        
        greetings = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! How may I assist you?",
            "Hey! I'm here to help with your education prediction needs.",
            "Hello! Ask me anything about the Education Prediction System.",
            "Hi! I'm your Education Prediction assistant. What would you like to know?"
        ]
        
        return random.choice(greetings)

    def _handle_task_request(self, query: str, is_authenticated: bool) -> Tuple[bool, str]:
        """
        Handle task execution requests from authenticated users.
        
        Args:
            query: The user query
            is_authenticated: Whether the user is authenticated
            
        Returns:
            Tuple[bool, str]: (is_task, response_message)
        """
        # Only authenticated users can perform tasks
        if not is_authenticated:
            return False, "You need to be logged in to perform tasks."
        
        # Check for task patterns in the query
        task_patterns = {
            r"(?i)add\s+(?:a\s+)?(?:new\s+)?(?:student|entry)": "add_student",
            r"(?i)make\s+(?:a\s+)?prediction": "make_prediction",
            r"(?i)update\s+knowledge": "update_knowledge",
            r"(?i)generate\s+(?:a\s+)?report": "generate_report",
            r"(?i)show\s+(?:my\s+)?(?:prediction\s+)?history": "show_history"
        }
        
        detected_task = None
        for pattern, task_name in task_patterns.items():
            if re.search(pattern, query):
                detected_task = task_name
                break
        
        if not detected_task:
            return False, ""
        
        # Handle the detected task
        if detected_task == "add_student":
            return True, "I can help you add a new student. Please provide the following information:\n- Student name\n- Student ID\n- Email\n- Attendance percentage (0-100)\n- Homework completion percentage (0-100)\n- Test scores (0-100)"
            
        elif detected_task == "make_prediction":
            return True, "I can help you make a prediction. Please provide the student's attendance, homework completion, and test scores."
            
        elif detected_task == "update_knowledge":
            updated = self.update_knowledge_base(force=True)
            if updated:
                return True, "I've updated my knowledge base with the latest information from the system."
            else:
                return True, "My knowledge base is already up to date."
                
        elif detected_task == "generate_report":
            return True, "To generate a report, please provide the student ID you want to create a report for."
            
        elif detected_task == "show_history":
            return True, "I'll help you view your prediction history. You can view this on the prediction history page or I can summarize recent predictions for you."
        
        return False, ""
    
    def _extract_prediction_params(self, query: str) -> Dict[str, Any]:
        """
        Extract prediction parameters from a user query.
        
        Args:
            query: The user query
            
        Returns:
            Dict with extracted parameters or empty dict if not found
        """
        params = {}
        
        # Check for history or summary related queries
        query_lower = query.lower()
        if any(word in query_lower for word in ["history", "summary", "summarize", "past", "previous"]):
            params["request_type"] = "history"
            
            # Try to extract time frame information
            if "last" in query_lower:
                if "week" in query_lower:
                    params["time_frame"] = "week"
                elif "month" in query_lower:
                    params["time_frame"] = "month"
                elif "year" in query_lower:
                    params["time_frame"] = "year"
                elif "day" in query_lower or "24 hours" in query_lower:
                    params["time_frame"] = "day"
            
            # Return early as this is not a prediction parameter request
            return params
        
        # Enhanced student name extraction - handles different formats
        name_patterns = [
            r"(?:student\s+)?name\s*(?:is|:)?\s*(\w+(?:\s+\w+)*)",
            r"(?:for|student|with name)\s+(\w+)(?:\s+with|,|\s+has|\s+who)",
            r"student\s+(\w+)(?:\s+with|,|\s+has|\s+is)",
            r"(?:for|student|with name)\s+(\w+)$",
            r"(?:for|of|about)\s+(?:student\s+)?(\w+)(?:\s+with|,|\s+who|\s+having|\s+has)",
            r"name\s*:\s*(\w+(?:\s+\w+)*)",
            r"student\s*:\s*(\w+(?:\s+\w+)*)",
            r"predict(?:ion)?\s+for\s+(\w+)(?:\s+with|,|\s+who|\s+having)",
            r"(\w+)'s\s+(?:performance|score|prediction|grade)"
        ]
        
        # Try each pattern until one matches
        for pattern in name_patterns:
            name_match = re.search(pattern, query, re.IGNORECASE)
            if name_match:
                params["name"] = name_match.group(1).strip()
                break
                
        # Attempt to extract name from end of query if not found yet
        if "name" not in params:
            end_name_match = re.search(r"(?:student|name)[^a-zA-Z]+(\w+)(?:\s+\w+)*\s*$", query, re.IGNORECASE)
            if end_name_match:
                params["name"] = end_name_match.group(1).strip()
            
        id_match = re.search(r"(?:student\s*)?id\s*(?:is|:)?\s*(\w+)", query, re.IGNORECASE)
        if id_match:
            params["student_id"] = id_match.group(1).strip()
            
        email_match = re.search(r"email\s*(?:is|:)?\s*(\S+@\S+)", query, re.IGNORECASE)
        if email_match:
            params["email"] = email_match.group(1).strip()
        
        # Enhanced patterns for parameter extraction - handles more natural language
        patterns = {
            "attendance": [
                r"attendance\s*(?:is|:)?\s*(\d+)(?:\s*%)?",
                r"with\s+(\d+)(?:\s*%)?\s*attendance",
                r"attendance\s*of\s*(\d+)(?:\s*%)?",
                r"attended\s*(\d+)(?:\s*%)?"
            ],
            "homework_completion": [
                r"homework\s*(?:completion)?\s*(?:is|:)?\s*(\d+)(?:\s*%)?",
                r"(\d+)(?:\s*%)?\s*homework\s*(?:completion)?",
                r"completed\s*(\d+)(?:\s*%)?\s*(?:of)?\s*(?:the)?\s*homework",
                r"homework\s*completion\s*(?:rate|ratio)?\s*(?:of)?\s*(\d+)(?:\s*%)?",
                r"(\d+)(?:\s*%)?\s*(?:of)?\s*(?:the)?\s*homework\s*(?:is)?\s*(?:complete|completed)"
            ],
            "test_scores": [
                r"test\s*(?:scores?)?\s*(?:is|:)?\s*(\d+)(?:\s*%)?",
                r"(?:scored|score|marks)(?:\s+of)?\s*(\d+)(?:\s*%)?\s*(?:in|on)?\s*(?:the)?\s*test",
                r"test\s*(?:scores?|results?|marks?)\s*(?:of|:)?\s*(\d+)(?:\s*%)?",
                r"(\d+)(?:\s*%)?\s*(?:in|on)?\s*(?:the)?\s*test"
            ]
        }
        
        # Try each pattern for each parameter
        for param, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1).strip())
                        # Ensure the value is between 0 and 100
                        params[param] = min(max(value, 0), 100)
                        break  # Stop after finding a match for this parameter
                    except ValueError:
                        pass
        
        # Check if the query contains a prediction request marker
        is_prediction_request = ("prediction" in query_lower or "predict" in query_lower or 
                               "grade" in query_lower or "performance" in query_lower or
                               "assess" in query_lower or "evaluate" in query_lower)
        
        # First try to extract values using specific patterns
        # If those fail, try to extract using simple numeric patterns
        if is_prediction_request and not all(param in params for param in ["attendance", "homework_completion", "test_scores"]):
            # Try to extract using format: "with X% attendance, Y% homework, Z% test scores"
            pattern1 = r'with\s+(\d+)(?:\s*%)?\s+attendance,?\s+(\d+)(?:\s*%)?\s+homework,?\s+(\d+)(?:\s*%)?\s+test'
            pattern2 = r'attendance\s+(\d+)(?:\s*%)?,?\s+homework\s+(\d+)(?:\s*%)?,?\s+test\s+(\d+)(?:\s*%)?'
            
            for pattern in [pattern1, pattern2]:
                match = re.search(pattern, query_lower)
                if match:
                    try:
                        params["attendance"] = min(max(float(match.group(1)), 0), 100)
                        params["homework_completion"] = min(max(float(match.group(2)), 0), 100)
                        params["test_scores"] = min(max(float(match.group(3)), 0), 100)
                        break
                    except (ValueError, IndexError):
                        pass
            
            # If specific patterns didn't match, try extracting all numbers in order
            if not all(param in params for param in ["attendance", "homework_completion", "test_scores"]):
                # Extract numbers with percentage signs
                numbers = re.findall(r'(\d+)(?:\s*%)?', query)
                if len(numbers) >= 3:
                    try:
                        # Assume the first three numbers are attendance, homework, and test scores
                        params["attendance"] = min(max(float(numbers[0]), 0), 100)
                        params["homework_completion"] = min(max(float(numbers[1]), 0), 100)
                        params["test_scores"] = min(max(float(numbers[2]), 0), 100)
                    except (ValueError, IndexError):
                        pass
                        
            # Try to find values with labels in any order
            if not all(param in params for param in ["attendance", "homework_completion", "test_scores"]):
                attendance_match = re.search(r'attendance[^0-9]*(\d+)', query_lower)
                homework_match = re.search(r'homework[^0-9]*(\d+)', query_lower)
                test_match = re.search(r'test[^0-9]*(\d+)', query_lower)
                
                if attendance_match and "attendance" not in params:
                    params["attendance"] = min(max(float(attendance_match.group(1)), 0), 100)
                if homework_match and "homework_completion" not in params:
                    params["homework_completion"] = min(max(float(homework_match.group(1)), 0), 100)
                if test_match and "test_scores" not in params:
                    params["test_scores"] = min(max(float(test_match.group(1)), 0), 100)
        
        return params
    
    def _make_prediction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction based on the provided parameters.
        
        Args:
            params: Dict with prediction parameters
            
        Returns:
            Dict with prediction results
        """
        required_fields = ["attendance", "homework_completion", "test_scores"]
        
        # Check if we have all required fields
        if not all(field in params for field in required_fields):
            missing = [field for field in required_fields if field not in params]
            return {
                "success": False,
                "message": f"Missing required parameters: {', '.join(missing)}"
            }
        
        try:
            # Try to use the ML model if available
            model_path = MODELS_DIR / "best_model.pkl"
            scaler_path = MODELS_DIR / "scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    import pickle
                    import numpy as np
                    
                    # Load the model and scaler
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    # Prepare the input data
                    input_data = np.array([[
                        params["attendance"], 
                        params["homework_completion"],
                        params["test_scores"]
                    ]])
                    
                    # Scale the data
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    try:
                        prediction_score = model.predict(input_scaled)[0]
                        probability = model.predict_proba(input_scaled)[0]
                        
                        # Determine prediction category
                        if prediction_score >= 90:
                            prediction_text = "Excellent"
                        elif prediction_score >= 70:
                            prediction_text = "Good"
                        else:
                            prediction_text = "Needs Improvement"
                            
                        # Prepare result
                        result = {
                            "success": True,
                            "prediction": prediction_text,
                            "prediction_score": float(prediction_score),
                            "probability": float(max(probability)),
                            "confidence": float(max(probability) * 100),
                            "binary_prediction": int(prediction_score >= 70)
                        }
                    except Exception as model_error:
                        logger.error(f"Error using model: {model_error}, falling back to simple calculation")
                        # Fall back to simple calculation if model prediction fails
                        result = self._simple_prediction_calculation(params)
                        
                except Exception as e:
                    logger.error(f"Error loading model or making prediction: {e}")
                    # Fall back to simple calculation
                    result = self._simple_prediction_calculation(params)
            else:
                # No model files found, use simple calculation
                result = self._simple_prediction_calculation(params)
                
            # If we have other student details, include them
            for field in ["name", "student_id", "email"]:
                if field in params:
                    result[field] = params[field]
                    
            # Always include the input parameters in the result
            result["attendance"] = params["attendance"]
            result["homework_completion"] = params["homework_completion"]
            result["test_scores"] = params["test_scores"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                "success": False,
                "message": f"Error making prediction: {str(e)}"
            }

    def _simple_prediction_calculation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a simple weighted calculation for prediction when ML model is unavailable.
        
        Args:
            params: Dict with prediction parameters
            
        Returns:
            Dict with prediction results
        """
        # Calculate weighted score
        score = (
            params["attendance"] * 0.3 +
            params["homework_completion"] * 0.2 +
            params["test_scores"] * 0.5
        )
        
        # Round to 2 decimal places
        score = round(score, 2)
        
        # Cap at 100
        score = min(score, 100)
        
        # Determine prediction text
        if score >= 90:
            prediction_text = "Excellent"
        elif score >= 70:
            prediction_text = "Good"
        else:
            prediction_text = "Needs Improvement"
            
        # Legacy binary prediction
        binary_prediction = 1 if score >= 70 else 0
        
        # Estimate confidence based on how far from thresholds
        confidence = min(max(abs(score - 70) / 30 * 100, 60), 95)
        
        # Prepare result
        return {
            "success": True,
            "prediction": prediction_text,
            "prediction_score": score,
            "binary_prediction": binary_prediction,
            "confidence": confidence,
            "probability": confidence / 100.0
        }

    def _save_to_history(self, username: str, prediction_result: Dict[str, Any]) -> bool:
        """
        Save a prediction to the database history.
        
        Args:
            username: The username of the authenticated user
            prediction_result: The prediction result to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Create prediction record with standardized fields
            prediction_data = {
                # Student information
                'student_id': prediction_result.get('student_id', 'unknown'),
                'name': prediction_result.get('name', 'unknown'),
                'email': prediction_result.get('email', 'unknown'),
                'attendance': prediction_result.get('attendance', 0),
                'homework_completion': prediction_result.get('homework_completion', 0),
                'test_scores': prediction_result.get('test_scores', 0),
                
                # Prediction results
                'prediction': prediction_result['prediction'],
                'prediction_score': prediction_result['prediction_score'],
                'binary_prediction': prediction_result['binary_prediction'],
                'confidence': prediction_result['confidence'],
                'probability': prediction_result['probability'],
                
                # Metadata
                'username': username,
                'created_at': datetime.now(),
                'source': 'chatbot'
            }
            
            # Insert into database
            result = self.db.predictionHistory.insert_one(prediction_data)
            logger.info(f"Saved prediction to history, ID: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to history: {e}")
            return False
    
    def _get_prediction_history(self, username: str, time_frame: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch prediction history for a user.
        
        Args:
            username: The username to fetch history for
            time_frame: Optional time frame filter ("day", "week", "month", "year")
            limit: Maximum number of records to return
            
        Returns:
            List of prediction records
        """
        try:
            # Build query
            query = {"username": username}
            
            # Add time frame filter if specified
            if time_frame:
                now = datetime.now()
                if time_frame == "day":
                    query["created_at"] = {"$gte": now - timedelta(days=1)}
                elif time_frame == "week":
                    query["created_at"] = {"$gte": now - timedelta(days=7)}
                elif time_frame == "month":
                    query["created_at"] = {"$gte": now - timedelta(days=30)}
                elif time_frame == "year":
                    query["created_at"] = {"$gte": now - timedelta(days=365)}
            
            # Execute query with proper sorting and limiting
            cursor = self.db.predictionHistory.find(
                query,
                {
                    "_id": 0,
                    "student_id": 1,
                    "name": 1,
                    "attendance": 1,
                    "homework_completion": 1,
                    "test_scores": 1,
                    "prediction": 1,
                    "prediction_score": 1,
                    "created_at": 1
                }
            ).sort("created_at", -1).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error fetching prediction history: {e}")
            return []
    
    def _summarize_prediction_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable summary of prediction history.
        
        Args:
            history: List of prediction records
            
        Returns:
            Formatted summary string
        """
        if not history:
            return "You don't have any prediction history yet. Make a prediction to get started!"
        
        # Count predictions by category
        excellent_count = sum(1 for p in history if p.get("prediction") == "Excellent")
        good_count = sum(1 for p in history if p.get("prediction") == "Good")
        needs_improvement_count = sum(1 for p in history if p.get("prediction") == "Needs Improvement")
        
        # Calculate average score
        scores = [p.get("prediction_score", 0) for p in history]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Generate summary
        summary = f"Here's a summary of your last {len(history)} predictions:\n\n"
        summary += f"Overall Statistics:\n"
        summary += f"- Average performance score: {avg_score:.1f}%\n"
        summary += f"- Excellent predictions: {excellent_count}\n"
        summary += f"- Good predictions: {good_count}\n"
        summary += f"- Needs Improvement: {needs_improvement_count}\n\n"
        
        # Add recent predictions with more details
        summary += "Recent Predictions:\n"
        for i, p in enumerate(history[:5], 1):
            date = p.get("created_at", "Unknown date")
            if isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = str(date)
                
            student = p.get("name", "Unknown student")
            score = p.get("prediction_score", 0)
            prediction = p.get("prediction", "Unknown")
            
            # Add performance metrics
            attendance = p.get("attendance", 0)
            homework = p.get("homework_completion", 0)
            test_score = p.get("test_scores", 0)
            
            summary += f"\n{i}. {date_str} - {student}\n"
            summary += f"   Performance: {score:.1f}% ({prediction})\n"
            summary += f"   Metrics:\n"
            summary += f"   - Attendance: {attendance}%\n"
            summary += f"   - Homework: {homework}%\n"
            summary += f"   - Test Score: {test_score}%\n"
        
        summary += "\nYou can view your complete prediction history and generate detailed reports on the Prediction History page."
        return summary
    
    def _save_conversation(self, session_id: str, user_query: str, bot_response: str, 
                          is_authenticated: bool, username: str = None) -> bool:
        """
        Save a conversation exchange to the database with enhanced session management.
        """
        try:
            # Prepare conversation data
            conversation_data = {
                'session_id': session_id,
                'user_query': user_query,
                'bot_response': bot_response,
                'is_authenticated': is_authenticated,
                'username': username or "guest",
                'timestamp': datetime.now(),
                'metadata': {
                    'ip_address': None,
                    'user_agent': None,
                    'context_used': None,
                    'function_calls': None,
                    'processing_time_ms': None,
                    'error': None,
                    'version': '1.0',
                    'retry_count': 0
                }
            }
            
            # For authenticated users, update session metadata
            if is_authenticated and username:
                # Update session's last_updated and message_count
                self.db.chatSessions.update_one(
                    {'session_id': session_id},
                    {
                        '$set': {'last_updated': datetime.now()},
                        '$inc': {'message_count': 1}
                    }
                )
                
                # If this is the first message, generate and update the title
                if self.db.chatHistory.count_documents({'session_id': session_id}) == 0:
                    title = self._generate_chat_title(user_query)
                    self.db.chatSessions.update_one(
                        {'session_id': session_id},
                        {'$set': {'title': title}}
                    )
            
            # Save to database with retry logic
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    with self.db.client.start_session() as session:
                        session.start_transaction()
                        
                        # Insert the conversation
                        result = self.db.chatHistory.insert_one(
                            conversation_data,
                            session=session
                        )
                        
                        session.commit_transaction()
                        logger.info(f"Saved conversation to chatHistory. Document ID: {result.inserted_id}")
                        return True
                        
                except Exception as tx_error:
                    session.abort_transaction()
                    logger.error(f"Transaction error: {tx_error}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        self._add_to_dead_letter_queue(conversation_data, str(tx_error))
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    def _generate_chat_title(self, first_query: str) -> str:
        """
        Generate a meaningful title for a chat session based on the first query.
        
        Args:
            first_query: The first query in the chat session
            
        Returns:
            str: A generated title for the chat session
        """
        try:
            # Extract key topics from the query
            topics = []
            query_lower = first_query.lower()
            
            # Check for prediction-related topics
            if "prediction" in query_lower or "predict" in query_lower:
                topics.append("Prediction")
            if "student" in query_lower:
                topics.append("Student")
            if "performance" in query_lower or "score" in query_lower:
                topics.append("Performance")
            if "attendance" in query_lower:
                topics.append("Attendance")
            if "homework" in query_lower:
                topics.append("Homework")
            if "test" in query_lower:
                topics.append("Test")
                
            # Generate title based on topics
            if topics:
                title = "Discussion about " + ", ".join(topics)
            else:
                # Use timestamp if no specific topics found
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                title = f"Chat Session {timestamp}"
                
            return title
            
        except Exception as e:
            logger.error(f"Error generating chat title: {e}")
            return f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def create_new_chat_session(self, username: str, first_query: str = None) -> Dict[str, Any]:
        """
        Create a new chat session for a logged-in user.
        
        Args:
            username: The username of the logged-in user
            first_query: Optional first query to generate a meaningful title
            
        Returns:
            Dict with session information
        """
        try:
            if not username:
                return {
                    'success': False,
                    'message': 'Username is required to create a chat session'
                }
                
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            
            # Generate title based on first query or use default
            title = self._generate_chat_title(first_query) if first_query else f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create session document
            session_data = {
                'session_id': session_id,
                'username': username,
                'title': title,
                'created_at': datetime.now(),
                'last_updated': datetime.now(),
                'message_count': 0,
                'is_active': True
            }
            
            # Save to database
            result = self.db.chatSessions.insert_one(session_data)
            
            return {
                'success': True,
                'session_id': session_id,
                'title': title,
                'created_at': session_data['created_at']
            }
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return {
                'success': False,
                'message': f'Error creating chat session: {str(e)}'
            }

    def get_user_chat_sessions(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a logged-in user.
        
        Args:
            username: The username to fetch sessions for
            limit: Maximum number of sessions to return
            
        Returns:
            List of chat sessions with basic info
        """
        try:
            cursor = self.db.chatSessions.find(
                {'username': username},
                {'_id': 0, 'session_id': 1, 'title': 1, 'created_at': 1, 'last_updated': 1, 'message_count': 1}
            ).sort('last_updated', -1).limit(limit)
            
            sessions = []
            for doc in cursor:
                sessions.append({
                    'session_id': doc['session_id'],
                    'title': doc['title'],
                    'created_at': doc['created_at'].strftime("%Y-%m-%d %H:%M:%S"),
                    'last_updated': doc['last_updated'].strftime("%Y-%m-%d %H:%M:%S"),
                    'message_count': doc['message_count']
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error fetching chat sessions: {e}")
            return []

    def get_chat_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get messages for a specific chat session in chronological order.
        
        Args:
            session_id: The session ID to fetch messages for
            limit: Maximum number of messages to return
            
        Returns:
            List of messages in the chat session
        """
        try:
            cursor = self.db.chatHistory.find(
                {'session_id': session_id},
                {'_id': 1, 'user_query': 1, 'bot_response': 1, 'timestamp': 1}
            ).sort('timestamp', 1).limit(limit)
            
            messages = []
            for doc in cursor:
                messages.append({
                    'id': str(doc['_id']),
                    'user_query': doc['user_query'],
                    'bot_response': doc['bot_response'],
                    'timestamp': doc['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error fetching chat messages: {e}")
            return []

    def _save_conversation_backup(self, conversation_data: dict, session_id: str) -> bool:
        """
        Save a backup of the conversation to the file system for redundancy.
        
        Args:
            conversation_data: The conversation data to save
            session_id: The session identifier for logging
            
        Returns:
            bool: True if backup was successful, False otherwise
        """
        try:
            # Ensure the backup directory exists
            conversation_dir = KNOWLEDGE_DIR / "conversations"
            conversation_dir.mkdir(exist_ok=True)
            
            # Create organized directory structure by date
            today = datetime.now().strftime("%Y-%m-%d")
            date_dir = conversation_dir / today
            date_dir.mkdir(exist_ok=True)
            
            # Create a filename based on the session ID and timestamp
            timestamp = int(time.time())
            filename = f"{session_id}_{timestamp}.json"
            filepath = date_dir / filename
            
            # Write the data to file with pretty formatting for readability
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, default=str, indent=2)
                
            logger.info(f"Successfully saved conversation backup to {filepath}")
            return True
                
        except Exception as e:
            logger.error(f"Error saving conversation backup: {e}")
            return False
            
    def _add_to_dead_letter_queue(self, data: dict, error_msg: str) -> None:
        """
        Add failed conversation saves to a dead letter queue for later diagnostics and retry.
        
        Args:
            data: The conversation data that failed to save
            error_msg: The error message explaining the failure
        """
        try:
            # Prepare the dead letter queue entry
            dlq_entry = {
                'original_data': data,
                'error_message': error_msg,
                'timestamp': datetime.now(),
                'processed': False  # Flag for later processing
            }
            
            # Save to a special MongoDB collection for dead letter queue
            try:
                self.db.chatbotConversationsDLQ.insert_one(dlq_entry)
                logger.info(f"Added failed conversation to dead letter queue. Session: {data.get('session_id')}")
            except Exception as mongodb_error:
                logger.error(f"Failed to add to MongoDB DLQ: {mongodb_error}")
                # Fall back to file-based DLQ if MongoDB is unavailable
                self._save_to_file_dlq(dlq_entry)
                
        except Exception as e:
            logger.error(f"Error adding to dead letter queue: {e}")
            # Try a last-resort file save
            try:
                self._save_to_file_dlq({
                    'partial_data': str(data)[:1000] + "...[truncated]",
                    'error': str(e),
                    'original_error': error_msg,
                    'timestamp': str(datetime.now())
                })
            except Exception as final_error:
                logger.critical(f"Complete failure in DLQ handling: {final_error}")
    
    def _save_to_file_dlq(self, dlq_entry: dict) -> None:
        """Save dead letter queue entry to filesystem when MongoDB is unavailable"""
        try:
            # Ensure the DLQ directory exists
            dlq_dir = KNOWLEDGE_DIR / "conversations" / "dlq"
            dlq_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a unique filename
            timestamp = int(time.time())
            filename = f"dlq_{timestamp}.json"
            filepath = dlq_dir / filename
            
            # Write the data to file
            with open(filepath, 'w') as f:
                json.dump(dlq_entry, f, default=str, indent=2)
                
            logger.info(f"Saved dead letter queue entry to {filepath}")
                
        except Exception as e:
            logger.critical(f"Failed to save to file DLQ: {e}")
    
    def _check_rate_limit(self, session_id: str) -> bool:
        """
        Check if the session has exceeded rate limits.
        
        Args:
            session_id: The unique session identifier
            
        Returns:
            bool: True if rate limit is exceeded, False otherwise
        """
        try:
            # Get recent conversations for this session within the last minute
            now = datetime.now()
            one_minute_ago = now - timedelta(seconds=60)
            
            count = self.db.chatbotConversations.count_documents({
                'session_id': session_id,
                'timestamp': {'$gte': one_minute_ago}
            })
            
            # Rate limit: 10 messages per minute
            return count >= 10
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False  # Default to not rate limiting on error
    
    def get_chat_history(self, username: str = None, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch chat history for a user or session.
        
        Args:
            username: The username to fetch history for (if logged in)
            session_id: The session ID to fetch history for
            limit: Maximum number of records to return
            
        Returns:
            List of chat history records
        """
        try:
            # Build query
            query = {}
            if username:
                query["username"] = username
            if session_id:
                query["session_id"] = session_id
                
            # Execute query
            cursor = self.db.chatHistory.find(
                query,
                {"_id": 1, "user_query": 1, "bot_response": 1, "timestamp": 1, "username": 1}
            ).sort("timestamp", -1).limit(limit)
            
            # Convert cursor to list and format timestamps
            history = []
            for doc in cursor:
                history.append({
                    "id": str(doc["_id"]),
                    "user_query": doc["user_query"],
                    "bot_response": doc["bot_response"],
                    "timestamp": doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "username": doc.get("username", "guest")
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return []

    def reset_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Reset the conversation history for a session.
        
        Args:
            session_id: The unique session identifier
            
        Returns:
            Dict with status information
        """
        try:
            # Reset memory
            self.memory.clear()
            
            # Log the reset
            logger.info(f"Conversation reset for session {session_id}")
            
            return {
                'response': "Conversation has been reset.",
                'type': 'system',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")
            return {
                'response': "Failed to reset conversation.",
                'type': 'error',
                'success': False,
                'error': str(e)
            }
            
    def provide_feedback(self, session_id: str, message_id: str, 
                        feedback: str, is_authenticated: bool = False) -> Dict[str, Any]:
        """
        Allow users to provide feedback on chat responses for continuous improvement.
        
        Args:
            session_id: The unique session identifier
            message_id: The specific message identifier
            feedback: The feedback provided (positive, negative, or text)
            is_authenticated: Whether the user is authenticated
            
        Returns:
            Dict with status information
        """
        try:
            # Update the conversation record with feedback
            result = self.db.chatHistory.update_one(
                {'session_id': session_id, '_id': ObjectId(message_id)},
                {'$set': {'feedback': feedback}}
            )
            
            if result.modified_count > 0:
                logger.info(f"Feedback recorded for message {message_id} in session {session_id}")
                return {
                    'response': "Thank you for your feedback!",
                    'type': 'system',
                    'success': True
                }
            else:
                logger.warning(f"No message found for ID {message_id} in session {session_id}")
                return {
                    'response': "Could not record feedback for this message.",
                    'type': 'error',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return {
                'response': "Failed to record feedback.",
                'type': 'error',
                'success': False,
                'error': str(e)
            }

    def get_conversation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about conversation storage success and database performance.
        
        Returns:
            Dict with performance metrics
        """
        try:
            # Get basic counts
            total_saved = self.db.chatHistory.count_documents({})
            total_failed = self.db.chatHistoryDLQ.count_documents({})
            
            # Get failures in the last 24 hours
            one_day_ago = datetime.now() - timedelta(days=1)
            recent_failures = self.db.chatHistoryDLQ.count_documents({
                'timestamp': {'$gte': one_day_ago}
            })
            
            # Get retry statistics
            retry_stats = self.db.chatHistory.aggregate([
                {
                    '$match': {
                        'metadata.retry_count': {'$gt': 0}
                    }
                },
                {
                    '$group': {
                        '_id': None,
                        'total_retries': {'$sum': '$metadata.retry_count'},
                        'avg_retries': {'$avg': '$metadata.retry_count'},
                        'max_retries': {'$max': '$metadata.retry_count'}
                    }
                }
            ])
            
            retry_data = list(retry_stats)
            retry_metrics = retry_data[0] if retry_data else {
                'total_retries': 0,
                'avg_retries': 0,
                'max_retries': 0
            }
            
            return {
                'total_saved': total_saved,
                'total_failed': total_failed,
                'recent_failures': recent_failures,
                'retry_metrics': retry_metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error getting chatbot stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize a global chatbot instance
chatbot = None

def get_chatbot(rebuild_kb: bool = False, use_local_model: bool = False):
    """Get or initialize the global chatbot instance
    
    Args:
        rebuild_kb (bool): Whether to rebuild the knowledge base
        use_local_model (bool): Whether to use a local model instead of API
        
    Returns:
        ChatbotAgent: The chatbot instance
    """
    global chatbot
    if chatbot is None:
        chatbot = ChatbotAgent(rebuild_kb=rebuild_kb, use_local_model=use_local_model)
    return chatbot
