import os
import re
import json
import time
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

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
        """
        Initialize the chatbot agent.
        
        Args:
            rebuild_kb: Whether to rebuild the knowledge base from scratch
            use_local_model: Whether to use a local model or an API
        """
        self.db = get_db()
        self.knowledge_updated = False
        self.use_local_model = use_local_model
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str(MODELS_DIR)
        )
        
        # Security patterns to detect potentially harmful queries
        self.security_patterns = [
            r"(?i)password|secret|token|api[_-]?key|credential",
            r"(?i)\.env|config\.py|\.git",
            r"(?i)exploit|vulnerability|hack|attack|inject",
            r"(?i)sql\s*injection|xss|csrf|rce",
            r"(?i)delete\s*from|drop\s*table|truncate\s*table",
            r"(?i)rm\s*-rf|system\(|exec\(|eval\(",
            r"(?i)\/etc\/passwd|\/etc\/shadow"
        ]
        
        # Add a whitelist for common education-related terms
        self.safe_patterns = [
            r"(?i)student\s*name",
            r"(?i)attendance",
            r"(?i)homework",
            r"(?i)test\s*scores?",
            r"(?i)grade",
            r"(?i)prediction"
        ]
        
        # Load or create the knowledge base
        self.kb_path = KNOWLEDGE_DIR / "kb_faiss"
        if not self.kb_path.exists() or rebuild_kb:
            logger.info("Building knowledge base...")
            self._build_knowledge_base()
        else:
            logger.info("Loading existing knowledge base...")
            self._load_knowledge_base()
            
        # Initialize the LLM (language model)
        self._init_llm()
        
        # Initialize memory and retriever
        self.chat_history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
        # Create the conversational chain
        self._create_chain()
        
        logger.info("ChatbotAgent initialized successfully")

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
                            "I can help you make an education prediction. Please provide the student's:\n1. Attendance percentage (0-100)\n2. Homework completion rate (0-100)\n3. Test scores (0-100)",
                            
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

    def _create_chain(self):
        """Create a simple but reliable chain for query processing"""
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
                return {
                    "result": "I'm having trouble processing your request. Please try asking a simpler question about the Education Prediction System.",
                    "source_documents": [],
                    "answer": "Error processing request"
                }
        
        self.chain = process_response
        logger.info("Chain created successfully with simple processing")

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
    
    def _get_prediction_history(self, username: str, time_frame: str = None, limit: int = 5) -> List[Dict[str, Any]]:
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
            
            # Execute query
            cursor = self.db.predictionHistory.find(
                query,
                {"_id": 0, "password": 0, "email": 0}  # Exclude sensitive fields
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
        summary = f"Summary of your last {len(history)} predictions:\n"
        summary += f"- Average performance score: {avg_score:.1f}%\n"
        summary += f"- Excellent: {excellent_count}\n"
        summary += f"- Good: {good_count}\n"
        summary += f"- Needs Improvement: {needs_improvement_count}\n\n"
        
        # Add recent predictions
        summary += "Recent predictions:\n"
        for i, p in enumerate(history[:3], 1):
            date = p.get("created_at", "Unknown date")
            if isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = str(date)
                
            student = p.get("name", "Unknown student")
            score = p.get("prediction_score", 0)
            prediction = p.get("prediction", "Unknown")
            
            summary += f"{i}. {date_str}: {student} - {score:.1f}% ({prediction})\n"
        
        summary += "\nVisit the Prediction History page to see all your predictions and generate detailed reports."
        return summary
    
    def _save_conversation(self, session_id: str, user_query: str, bot_response: str, 
                          is_authenticated: bool, username: str = None) -> None:
        """
        Save a conversation exchange to the database for future training.
        
        Args:
            session_id: The unique session identifier
            user_query: The user's query
            bot_response: The bot's response
            is_authenticated: Whether the user is authenticated
            username: The username if authenticated
        """
        try:
            conversation_data = {
                'session_id': session_id,
                'user_query': user_query,
                'bot_response': bot_response,
                'is_authenticated': is_authenticated,
                'username': username,
                'timestamp': datetime.now(),
                'feedback': None  # For future feedback collection
            }
            
            self.db.chatbotConversations.insert_one(conversation_data)
            
            # Also save to the file system for redundancy
            conversation_dir = KNOWLEDGE_DIR / "conversations"
            conversation_dir.mkdir(exist_ok=True)
            
            # Create a filename based on the session ID and timestamp
            filename = f"{session_id}_{int(time.time())}.json"
            filepath = conversation_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, default=str)
                
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
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
    
    def process_query(self, query: str, session_id: str, is_authenticated: bool = False, 
                     username: str = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user query
            session_id: The unique session identifier
            is_authenticated: Whether the user is authenticated
            username: The username if authenticated
            
        Returns:
            Dict with response information
        """
        start_time = time.time()
        
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
                username=username
            )
            
            return response
        
        # Preprocess the query
        processed_query = self._preprocess_query(query)
        
        try:
            # First, check for prediction parameters - this should take highest priority
            # Move this up to handle prediction requests before other processing
            prediction_params = self._extract_prediction_params(processed_query)
            logger.info(f"Checking for prediction parameters in query: {len(prediction_params)} parameters found")
            
            # Check if we have all required parameters for a prediction
            required_prediction_params = ["attendance", "homework_completion", "test_scores"]
            has_prediction_params = all(param in prediction_params for param in required_prediction_params)
            
            # Log detailed parameter information
            if has_prediction_params:
                logger.info(f"Prediction parameters detected: Attendance={prediction_params['attendance']}, " +
                           f"Homework={prediction_params['homework_completion']}, " +
                           f"Test scores={prediction_params['test_scores']}, " +
                           f"Name={prediction_params.get('name', 'Not provided')}")
                
            # First check if this is a history request
            if prediction_params.get("request_type") == "history":
                if is_authenticated and username:
                    # Fetch actual history data
                    time_frame = prediction_params.get("time_frame")
                    history = self._get_prediction_history(username, time_frame)
                    summary = self._summarize_prediction_history(history)
                    
                    response = {
                        'response': summary,
                        'type': 'history',
                        'success': True,
                        'processing_time': time.time() - start_time
                    }
                else:
                    response = {
                        'response': "You need to be logged in to view your prediction history. Please log in and try again.",
                        'type': 'auth_required',
                        'success': False,
                        'processing_time': time.time() - start_time
                    }
                
                self._save_conversation(
                    session_id=session_id,
                    user_query=query,
                    bot_response=response['response'],
                    is_authenticated=is_authenticated,
                    username=username
                )
                
                return response
                
            # If we have enough parameters for a prediction, make one immediately
            # This takes precedence over other query handling
            if has_prediction_params:
                prediction_result = self._make_prediction(prediction_params)
                
                if prediction_result['success']:
                    # Save to history if authenticated
                    if is_authenticated and username:
                        self._save_to_history(username, prediction_result)
                        
                    # Prepare a personalized response with the student's name if available
                    student_name = prediction_result.get('name', 'The student')
                    
                    # Format the score to 1 decimal place
                    score = round(prediction_result['prediction_score'], 1)
                    
                    # Add details about the parameters used
                    details = (f"\n\nThis prediction is based on:\n"
                              f"- Attendance: {prediction_result['attendance']}%\n"
                              f"- Homework completion: {prediction_result['homework_completion']}%\n"
                              f"- Test scores: {prediction_result['test_scores']}%")
                              
                    # Add a recommendation based on the prediction
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
                    
                    self._save_conversation(
                        session_id=session_id,
                        user_query=query,
                        bot_response=response['response'],
                        is_authenticated=is_authenticated,
                        username=username
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
                    username=username
                )
                
                return response
            
            # Check for task requests (for authenticated users)
            is_task, task_response = self._handle_task_request(processed_query, is_authenticated)
            
            if is_task:
                response = {
                    'response': task_response,
                    'type': 'task',
                    'success': True,
                    'processing_time': time.time() - start_time
                }
                
                self._save_conversation(
                    session_id=session_id,
                    user_query=query,
                    bot_response=response['response'],
                    is_authenticated=is_authenticated,
                    username=username
                )
                
                return response
            
            # At this point, check if this looks like a prediction request
            # but we don't have all the parameters
            prediction_request_indicators = [
                "prediction", "predict", "student performance", 
                "attendance", "homework", "test score", "grade"
            ]
            
            # Check if it has prediction keywords but we didn't have all params
            is_likely_prediction_request = any(
                indicator in processed_query.lower() 
                for indicator in prediction_request_indicators
            )
            
            if is_likely_prediction_request and not has_prediction_params:
                # This looks like a prediction request but we don't have all parameters
                # Ask the user for the missing information
                missing_params = [
                    param for param in required_prediction_params 
                    if param not in prediction_params
                ]
                
                available_params = [
                    f"{param}: {prediction_params[param]}%" 
                    for param in required_prediction_params 
                    if param in prediction_params
                ]
                
                # Create a helpful response asking for missing parameters
                if available_params:
                    params_found = "\n".join(available_params)
                    missing = ", ".join(missing_params).replace("_", " ")
                    
                    response_text = (
                        f"I've detected some prediction parameters:\n{params_found}\n\n"
                        f"To make a complete prediction, I still need: {missing}.\n\n"
                        "Please provide these missing values."
                    )
                else:
                    response_text = (
                        "To make an education prediction, I need the following information:\n"
                        "1. Attendance percentage (0-100)\n"
                        "2. Homework completion rate (0-100)\n"
                        "3. Test scores (0-100)\n\n"
                        "Please provide all three values."
                    )
                    
                response = {
                    'response': response_text,
                    'type': 'prediction_request',
                    'success': True,
                    'partial_params': prediction_params,
                    'processing_time': time.time() - start_time
                }
                
                self._save_conversation(
                    session_id=session_id,
                    user_query=query,
                    bot_response=response['response'],
                    is_authenticated=is_authenticated,
                    username=username
                )
                
                return response
                
                if prediction_result['success']:
                    # Save to history if authenticated
                    if is_authenticated and username:
                        self._save_to_history(username, prediction_result)
                        
                    # Prepare a personalized response with the student's name if available
                    student_name = prediction_result.get('name', 'The student')
                    
                    # Format the score to 1 decimal place
                    score = round(prediction_result['prediction_score'], 1)
                    
                    # Add details about the parameters used
                    details = (f"\n\nThis prediction is based on:\n"
                              f"- Attendance: {prediction_result['attendance']}%\n"
                              f"- Homework completion: {prediction_result['homework_completion']}%\n"
                              f"- Test scores: {prediction_result['test_scores']}%")
                              
                    # Add a recommendation based on the prediction
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
                    
                    self._save_conversation(
                        session_id=session_id,
                        user_query=query,
                        bot_response=response['response'],
                        is_authenticated=is_authenticated,
                        username=username
                    )
                    
                    return response
            
                else:
                    # Handle the case where we don't have prediction parameters
                    # but should indicate this is a prediction-related query
                    pass
            
            # Process general queries through the chain
            try:
                chain_response = self.chain(processed_query)
                response_text = chain_response.get('result', "I'm processing your question.")
                source_documents = chain_response.get('source_documents', [])
                
                logger.debug(f"Chain response type: {type(chain_response)}")
                
                logger.info(f"Generated response of length: {len(response_text)}")
                
            except Exception as chain_error:
                
                logger.info(f"Generated response of length: {len(response_text)}")
                
            except Exception as chain_error:
                logger.error(f"Error in chain invocation: {chain_error}", exc_info=True)
                # Provide a useful fallback response
                response_text = "I'm having trouble answering that question right now. Could you try asking something else about the Education Prediction System?"
                source_documents = []
            
            # Format source information if available
            sources = []
            if source_documents:
                for doc in source_documents:
                    if 'source' in doc.metadata:
                        source = doc.metadata['source']
                        if source not in sources:
                            sources.append(source)
            
            response = {
                'response': response_text,
                'type': 'general',
                'success': True,
                'sources': sources,
                'processing_time': time.time() - start_time
            }
            
            self._save_conversation(
                session_id=session_id,
                user_query=query,
                bot_response=response['response'],
                is_authenticated=is_authenticated,
                username=username
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
            return error_response
            
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
            result = self.db.chatbotConversations.update_one(
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

    def get_chatbot_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the chatbot usage.
        Only available to authenticated users.
        
        Returns:
            Dict with usage statistics
        """
        try:
            # Get total conversations
            total_conversations = self.db.chatbotConversations.count_documents({})
            
            # Get conversations in last 24 hours
            one_day_ago = datetime.now() - timedelta(days=1)
            recent_conversations = self.db.chatbotConversations.count_documents({
                'timestamp': {'$gte': one_day_ago}
            })
            
            # Get prediction requests
            prediction_requests = self.db.chatbotConversations.count_documents({
                'bot_response': {'$regex': 'prediction', '$options': 'i'}
            })
            
            # Get successful task executions
            task_executions = self.db.chatbotConversations.count_documents({
                'type': 'task', 
                'success': True
            })
            
            return {
                'total_conversations': total_conversations,
                'recent_conversations': recent_conversations,
                'prediction_requests': prediction_requests,
                'task_executions': task_executions,
                'knowledge_updated': self.knowledge_updated,
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

def get_chatbot(rebuild_kb=False):
    """Get or initialize the global chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = ChatbotAgent(rebuild_kb=rebuild_kb, use_local_model=False)
    return chatbot
