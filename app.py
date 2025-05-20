import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, session as flask_session, flash, send_file, jsonify, Response
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
from fpdf import FPDF
from config import get_db
from functools import wraps
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from bson import ObjectId, json_util
from utils.chatbot import get_chatbot
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app)

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data.get('email', '')

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = db.users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
    except Exception as e:
        app.logger.error(f"Error loading user: {e}")
    return None

# Initialize chatbot
chatbot = get_chatbot(rebuild_kb=False)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
db = get_db()

@app.route('/')
def home():
    """Public home page accessible without login"""
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users = db.users
        username = request.form['username']
        existing_user = users.find_one({"username": username})

        if existing_user is None:
            hashed_password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
            user = {
                "username": username,
                "email": request.form['email'],
                "password": hashed_password
            }
            users.insert_one(user)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists!', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    # Redirect if user is already logged in
    if current_user.is_authenticated:
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('index')
        return redirect(next_page)
        
    if request.method == 'POST':
        users = db.users
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            user_data = users.find_one({"username": username})
            if user_data and check_password_hash(user_data['password'], password):
                user = User(user_data)
                login_user(user, remember=True)  # Enable remember me functionality
                flask_session['username'] = username  # Store username in session
                
                # Get the next page from the request args
                next_page = request.args.get('next')
                if not next_page or not next_page.startswith('/'):
                    next_page = url_for('index')
                    
                flash('Login successful!', 'success')
                return redirect(next_page)
            flash('Invalid credentials, please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Enhanced debug logging
        app.logger.info("==== PREDICTION FORM SUBMISSION STARTED ====")
        app.logger.info(f"Form data received: {request.form}")
        app.logger.info(f"Current user: {current_user.username if current_user.is_authenticated else 'Not authenticated'}")
        app.logger.info(f"User ID: {current_user.id if current_user.is_authenticated else 'None'}")
        try:
            session_data = dict(flask_session)
            app.logger.info(f"Session data: {session_data}")
        except Exception as sess_err:
            app.logger.warning(f"Could not log session data: {sess_err}")
        app.logger.info(f"Request method: {request.method}")
        app.logger.info(f"Request path: {request.path}")
        
        # Detailed authentication logging
        app.logger.info(f"Authentication status:")
        app.logger.info(f"  - current_user.is_authenticated: {current_user.is_authenticated}")
        app.logger.info(f"  - 'username' in session: {'username' in flask_session}")
        app.logger.info(f"  - session.get('user_id'): {flask_session.get('user_id')}")
        app.logger.info(f"  - session.get('_id'): {flask_session.get('_id')}")
        
        # Verify MongoDB connection from config (single consistent connection)
        try:
            # Use the get_db function from config
            app.logger.info("Getting database connection from config.py...")
            db = get_db()
            
            # Test connection
            db.client.admin.command('ping')
            app.logger.info("MongoDB connection verified: Connection to Atlas is active")
            
            # Check collections
            collections = db.list_collection_names()
            app.logger.info(f"Available collections: {collections}")
        except Exception as db_err:
            app.logger.error(f"CRITICAL: MongoDB connection failed: {db_err}")
            app.logger.error(f"Error type: {type(db_err).__name__}")
            flash("Database connection error. Please try again later.", "danger")
            return redirect(url_for('index'))
        
        # Check if user is authenticated with detailed logging
        if not current_user.is_authenticated:
            app.logger.error("User not authenticated despite @login_required decorator")
            app.logger.error(f"current_user object: {vars(current_user)}")
            flash("Please login to make predictions", 'danger')
            return redirect(url_for('login'))

        if 'username' not in flask_session:
            app.logger.error("Username not in session even though user is authenticated")
            app.logger.error(f"Session keys: {list(flask_session.keys())}")
            flash("Session error. Please log out and log in again.", 'danger')
            return redirect(url_for('login'))
            
        # Get and validate form data
        app.logger.info("Validating form data...")
        name = request.form['name'].strip()
        student_id = request.form['student_id'].strip()
        email = request.form['email'].strip()
        
        app.logger.info(f"Extracted text fields - Name: {name}, Student ID: {student_id}, Email: {email}")
        
        # Validate and cap numeric inputs
        try:
            app.logger.info(f"Raw numeric inputs - Attendance: '{request.form.get('attendance')}', " +
                          f"Homework: '{request.form.get('homework_completion')}', " +
                          f"Test Scores: '{request.form.get('test_scores')}'")
            
            attendance = min(max(float(request.form['attendance']), 0), 100)
            homework_completion = min(max(float(request.form['homework_completion']), 0), 100)
            test_scores = min(max(float(request.form['test_scores']), 0), 100)
            
            app.logger.info(f"Processed numeric inputs - Attendance: {attendance}, " +
                          f"Homework: {homework_completion}, Test Scores: {test_scores}")
            
            # Calculate capped percentage
            percentage = round((test_scores * 0.5) + (attendance * 0.3) + (homework_completion * 0.2), 2)
            percentage = min(percentage, 100)  # Ensure percentage never exceeds 100
            app.logger.info(f"Calculated percentage: {percentage}%")
        except ValueError as e:
            app.logger.error(f"Numeric validation error: {e}")
            app.logger.error(f"Raw form data for numeric fields: {request.form}")
            flash("Invalid input: Please enter numbers between 0 and 100", 'danger')
            return redirect(url_for('index'))
        
        # Determine prediction text based on percentage
        if percentage >= 80:
            prediction_text = "Excellent"
        elif percentage >= 60:
            prediction_text = "Good"
        else:
            prediction_text = "Needs Improvement"
        
        app.logger.info(f"Prediction category determined: {prediction_text}")
            
        # Legacy values for backward compatibility
        binary_prediction = 1 if percentage >= 60 else 0
        
        # Create prediction record with standardized fields
        app.logger.info("Preparing document for MongoDB insertion...")
        prediction_data = {
            # Student information
            'student_id': student_id,
            'name': name,
            'email': email,
            'attendance': attendance,
            'homework_completion': homework_completion,
            'test_scores': test_scores,
            
            # Prediction results
            'prediction': prediction_text,
            'prediction_score': percentage,
            'binary_prediction': binary_prediction,  # For model training compatibility
            'confidence': percentage / 100.0,  # Always store in 0-1 range for API consumption
            'probability': percentage,  # Always store as percentage (0-100) for template display
            
            # Metadata
            'username': current_user.username,
            'created_at': datetime.now(),
            'client_info': {
                'ip': request.remote_addr,
                'user_agent': request.user_agent.string,
                'platform': request.user_agent.platform,
                'browser': request.user_agent.browser,
            }
        }
        
        app.logger.info(f"Document prepared for insertion: {json.dumps(prediction_data, default=str)}")
        
        # Use a single database connection from config.py with transaction support
        try:
            # Get a reference to the database using the global connection
            app.logger.info("Using the database connection from config.py...")
            db = get_db()
            
            # Get a reference to the collection
            prediction_collection = db.predictionHistory
            
            # Start a session for transaction support
            app.logger.info("Starting MongoDB session for transaction...")
            with db.client.start_session() as session:
                # Start a transaction
                session.start_transaction()
                
                try:
                    # Insert the prediction with transaction support
                    app.logger.info("Attempting to insert data into predictionHistory with transaction...")
                    result = prediction_collection.insert_one(prediction_data, session=session)
                    
                    if result.acknowledged:
                        app.logger.info(f"✅ MongoDB insert SUCCESS: Document inserted with ID: {result.inserted_id}")
                        
                        # Verify the document was actually inserted
                        verification = prediction_collection.find_one(
                            {"_id": result.inserted_id}, 
                            session=session
                        )
                        
                        if verification:
                            app.logger.info(f"✅ Document verification SUCCESS: Document found in database")
                            # Compare fields to ensure data integrity
                            for key in ['student_id', 'name', 'email', 'prediction']:
                                if verification.get(key) != prediction_data.get(key):
                                    app.logger.warning(f"Data mismatch in {key}: {verification.get(key)} != {prediction_data.get(key)}")
                            
                            # Commit the transaction
                            session.commit_transaction()
                            app.logger.info("Transaction committed successfully")
                        else:
                            # Document verification failed, abort transaction
                            app.logger.error(f"❌ Document verification FAILED: Document not found after insertion")
                            session.abort_transaction()
                            app.logger.info("Transaction aborted due to verification failure")
                            flash("Database error: Document verification failed", 'danger')
                            return redirect(url_for('index'))
                    else:
                        # Insert not acknowledged, abort transaction
                        app.logger.error(f"❌ MongoDB insert FAILED: Insert not acknowledged by server")
                        session.abort_transaction()
                        app.logger.info("Transaction aborted due to unacknowledged insert")
                        flash("Database error: Insert not acknowledged", 'danger')
                        return redirect(url_for('index'))
                        
                except Exception as tx_error:
                    # Transaction failed, abort and log
                    app.logger.error(f"TRANSACTION ERROR: {str(tx_error)}")
                    app.logger.error(f"Error type: {type(tx_error).__name__}")
                    try:
                        session.abort_transaction()
                        app.logger.info("Transaction aborted due to error")
                    except Exception as abort_error:
                        app.logger.error(f"Failed to abort transaction: {str(abort_error)}")
                    
                    flash(f"Database transaction error: {str(tx_error)}", 'danger')
                    return redirect(url_for('index'))
                    
        except Exception as db_error:
            app.logger.error(f"DATABASE ERROR: Failed to insert prediction data: {db_error}")
            app.logger.error(f"Error type: {type(db_error).__name__}")
            app.logger.error(f"Error location: {db_error.__traceback__.tb_frame.f_code.co_filename}:{db_error.__traceback__.tb_lineno}")
            
            # Try to determine the specific error type
            error_message = str(db_error)
            if "not authorized" in error_message.lower():
                flash("Database permission error: Not authorized to write data", 'danger')
            elif "connection" in error_message.lower():
                flash("Database connection error: Could not connect to MongoDB Atlas", 'danger')
            else:
                flash(f"Database error: {str(db_error)}", 'danger')
                
            return redirect(url_for('index'))
        
        # Store in session for PDF generation
        app.logger.info("Storing data in session for PDF generation")
        session_data = {k: v for k, v in prediction_data.items() if not isinstance(v, datetime)}
        if '_id' in session_data:
            session_data['_id'] = str(session_data['_id'])
        flask_session['student_data'] = session_data
    
        # Determine prediction message based on text
        if prediction_text == "Excellent":
            prediction_message = "Excellent Result! The student is showing outstanding performance."
        elif prediction_text == "Good":
            prediction_message = "Good Result! The student is likely to perform well."
        else:
            prediction_message = "The student may need additional support based on current indicators."

        # --- START: Retrain and save the model ---
        history = list(db.predictionHistory.find({}))

        if len(history) >= 5:  # Minimum data to avoid crash
            X = np.array([[s['attendance'], s['homework_completion'], s['test_scores']] for s in history])
            y = np.array([s.get('binary_prediction', 1 if s.get('prediction', '').lower() in ['excellent', 'good'] else 0) for s in history])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression()
            model.fit(X_scaled, y)
          
            with open('./models/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                print("Scaler.pkl updated successfully.")

            with open('./models/best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                print("Best_model.pkl updated successfully.")
        # --- END: Retrain and save the model ---

        app.logger.info(f"Prediction complete: {prediction_text}, Score: {percentage}%")
        app.logger.info("==== PREDICTION FORM SUBMISSION COMPLETED SUCCESSFULLY ====")
        app.logger.info("Rendering result template...")
        app.logger.info("==== PREDICTION FORM SUBMISSION COMPLETED SUCCESSFULLY ====")
        return render_template('result.html', prediction=prediction_text, probability=percentage, prediction_message=prediction_message, student=prediction_data)

    except Exception as e:
        app.logger.error(f"PREDICTION ERROR: {str(e)}")
        app.logger.error(f"Error type: {type(e).__name__}")
        app.logger.error(f"Error location: {str(e.__traceback__.tb_frame.f_code.co_filename)} line {e.__traceback__.tb_lineno}")
        app.logger.error(f"Request form data: {request.form}")
        try:
            session_data = dict(flask_session)
            app.logger.error(f"Session data: {session_data}")
        except Exception as sess_err:
            app.logger.warning(f"Could not log session data in error handler: {sess_err}")
        
        # Try to log MongoDB status
        try:
            mongodb_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            mongodb_client.admin.command('ping')
            app.logger.info("MongoDB connection is still active despite the error")
        except Exception as db_err:
            app.logger.error(f"MongoDB connection failed during error handling: {db_err}")
            
        app.logger.error("==== PREDICTION FORM SUBMISSION FAILED ====")
        flash(f"Prediction failed: {e}", 'danger')
        return redirect(url_for('index'))

# PDF Report Generation
@app.route('/report/<student_id>', methods=['GET'])
@login_required
def report(student_id):
    student = flask_session.get('student_data')
    
    if student and student['student_id'] == student_id:
        try:
            # Use the text-based prediction directly
            prediction = student['prediction']
            probability = student['probability']
            percentage = round((student['test_scores'] * 0.5) + (student['attendance'] * 0.3) + (student['homework_completion'] * 0.2), 2)

       # ... inside your report() function, replace the PDF generation section with:

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Header Bar
            pdf.set_fill_color(255, 65, 108)  # #ff416c
            pdf.rect(0, 0, 210, 20, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "STUDENT APPRAISAL REPORT", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, "MONTH OF MAY 2025", ln=True, align='C')
            pdf.ln(5)

            # Reset text color for body
            pdf.set_text_color(40, 40, 40)

            # Student Details Section
            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 10, "Student Details", ln=True)
            pdf.set_draw_color(255, 65, 108)
            pdf.set_line_width(0.8)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Student Name:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, student['name'], ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Enrollment:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, student['student_id'], ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Email:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, student['email'], ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Attendance:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, f"{student['attendance']}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Homework Completion:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, f"{student['homework_completion']}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Test Scores:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, f"{student['test_scores']}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Overall Percentage:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, f"{percentage}%", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Faculty:", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, "Aseef Ahmed", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(50, 10, "Coordinator", ln=False)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 10, "Maham Haider", ln=True)

            # Prediction and Remarks Section
            pdf.ln(5)
            pdf.set_draw_color(255, 65, 108)
            pdf.set_line_width(0.8)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 10, "Remarks", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.set_text_color(255, 65, 108)
            pdf.cell(0, 10, f"{prediction}", ln=True)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 10, f"Confidence: {probability}%", ln=True)
            pdf.ln(10)

            # Footer Bar
            pdf.set_y(-25)
            pdf.set_fill_color(255, 65, 108)
            pdf.rect(0, pdf.get_y(), 210, 20, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 10, "(92-21) 36630102-3 | info@aptechnn.com | NORTH NAZIMABAD KARACHI-PAKISTAN", ln=True, align='C')

            # Save the PDF
            pdf_file = f"report_{student['student_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            pdf.output(pdf_file)

            # Ensure the PDF file exists before sending it
            if os.path.exists(pdf_file):
                return send_file(pdf_file, as_attachment=True)
            else:
                flash('Error generating the PDF report', 'danger')
                return redirect(url_for('index'))

        except Exception as e:
            flash(f"Error generating PDF: {e}", 'danger')
            return redirect(url_for('index'))
    else:
        flash('Student not found!', 'danger')
        return redirect(url_for('index'))

# Add new route for prediction history
@app.route('/prediction-history')
@login_required
def prediction_history():
    # Get all predictions for the current user with required fields
    query = {
        'username': current_user.username,
        "student_id": {"$exists": True, "$ne": ""},
        "name": {"$exists": True, "$ne": ""},
        "email": {"$exists": True, "$ne": ""}
    }
    
    # Sort by most recent first - use created_at or timestamp
    sort_order = [('created_at', -1)]
    
    # Fetch raw predictions from MongoDB
    raw_predictions = list(db.predictionHistory.find(query).sort(sort_order))
    
    # Process predictions for display
    processed_predictions = []
    for pred in raw_predictions:
        try:
            # Create a standardized prediction object
            processed_pred = {
                'student_id': pred.get('student_id', ''),
                'name': pred.get('name', ''),
                'email': pred.get('email', '')
            }
            
            # Process timestamp for display
            if 'created_at' in pred and pred['created_at']:
                if isinstance(pred['created_at'], datetime):
                    processed_pred['timestamp'] = pred['created_at']
                else:
                    try:
                        processed_pred['timestamp'] = datetime.fromisoformat(str(pred['created_at']))
                    except (ValueError, TypeError):
                        processed_pred['timestamp'] = datetime.now()
            elif 'timestamp' in pred and pred['timestamp']:
                if isinstance(pred['timestamp'], datetime):
                    processed_pred['timestamp'] = pred['timestamp']
                else:
                    try:
                        processed_pred['timestamp'] = datetime.fromisoformat(str(pred['timestamp']))
                    except (ValueError, TypeError):
                        processed_pred['timestamp'] = datetime.now()
            else:
                processed_pred['timestamp'] = datetime.now()
            
            # Process numeric fields with validation and capping
            try:
                # Cap all numeric values at 100
                processed_pred['attendance'] = min(max(float(pred.get('attendance', 0)), 0), 100)
                processed_pred['homework_completion'] = min(max(float(pred.get('homework_completion', 0)), 0), 100)
                processed_pred['test_scores'] = min(max(float(pred.get('test_scores', 0)), 0), 100)
                
                # Calculate prediction score if not already present
                if 'prediction_score' in pred and pred['prediction_score'] is not None:
                    processed_pred['prediction_score'] = min(float(pred['prediction_score']), 100)
                else:
                    # Calculate from components
                    processed_pred['prediction_score'] = round((
                        (processed_pred['test_scores'] * 0.5) + 
                        (processed_pred['attendance'] * 0.3) + 
                        (processed_pred['homework_completion'] * 0.2)
                    ), 2)
                
                # Ensure prediction_score never exceeds 100
                processed_pred['prediction_score'] = min(processed_pred['prediction_score'], 100)
                
                # Get probability value directly from prediction_score
                processed_pred['probability'] = processed_pred['prediction_score']
                
                # Simplified prediction text standardization
                if 'prediction' in pred and isinstance(pred['prediction'], str) and pred['prediction'].strip():
                    # Convert to title case for consistency
                    pred_text = pred['prediction'].lower().strip()
                    
                    # Map to standard categories
                    if 'excellent' in pred_text:
                        processed_pred['prediction'] = 'Excellent'
                    elif 'good' in pred_text:
                        processed_pred['prediction'] = 'Good'
                    elif 'needs improvement' in pred_text or 'improvement' in pred_text:
                        processed_pred['prediction'] = 'Needs Improvement'
                    else:
                        # Use score to determine prediction if text doesn't match standard categories
                        score = processed_pred['prediction_score']
                        if score >= 80:
                            processed_pred['prediction'] = 'Excellent'
                        elif score >= 60:
                            processed_pred['prediction'] = 'Good'
                        else:
                            processed_pred['prediction'] = 'Needs Improvement'
                else:
                    # Determine prediction based on score
                    score = processed_pred['prediction_score']
                    if score >= 80:
                        processed_pred['prediction'] = 'Excellent'
                    elif score >= 60:
                        processed_pred['prediction'] = 'Good'
                    else:
                        processed_pred['prediction'] = 'Needs Improvement'
                
                processed_predictions.append(processed_pred)
                
            except (ValueError, TypeError) as e:
                app.logger.warning(f"Skipping entry with invalid numeric data: {e}")
                continue
                
        except Exception as e:
            app.logger.warning(f"Error processing prediction: {str(e)}")
            continue
    
    return render_template('prediction_history.html', predictions=processed_predictions)

@app.route('/analytics')
@login_required
def analytics():
    try:
        # Get predictions for the current user
        username = current_user.username  # Use current_user instead of session
        
        # Get all predictions for current user
        all_predictions = list(db.predictionHistory.find({
            "username": username
        }).sort("created_at", -1))
        
        # Initialize counters
        total_students = len(all_predictions)
        total_attendance = 0
        total_homework = 0
        total_test_scores = 0
        prediction_counts = {
            'Excellent': 0,
            'Good': 0,
            'Needs Improvement': 0
        }
        
        # Process predictions
        processed_predictions = []
        for pred in all_predictions:
            try:
                # Get numeric values with validation
                attendance = min(max(float(pred.get('attendance', 0)), 0), 100)
                homework = min(max(float(pred.get('homework_completion', 0)), 0), 100)
                test_scores = min(max(float(pred.get('test_scores', 0)), 0), 100)
                
                # Update totals
                total_attendance += attendance
                total_homework += homework
                total_test_scores += test_scores
                
                # Process prediction status
                prediction = pred.get('prediction', '')
                if isinstance(prediction, str):
                    pred_lower = prediction.lower()
                    if 'excellent' in pred_lower:
                        prediction_text = 'Excellent'
                    elif 'good' in pred_lower:
                        prediction_text = 'Good'
                    else:
                        prediction_text = 'Needs Improvement'
                else:
                    # Use score to determine prediction
                    score = (test_scores * 0.5) + (attendance * 0.3) + (homework * 0.2)
                    if score >= 80:
                        prediction_text = 'Excellent'
                    elif score >= 60:
                        prediction_text = 'Good'
                    else:
                        prediction_text = 'Needs Improvement'
                
                # Update prediction counts
                prediction_counts[prediction_text] += 1
                
                # Create processed prediction object
                processed_pred = {
                    'name': pred.get('name', 'N/A'),
                    'student_id': pred.get('student_id', 'N/A'),
                    'attendance': round(attendance, 1),
                    'homework_completion': round(homework, 1),
                    'test_scores': round(test_scores, 1),
                    'prediction': prediction_text,
                    'formatted_date': pred.get('created_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                processed_predictions.append(processed_pred)
                
            except Exception as e:
                app.logger.error(f"Error processing prediction: {str(e)}")
                continue
        
        # Calculate analytics data
        analytics_data = {
            'total_students': total_students,
            'good_predictions': prediction_counts['Excellent'] + prediction_counts['Good'],
            'needs_improvement': prediction_counts['Needs Improvement'],
            'avg_attendance': round(total_attendance / total_students if total_students > 0 else 0, 1),
            'avg_homework': round(total_homework / total_students if total_students > 0 else 0, 1),
            'avg_test_scores': round(total_test_scores / total_students if total_students > 0 else 0, 1),
            'prediction_counts': prediction_counts
        }
        
        return render_template(
            'student_analytics.html',
            analytics=analytics_data,
            predictions=processed_predictions,
            active_page='analytics'  # Add this for navbar highlighting
        )
        
    except Exception as e:
        app.logger.error(f"Failed to load analytics: {str(e)}")
        flash(f"Failed to load analytics: {str(e)}", 'danger')
        return render_template('student_analytics.html', error=str(e))

# Custom JSON encoder for MongoDB types
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(MongoJSONEncoder, self).default(obj)

# API endpoint for prediction history data
@app.route('/api/predictions')
@login_required
def api_predictions():
    """
    API endpoint to fetch paginated prediction history
    Query parameters:
      - skip: number of records to skip (default 0)
      - limit: maximum number of records to return (default 20)
    """
    try:
        skip = request.args.get('skip', default=0, type=int)
        limit = request.args.get('limit', default=20, type=int)
        
        # Enhanced query to ensure we get only valid entries with required fields
        query = {
            "student_id": {"$exists": True, "$ne": ""},
            "name": {"$exists": True, "$ne": ""},
            "email": {"$exists": True, "$ne": ""},
            "attendance": {"$exists": True, "$ne": None, "$type": ["number", "double"]},
            "homework_completion": {"$exists": True, "$ne": None, "$type": ["number", "double"]},
            "test_scores": {"$exists": True, "$ne": None, "$type": ["number", "double"]},
            "prediction": {"$exists": True, "$ne": ""},
            "username": current_user.username
        }
        
        # Sort by most recent first
        sort_order = [("created_at", -1)]
        
        # Fetch predictions from MongoDB
        predictions = list(db.predictionHistory.find(query).sort(sort_order).skip(skip).limit(limit))
        
        processed_predictions = []
        for pred in predictions:
            try:
                # Create a standardized prediction object
                processed_pred = {
                    'id': str(pred['_id']) if '_id' in pred else '',
                    'student_id': pred['student_id'],
                    'name': pred['name'],
                    'email': pred['email']
                }
                
                # Process timestamp
                if 'created_at' in pred and pred['created_at']:
                    processed_pred['timestamp'] = pred['created_at'].isoformat() if hasattr(pred['created_at'], 'isoformat') else str(pred['created_at'])
                else:
                    processed_pred['timestamp'] = datetime.now().isoformat()
                
                # Process numeric fields with validation
                try:
                    # Cap all numeric values at 100
                    attendance = min(max(float(pred['attendance']), 0), 100)
                    homework = min(max(float(pred['homework_completion']), 0), 100)
                    test_scores = min(max(float(pred['test_scores']), 0), 100)
                    
                    processed_pred['attendance'] = round(attendance, 1)
                    processed_pred['homework_completion'] = round(homework, 1)
                    processed_pred['test_scores'] = round(test_scores, 1)
                    
                    # Calculate prediction score using weighted formula
                    prediction_score = round(
                        (test_scores * 0.5) +  # Test scores weight: 50%
                        (attendance * 0.3) +   # Attendance weight: 30%
                        (homework * 0.2),      # Homework weight: 20%
                        1
                    )
                    
                    # Ensure prediction_score never exceeds 100
                    prediction_score = min(prediction_score, 100)
                    processed_pred['prediction_score'] = prediction_score
                    
                    # Calculate confidence (normalized to 0-1 range)
                    processed_pred['confidence'] = round(prediction_score / 100, 2)
                    
                    # Get probability value (as percentage)
                    processed_pred['probability'] = prediction_score
                    
                    # Determine prediction text based on score
                    if prediction_score >= 80:
                        processed_pred['prediction'] = 'Excellent'
                        processed_pred['grade'] = 'A'
                    elif prediction_score >= 70:
                        processed_pred['prediction'] = 'Good'
                        processed_pred['grade'] = 'B'
                    elif prediction_score >= 60:
                        processed_pred['prediction'] = 'Good'
                        processed_pred['grade'] = 'C'
                    else:
                        processed_pred['prediction'] = 'Needs Improvement'
                        processed_pred['grade'] = 'D'
                    
                    # Add detailed performance indicators
                    processed_pred['performance_indicators'] = {
                        'attendance_status': 'Good' if attendance >= 75 else 'Needs Improvement',
                        'homework_status': 'Good' if homework >= 70 else 'Needs Improvement',
                        'test_status': 'Good' if test_scores >= 60 else 'Needs Improvement'
                    }
                    
                    processed_predictions.append(processed_pred)
                    
                except (ValueError, TypeError) as e:
                    app.logger.warning(f"Skipping entry with invalid numeric data: {e}")
                    continue
                    
            except Exception as e:
                app.logger.warning(f"Error processing prediction: {str(e)}")
                continue
        
        return Response(
            json_util.dumps(processed_predictions),
            mimetype='application/json'
        )
    
    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Chat endpoints
@app.route('/chat')
def chat():
    """Render the chat interface."""
    session_id = request.args.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    
    return render_template('chat.html',
                         session_id=session_id,
                         is_authenticated=current_user.is_authenticated,
                         username=current_user.username if current_user.is_authenticated else None)

@app.route('/chat/authenticated')
@login_required
def authenticated_chat():
    """Authenticated chat endpoint with enhanced capabilities"""
    # Generate a unique session ID for this chat session if not already present
    if 'chat_session_id' not in flask_session:
        flask_session['chat_session_id'] = str(uuid.uuid4())
    
    return render_template('chat.html', 
                           is_authenticated=True,
                           username=current_user.username,
                           session_id=flask_session['chat_session_id'],
                           now=datetime.now())

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat requests (REST fallback if WebSockets not available)"""
    try:
        data = request.json
        query = data.get('message') or data.get('query', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not query:
            return jsonify({
                'response': "No message provided",
                'type': 'error',
                'success': False
            }), 400
        
        is_authenticated = current_user.is_authenticated
        username = current_user.username if is_authenticated else None
        
        # Process query through chatbot
        response = chatbot.process_query(
            query=query,
            session_id=session_id,
            is_authenticated=is_authenticated,
            username=username
        )
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Chat API Error: {str(e)}")
        return jsonify({
            'response': "I encountered an error while processing your request.",
            'type': 'error',
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/reset', methods=['POST'])
def reset_chat():
    """Reset the chat conversation history"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'response': "Missing session_id parameter",
                'type': 'error',
                'success': False
            }), 400
            
        response = chatbot.reset_conversation(session_id)
        return jsonify(response)
            
    except Exception as e:
        app.logger.error(f"Chat Reset Error: {str(e)}")
        return jsonify({
            'response': "Error resetting conversation",
            'type': 'error',
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/feedback', methods=['POST'])
def chat_feedback():
    """Submit feedback for a chat message"""
    try:
        data = request.json
        session_id = data.get('session_id')
        message_id = data.get('message_id')
        feedback = data.get('feedback')
        
        if not all([session_id, message_id, feedback]):
            return jsonify({
                'response': "Missing required parameters",
                'type': 'error',
                'success': False
            }), 400
            
        is_authenticated = current_user.is_authenticated
        
        response = chatbot.provide_feedback(
            session_id=session_id,
            message_id=message_id,
            feedback=feedback,
            is_authenticated=is_authenticated
        )
        
        return jsonify(response)
            
    except Exception as e:
        app.logger.error(f"Chat Feedback Error: {str(e)}")
        return jsonify({
            'response': "Error submitting feedback",
            'type': 'error',
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/stats')
@login_required
def chat_stats():
    """Get chatbot usage statistics (authenticated users only)"""
    try:
        stats = chatbot.get_chatbot_stats()
        return jsonify(stats)
    
    except Exception as e:
        app.logger.error(f"Chat Stats Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WebSocket endpoints for real-time chat
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    app.logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    app.logger.info(f"Client disconnected: {request.sid}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle incoming chat messages via WebSocket"""
    query = data.get('query', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    request_id = data.get('request_id', '')
    
    is_authenticated = current_user.is_authenticated
    username = current_user.username if is_authenticated else None
    
    try:
        # Process query through chatbot
        response = chatbot.process_query(
            query=query,
            session_id=session_id,
            is_authenticated=is_authenticated,
            username=username
        )
        
        # Add request_id to response for client-side message matching
        response['request_id'] = request_id
        
        # Emit response back to the client
        emit('chat_response', response)
    
    except Exception as e:
        app.logger.error(f"WebSocket Chat Error: {str(e)}")
        emit('chat_response', {
            'response': "I encountered an error while processing your request.",
            'type': 'error',
            'success': False,
            'error': str(e),
            'request_id': request_id
        })

@app.route('/api/chat/sessions', methods=['GET'])
@login_required
def get_chat_sessions():
    """Get all chat sessions for the current user."""
    try:
        sessions = chatbot.get_user_chat_sessions(current_user.username)
        return jsonify(sessions)
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        return jsonify({'error': 'Failed to get chat sessions'}), 500

@app.route('/api/chat/sessions/new', methods=['POST'])
@login_required
def create_chat_session():
    """Create a new chat session for the current user."""
    try:
        session_id = chatbot.create_new_chat_session(current_user.username)
        return jsonify({'session_id': session_id})
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        return jsonify({'error': 'Failed to create chat session'}), 500

@app.route('/api/chat/sessions/<session_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(session_id):
    """Get all messages for a specific chat session."""
    try:
        messages = chatbot.get_chat_messages(session_id, current_user.username)
        return jsonify({'messages': messages})
    except Exception as e:
        logger.error(f"Error getting chat messages: {str(e)}")
        return jsonify({'error': 'Failed to get chat messages'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)
