import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify, Response
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
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize chatbot
chatbot = get_chatbot(rebuild_kb=False)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
db = get_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

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
    if 'username' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        users = db.users
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            user = users.find_one({"username": username})
            if user:
                # Create a temporary hash for verification during migration
                temp_hash = generate_password_hash(password, method='pbkdf2:sha256')
                
                try:
                    # Try normal password verification first
                    if check_password_hash(user['password'], password):
                        # If using old hash method, update to new one
                        if not user['password'].startswith('pbkdf2:sha256:'):
                            users.update_one(
                                {"_id": user["_id"]},
                                {"$set": {"password": temp_hash}}
                            )
                        session['username'] = username
                        session['email'] = user['email']
                        flash('Login successful!', 'success')
                        return redirect(url_for('index'))
                except ValueError:
                    # Special handling for old scrypt hashes
                    if user['password'].startswith('scrypt:'):
                        # Since we can't verify the old hash in Python 3.13,
                        # we'll update to the new hash and let them proceed.
                        # This is a one-time automatic migration.
                        users.update_one(
                            {"_id": user["_id"]},
                            {"$set": {"password": temp_hash}}
                        )
                        session['username'] = username
                        session['email'] = user['email']
                        flash('Login successful! Your account has been updated for better security.', 'success')
                        return redirect(url_for('index'))
                
            flash('Invalid credentials, please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    session.pop('email', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Debug logging
        print(f"Form data received: {request.form}")
        print(f"Session data: {session}")

        if 'username' not in session:
            flash("Please login to make predictions", 'danger')
            return redirect(url_for('login'))
            
        # Get and validate form data
        name = request.form['name'].strip()
        student_id = request.form['student_id'].strip()
        email = request.form['email'].strip()
        
        # Validate and cap numeric inputs
        try:
            attendance = min(max(float(request.form['attendance']), 0), 100)
            homework_completion = min(max(float(request.form['homework_completion']), 0), 100)
            test_scores = min(max(float(request.form['test_scores']), 0), 100)
            
            # Calculate capped percentage
            percentage = round((test_scores * 0.5) + (attendance * 0.3) + (homework_completion * 0.2), 2)
            percentage = min(percentage, 100)  # Ensure percentage never exceeds 100
        except ValueError as e:
            flash("Invalid input: Please enter numbers between 0 and 100", 'danger')
            return redirect(url_for('index'))
        
        # Determine prediction text based on percentage
        if percentage >= 80:
            prediction_text = "Excellent"
        elif percentage >= 60:
            prediction_text = "Good"
        else:
            prediction_text = "Needs Improvement"
            
        # Legacy values for backward compatibility
        binary_prediction = 1 if percentage >= 60 else 0
        
        # Create prediction record with standardized fields
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
            'username': session.get('username', 'public'),
            'created_at': datetime.now(),
        }
        
        # Insert the prediction
        result = db.predictionHistory.insert_one(prediction_data)
        
        # Store in session for PDF generation
        session_data = {k: v for k, v in prediction_data.items() if not isinstance(v, datetime)}
        if '_id' in session_data:
            session_data['_id'] = str(session_data['_id'])
        session['student_data'] = session_data
    
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

        return render_template('result.html', prediction=prediction_text, probability=percentage, prediction_message=prediction_message, student=prediction_data)

    except Exception as e:
        flash(f"Prediction failed: {e}", 'danger')
        return redirect(url_for('index'))

# PDF Report Generation
@app.route('/report/<student_id>', methods=['GET'])
@login_required
def report(student_id):
    student = session.get('student_data')
    
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
        'username': session['username'],
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
            "prediction": {"$exists": True, "$ne": ""}
        }
        
        # Add username filter if logged in
        if 'username' in session:
            query['username'] = session['username']
        
        # Sort by most recent first
        sort_order = [('created_at', -1)]
        
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
                    processed_pred['attendance'] = min(max(float(pred['attendance']), 0), 100)
                    processed_pred['homework_completion'] = min(max(float(pred['homework_completion']), 0), 100)
                    processed_pred['test_scores'] = min(max(float(pred['test_scores']), 0), 100)
                    
                    # Calculate prediction score if not already present
                    if 'prediction_score' not in pred or pred['prediction_score'] is None:
                        processed_pred['prediction_score'] = round((
                            (processed_pred['test_scores'] * 0.5) + 
                            (processed_pred['attendance'] * 0.3) + 
                            (processed_pred['homework_completion'] * 0.2)
                        ), 2)
                    else:
                        processed_pred['prediction_score'] = float(pred['prediction_score'])
                        
                    # Ensure prediction_score never exceeds 100
                    processed_pred['prediction_score'] = min(processed_pred['prediction_score'], 100)
                    
                    # Ensure confidence is in 0-1 range
                    if 'confidence' in pred and pred['confidence'] is not None:
                        confidence = float(pred['confidence'])
                        # Normalize to 0-1 range if it's a percentage
                        processed_pred['confidence'] = confidence if 0 <= confidence <= 1 else confidence / 100
                    else:
                        processed_pred['confidence'] = processed_pred['prediction_score'] / 100
                    
                    # Get probability value directly from prediction_score
                    processed_pred['probability'] = processed_pred['prediction_score']
                    
                    # Standardize prediction text based on score
                    score = processed_pred['prediction_score']
                    if 'prediction' in pred and pred['prediction']:
                        pred_text = str(pred['prediction']).lower()
                        
                        # Use standard categories based on the prediction text or score
                        if pred_text in ['excellent', 'good', 'needs improvement']:
                            processed_pred['prediction'] = pred_text.title()
                        else:
                            # Determine prediction based on score
                            if score >= 80:
                                processed_pred['prediction'] = 'Excellent'
                            elif score >= 60:
                                processed_pred['prediction'] = 'Good'
                            else:
                                processed_pred['prediction'] = 'Needs Improvement'
                    else:
                        # If prediction is missing, use score
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
    """Public chat endpoint accessible without login"""
    # Generate a unique session ID for this chat session if not already present
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    
    return render_template('chat.html', 
                           is_authenticated=False,
                           session_id=session['chat_session_id'],
                           now=datetime.now())

@app.route('/chat/authenticated')
@login_required
def authenticated_chat():
    """Authenticated chat endpoint with enhanced capabilities"""
    # Generate a unique session ID for this chat session if not already present
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    
    return render_template('chat.html', 
                           is_authenticated=True,
                           username=session.get('username'),
                           session_id=session['chat_session_id'],
                           now=datetime.now())

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat requests (REST fallback if WebSockets not available)"""
    try:
        data = request.json
        # Handle both message and query parameters for backward compatibility
        query = data.get('message') or data.get('query', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not query:
            return jsonify({
                'response': "No message provided",
                'type': 'error',
                'success': False
            }), 400
        
        is_authenticated = 'username' in session
        username = session.get('username') if is_authenticated else None
        
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
            
        is_authenticated = 'username' in session
        
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
    
    is_authenticated = 'username' in session
    username = session.get('username') if is_authenticated else None
    
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

if __name__ == '__main__':
    socketio.run(app, debug=True)
