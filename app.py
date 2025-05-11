import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify, Response
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

app = Flask(__name__)
app.secret_key = 'your_secret_key'

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
            hashed_password = generate_password_hash(request.form['password'])
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
    if request.method == 'POST':
        users = db.users
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            user = users.find_one({"username": username})
            if user and check_password_hash(user['password'], password):
                session['username'] = username
                session['email'] = user['email']
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
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
        name = request.form['name']
        student_id = request.form['student_id']
        email = request.form['email']
        attendance = float(request.form['attendance'])
        homework_completion = float(request.form['homework_completion'])
        test_scores = float(request.form['test_scores'])

        percentage = (test_scores * 0.5) + (attendance * 0.3) + (homework_completion * 0.2)
        prediction = 1 if percentage > 60 else 0
        probability = round(percentage, 2)
        confidence = probability / 100.0  # Convert to 0-1 range for consistency

        # Convert numeric prediction to text status
        prediction_text = "Good" if prediction == 1 else "Needs Improvement"
        
        # Create predicted_scores object for visualization
        predicted_scores = {
            "Attendance Score": attendance,
            "Homework Score": homework_completion,
            "Test Score": test_scores,
            "Overall": percentage
        }

        # Create a single comprehensive prediction record
        prediction_data = {
            # Student information
            'name': name,
            'student_id': student_id, 
            'email': email,
            'attendance': attendance,
            'homework_completion': homework_completion,
            'test_scores': test_scores,
            
            # Prediction results
            'prediction': prediction_text,  # Store as text for display
            'prediction_number': prediction,  # Store original numeric value
            'prediction_score': percentage,  # Overall score for comparison
            'probability': probability,  # Original probability (percent format)
            'confidence': confidence,  # Normalized confidence (0-1 range)
            'predicted_scores': predicted_scores,  # For visualization
            
            # Metadata
            'username': session.get('username', 'public'),  # Include user if available
            'created_at': datetime.now(),
            'timestamp': datetime.now()  # For compatibility with both naming conventions
        }
        
        # Insert the prediction once
        result = db.predictionHistory.insert_one(prediction_data)
        
        # Store in session for PDF generation
        prediction_data_for_session = prediction_data.copy()
        prediction_data_for_session.pop('_id', None)  # Remove ObjectId for session
        session['student_data'] = prediction_data_for_session
    
    
        prediction_message = "Good Result! The student is likely to perform well." if prediction == 1 else "Bad Result! The student may not succeed based on current indicators."

        # --- START: Retrain and save the model ---
        history = list(db.predictionHistory.find({}))

        if len(history) >= 5:  # Minimum data to avoid crash
            X = np.array([[s['attendance'], s['homework_completion'], s['test_scores']] for s in history])
            y = np.array([s['prediction'] for s in history])

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

        return render_template('result.html', prediction=prediction, probability=probability, prediction_message=prediction_message, student=student_data)

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
            prediction = 'Good Student' if student['prediction'] == 1 else 'Needs Improvement'
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
    # Get all predictions for the current user
    predictions = list(db.predictionHistory.find(
        {'username': session['username']},
        {'_id': 0}  # Exclude MongoDB _id field
    ).sort('timestamp', -1))  # Sort by timestamp in descending order
    
    return render_template('prediction_history.html', predictions=predictions)

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
        
        # Fetch predictions from MongoDB with pagination
        # Don't exclude _id here - we'll handle it during serialization
        predictions = list(db.predictionHistory.find().sort('created_at', -1).skip(skip).limit(limit))
        
        # Process results to be JSON serializable
        processed_predictions = []
        for pred in predictions:
            # Create a new dictionary for the processed prediction
            processed_pred = {}
            
            # Handle _id explicitly
            if '_id' in pred:
                processed_pred['id'] = str(pred['_id'])
            
            # Add timestamp
            if 'created_at' in pred:
                processed_pred['timestamp'] = pred['created_at'].isoformat() if hasattr(pred['created_at'], 'isoformat') else str(pred['created_at'])
            elif 'timestamp' in pred:
                processed_pred['timestamp'] = pred['timestamp'].isoformat() if hasattr(pred['timestamp'], 'isoformat') else str(pred['timestamp'])
            else:
                # Default timestamp if none exists
                processed_pred['timestamp'] = datetime.now().isoformat()
            
            # Ensure required fields have default values
            for field in ['student_id', 'name', 'email', 'attendance', 'homework_completion', 'test_scores']:
                processed_pred[field] = 'N/A'
                
            # Default numerical values
            for field in ['attendance', 'homework_completion', 'test_scores', 'prediction_score']:
                if field not in processed_pred or processed_pred[field] == 'N/A':
                    processed_pred[field] = 0
            
            # Flatten student_data if present
            if 'student_data' in pred and isinstance(pred['student_data'], dict):
                for key, value in pred['student_data'].items():
                    # Avoid ObjectId and datetime issues
                    if isinstance(value, ObjectId):
                        processed_pred[key] = str(value)
                    elif isinstance(value, datetime):
                        processed_pred[key] = value.isoformat()
                    else:
                        processed_pred[key] = value
            
            # Copy remaining fields, handling special types
            for key, value in pred.items():
                if key not in ['_id', 'student_data', 'created_at'] and key not in processed_pred:
                    if isinstance(value, ObjectId):
                        processed_pred[key] = str(value)
                    elif isinstance(value, datetime):
                        processed_pred[key] = value.isoformat()
                    else:
                        processed_pred[key] = value
            
            # Convert numeric prediction to text status
            if 'prediction' in processed_pred:
                # If prediction is stored as 0/1
                if processed_pred['prediction'] in [0, 1, '0', '1']:
                    numeric_pred = int(processed_pred['prediction'])
                    if numeric_pred == 1:
                        processed_pred['prediction'] = 'Good'
                    else:
                        processed_pred['prediction'] = 'Needs Improvement'
                # If prediction is already a string but needs normalization
                elif isinstance(processed_pred['prediction'], str):
                    pred_lower = processed_pred['prediction'].lower()
                    if 'excellent' in pred_lower:
                        processed_pred['prediction'] = 'Excellent'
                    elif 'good' in pred_lower:
                        processed_pred['prediction'] = 'Good'
                    elif 'average' in pred_lower:
                        processed_pred['prediction'] = 'Average'
                    elif 'needs' in pred_lower or 'improvement' in pred_lower or 'poor' in pred_lower:
                        processed_pred['prediction'] = 'Needs Improvement'
                    else:
                        # Default if string doesn't match any known category
                        processed_pred['prediction'] = 'Undefined'
            else:
                # Default if prediction field is missing
                processed_pred['prediction'] = 'Undefined'
                
            # Calculate overall score if not present
            if 'prediction_score' not in processed_pred:
                # Try to calculate from components if available
                if all(k in processed_pred for k in ['attendance', 'homework_completion', 'test_scores']):
                    try:
                        attendance = float(processed_pred['attendance'])
                        homework = float(processed_pred['homework_completion'])
                        tests = float(processed_pred['test_scores'])
                        processed_pred['prediction_score'] = (tests * 0.5) + (attendance * 0.3) + (homework * 0.2)
                    except (ValueError, TypeError):
                        processed_pred['prediction_score'] = 0
                # Use probability if available
                elif 'probability' in processed_pred:
                    processed_pred['prediction_score'] = processed_pred['probability']
                else:
                    processed_pred['prediction_score'] = 0
            
            # Ensure confidence is present and in decimal form (0-1 range)
            if 'confidence' not in processed_pred:
                if 'probability' in processed_pred:
                    # Normalize probability to 0-1 range if it looks like a percentage
                    prob_value = float(processed_pred['probability'])
                    if prob_value > 1:
                        processed_pred['confidence'] = prob_value / 100
                    else:
                        processed_pred['confidence'] = prob_value
                else:
                    # Default confidence based on prediction_score
                    score = float(processed_pred['prediction_score'])
                    if score > 1:
                        processed_pred['confidence'] = min(score / 100, 1.0)  # Normalize to 0-1
                    else:
                        processed_pred['confidence'] = score
            
            processed_predictions.append(processed_pred)
        
        # Use json_util from bson to handle MongoDB types properly
        return Response(
            json_util.dumps(processed_predictions),
            mimetype='application/json'
        )
    
    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
