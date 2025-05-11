import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify, Response
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, after_this_request
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import pandas as pd
from fpdf import FPDF
from config import get_db, logger
from functools import wraps
import pickle
import os
import logging
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

# Define feature names for consistency
FEATURE_NAMES = ['attendance', 'homework_completion', 'test_scores']

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Validate form data
        required_fields = ['name', 'student_id', 'email', 'attendance', 'homework_completion', 'test_scores']
        for field in required_fields:
            if field not in request.form:
                flash(f"Missing required field: {field}", 'danger')
                return redirect(url_for('index'))

        # Debug logging
        logger.debug(f"Form data received: {request.form}")

        if 'username' not in session:
            flash("Please login to make predictions", 'danger')
            return redirect(url_for('login'))

        # Get and validate form data
        name = request.form['name']
        student_id = request.form['student_id']
        email = request.form['email']
        
        try:
            attendance = float(request.form['attendance'])
            homework_completion = float(request.form['homework_completion'])
            test_scores = float(request.form['test_scores'])
        except ValueError:
            flash("Invalid numerical values provided", 'danger')
            return redirect(url_for('index'))

        logger.info(f"Processing prediction for student {student_id}")

        # Calculate percentage (for fallback and display)
        percentage = (test_scores * 0.5) + (attendance * 0.3) + (homework_completion * 0.2)
        
        # Calculate prediction status based on percentage
        if percentage >= 80:
            prediction_text = "Excellent"
            prediction_message = "Excellent! The student shows strong potential."
        elif percentage >= 60:
            prediction_text = "Good"
            prediction_message = "Good Result! The student is likely to perform well."
        else:
            prediction_text = "Needs Improvement"
            prediction_message = "Attention Needed: The student may need additional support."
            
        prediction = 1 if percentage >= 60 else 0  # Keep for backward compatibility
        probability = round(percentage, 2)
        confidence = probability / 100.0  # Convert to 0-1 range for consistency
        
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
        try:
            result = db.predictionHistory.insert_one(prediction_data)
            logger.info(f"Prediction saved with ID: {result.inserted_id}")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            flash("Failed to save prediction to database, but will continue with results", 'warning')

        # Store in session for PDF generation - create a simplified version that avoids complex objects
        session['student_data'] = {
            'name': name,
            'student_id': student_id,
            'email': email,
            'attendance': attendance,
            'homework_completion': homework_completion,
            'test_scores': test_scores,
            'prediction': prediction_text,
            'probability': probability
        }
        
        logger.info(f"Session data stored for student: {student_id}")

        # --- START: Retrain and save the model ---
        try:
            history = list(db.predictionHistory.find({}))

            if len(history) >= 5:  # Minimum data to avoid crash
                # Create directory if it doesn't exist
                os.makedirs('./models', exist_ok=True)
                
                X = np.array([[s['attendance'], s['homework_completion'], s['test_scores']] for s in history])
                y = np.array([s['prediction_number'] if 'prediction_number' in s else (1 if s['prediction'] in ['Good', 'Excellent'] else 0) for s in history])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LogisticRegression()
                model.fit(X_scaled, y)
              
                with open('./models/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                    logger.info("Scaler.pkl updated successfully.")

                with open('./models/best_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                    logger.info("Best_model.pkl updated successfully.")
        except Exception as model_error:
            logger.error(f"Error updating model: {model_error}")
        # --- END: Retrain and save the model ---

        return render_template(
            'result.html',
            prediction=prediction_text,
            probability=probability,
            prediction_message=prediction_message,
            student=prediction_data,
            show_report=True  # Flag to show report download button
        )

    except Exception as e:
        flash(f"Prediction failed: {e}", 'danger')
        return redirect(url_for('index'))

@app.route('/report/<student_id>', methods=['GET'])
@login_required
def report(student_id):
    try:
        # Try to get student data from session first
        student = session.get('student_data')
        logger.debug(f"Session student data: {student}")
        
        # Verify we have the correct student ID
        if not student or str(student.get('student_id')) != str(student_id):
            logger.info(f"Student {student_id} not found in session, trying database")
            # Try to get from database if not in session
            student = db.predictionHistory.find_one({"student_id": student_id})
            if not student:
                flash("Student data not found for report generation", 'danger')
                return redirect(url_for('prediction_history'))
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Student Performance Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Student Information Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Student Information:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Name: {student["name"]}', 0, 1)
        pdf.cell(0, 10, f'Student ID: {student["student_id"]}', 0, 1)
        pdf.cell(0, 10, f'Email: {student["email"]}', 0, 1)
        pdf.ln(5)
        
        # Performance Metrics Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Performance Metrics:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Attendance: {student["attendance"]}%', 0, 1)
        pdf.cell(0, 10, f'Homework Completion: {student["homework_completion"]}%', 0, 1)
        pdf.cell(0, 10, f'Test Scores: {student["test_scores"]}%', 0, 1)
        pdf.ln(5)
        
        # Prediction Results Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Prediction Result:', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Get prediction text and determine status
        prediction = student.get('prediction', 'Undefined')
        probability = student.get('probability', 0)
        
        if isinstance(prediction, (int, float)):
            prediction_text = "Good" if prediction == 1 else "Needs Improvement"
        else:
            prediction_text = prediction

        pdf.cell(0, 10, f'Performance Prediction: {prediction_text}', 0, 1)
        pdf.cell(0, 10, f'Success Probability: {probability}%', 0, 1)
        
        # Recommendations Section
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Recommendations:', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Convert probability to float for comparison
        try:
            prob_value = float(probability)
        except (ValueError, TypeError):
            prob_value = 0
            
        if prob_value >= 80:
            recommendation = 'Excellent performance! Continue with the current study habits and consider taking on additional challenging materials.'
        elif prob_value >= 60:
            recommendation = 'Good performance. Focus on maintaining consistency and look for areas of potential improvement.'
        else:
            recommendation = 'Areas need attention. Consider increasing study hours and seeking additional support in challenging subjects.'
        
        pdf.multi_cell(0, 10, recommendation)
        
        # Footer
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f'Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = os.path.join(temp_dir, f'student_report_{student_id}_{timestamp}.pdf')
        
        try:
            # Save and send PDF
            pdf.output(pdf_filename)
            
            return_data = send_file(
                pdf_filename,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'student_report_{student_id}.pdf'
            )
            
            # Clean up file after sending
            @after_this_request
            def remove_file(response):
                try:
                    if os.path.exists(pdf_filename):
                        os.remove(pdf_filename)
                except Exception as e:
                    logger.error(f"Error removing temp file: {e}")
                return response
            
            return return_data
            
        except Exception as e:
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
            raise e
            
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        flash(f"Failed to generate report: {e}", 'danger')
        return redirect(url_for('prediction_history'))

@app.route('/prediction-history', methods=['GET'])
@login_required
def prediction_history():
    try:
        username = session.get('username')
        user_predictions = list(db.predictionHistory.find({"username": username}).sort("timestamp", -1))

        for prediction in user_predictions:
            if 'timestamp' in prediction:
                prediction['formatted_date'] = prediction['timestamp'].strftime("%Y-%m-%d %H:%M:%S")

        return render_template('prediction_history.html', predictions=user_predictions)
    except Exception as e:
        flash(f"Failed to retrieve prediction history: {e}", 'danger')
        return redirect(url_for('index'))

@app.route('/analytics')
@login_required
def analytics():
    try:
        # Retry logic for MongoDB connection
        max_retries = 3
        retry_count = 0
        predictions = []
        
        while retry_count < max_retries:
            try:
                predictions = list(db.predictionHistory.find({}).sort("timestamp", -1))
                logger.info(f"Retrieved {len(predictions)} predictions for analytics")
                break
            except Exception as mongo_error:
                retry_count += 1
                logger.warning(f"MongoDB connection attempt {retry_count} failed: {str(mongo_error)}")
                if retry_count == max_retries:
                    flash("Unable to connect to database. Please try again later.", 'danger')
                    return redirect(url_for('index'))
        
        # Process predictions
        processed_predictions = []
        total_attendance = 0
        total_homework = 0
        total_tests = 0
        
        for pred in predictions:
            processed_pred = pred.copy()
            
            # Format date
            processed_pred['formatted_date'] = pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if 'timestamp' in pred else "Unknown"
            
            # Normalize prediction status
            pred_status = pred.get('prediction', 'Undefined')
            if isinstance(pred_status, (int, float)):
                processed_pred['prediction'] = "Good" if pred_status == 1 else "Needs Improvement"
            elif isinstance(pred_status, str):
                pred_lower = pred_status.lower()
                processed_pred['prediction'] = "Good" if any(x in pred_lower for x in ['good', 'excellent']) else "Needs Improvement"
            else:
                processed_pred['prediction'] = "Needs Improvement"
            
            # Calculate probability if missing
            if 'probability' not in processed_pred or not isinstance(processed_pred['probability'], (int, float)):
                try:
                    attendance = float(pred.get('attendance', 0))
                    homework = float(pred.get('homework_completion', 0))
                    tests = float(pred.get('test_scores', 0))
                    
                    total_attendance += attendance
                    total_homework += homework
                    total_tests += tests
                    
                    processed_pred['probability'] = round(
                        (tests * 0.5) + (attendance * 0.3) + (homework * 0.2),
                        2
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error calculating probability for student {pred.get('student_id', 'unknown')}: {e}")
                    processed_pred['probability'] = 0.0
            else:
                # Add to totals if probability exists
                try:
                    attendance = float(pred.get('attendance', 0))
                    homework = float(pred.get('homework_completion', 0))
                    tests = float(pred.get('test_scores', 0))
                    
                    total_attendance += attendance
                    total_homework += homework
                    total_tests += tests
                except (ValueError, TypeError):
                    pass
            
            processed_predictions.append(processed_pred)
        
        # Calculate analytics
        total_students = len(processed_predictions)
        if total_students > 0:
            analytics_data = {
                'total_students': total_students,
                'good_predictions': sum(1 for p in processed_predictions if p['prediction'] == "Good"),
                'needs_improvement': sum(1 for p in processed_predictions if p['prediction'] == "Needs Improvement"),
                'avg_probability': round(sum(float(p.get('probability', 0)) for p in processed_predictions) / total_students, 1),
                'avg_attendance': round(total_attendance / total_students, 1),
                'avg_homework': round(total_homework / total_students, 1),
                'avg_tests': round(total_tests / total_students, 1)
            }
        else:
            analytics_data = {
                'total_students': 0,
                'good_predictions': 0,
                'needs_improvement': 0,
                'avg_probability': 0,
                'avg_attendance': 0,
                'avg_homework': 0,
                'avg_tests': 0
            }
        
        logger.info(f"Analytics data prepared successfully: {len(processed_predictions)} records")
        
        return render_template(
            'student_analytics.html',
            predictions=processed_predictions,
            analytics=analytics_data
        )
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        flash(f"Failed to load analytics: {str(e)}", 'danger')
        return redirect(url_for('index'))



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
