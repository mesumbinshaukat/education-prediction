from flask import Blueprint, render_template
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
import plotly
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create blueprint
student_analytics_route = Blueprint('student_analytics', __name__)

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client['edupredict']  # Replace with actual DB name
collection = db.predictionHistory

@student_analytics_route.route('/student_analytics')
def student_analytics():
    data = list(collection.find())

    if not data:
        return render_template("student_analytics.html", error="No data to display.")

    df = pd.DataFrame(data)

    # Clean and format
    df.drop(columns=['_id'], inplace=True, errors='ignore')
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # Build charts
    charts = {}
    charts['prediction_distribution'] = json.dumps(
        px.histogram(df, x='prediction', title='Prediction Distribution').to_dict(),
        cls=plotly.utils.PlotlyJSONEncoder
    )

    charts['attendance'] = json.dumps(
        px.box(df, y='attendance', title='Attendance Distribution').to_dict(),
        cls=plotly.utils.PlotlyJSONEncoder
    )

    charts['test_scores'] = json.dumps(
        px.histogram(df, x='test_scores', nbins=20, title='Test Scores Histogram').to_dict(),
        cls=plotly.utils.PlotlyJSONEncoder
    )

    return render_template("student_analytics.html", charts=charts)
