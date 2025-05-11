from flask import Flask, render_template
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
import plotly
import json

app = Flask(__name__)

@app.route('/student_analytics')
def student_analytics():
    client = MongoClient("mongodb://localhost:27017/")
    db = client['your_database_name']
    collection = db.predictionHistory

    data = list(collection.find())
    if not data:
        return "No data to display."

    df = pd.DataFrame(data)
    df.drop(columns=['_id'], inplace=True, errors='ignore')
    df['created_at'] = pd.to_datetime(df['created_at'])

    charts = {}

    charts['prediction_distribution'] = json.dumps(px.histogram(df, x='prediction', title='Prediction Distribution').to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    charts['attendance'] = json.dumps(px.box(df, y='attendance', title='Attendance Distribution').to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    charts['test_scores'] = json.dumps(px.histogram(df, x='test_scores', nbins=20, title='Test Scores Histogram').to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("analytics.html", charts=charts)

if __name__ == '__main__':
    app.run(debug=True)
