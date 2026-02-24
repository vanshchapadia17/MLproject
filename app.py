from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.components.drift_monitor import (
    log_prediction, get_logs, get_stats, build_trend_chart
)

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('reading_score')),
        writing_score=float(request.form.get('writing_score')),
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results, shap_img = predict_pipeline.predict(pred_df)

    # Log for drift monitoring
    log_prediction(
        raw_features={
            "gender":                      data.gender,
            "race_ethnicity":              data.race_ethnicity,
            "parental_level_of_education": data.parental_level_of_education,
            "lunch":                       data.lunch,
            "test_preparation_course":     data.test_preparation_course,
            "reading_score":               data.reading_score,
            "writing_score":               data.writing_score,
        },
        prediction=float(results[0])
    )

    return render_template('home.html', results=results[0], shap_img=shap_img)


@app.route('/monitor')
def monitor():
    df = get_logs()
    if df is None:
        return render_template('monitor.html', stats=None, recent=None, trend_img=None)

    stats     = get_stats(df)
    recent    = df.tail(10).iloc[::-1].to_dict('records')
    trend_img = build_trend_chart(df)
    return render_template('monitor.html', stats=stats, recent=recent, trend_img=trend_img)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
