from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('Hashtag_sum_2_10.csv')
df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce') or pd.to_datetime(x, format='%m/%d/%Y'))
start_date = '2018-04-01'
end_date = '2023-04-20'
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Create a complete date range and fill missing dates
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
df_full = pd.DataFrame({'date': all_dates})
df = df_full.merge(df, on='date', how='left').fillna(0)

# API endpoint for forecasting
@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    hashtag = data['hashtag']  # e.g., '#covid19'
    periods = data['periods']  # e.g., 10 for 10 days

    # Prepare data for Prophet
    df_prophet = df[['date', hashtag.replace('#', '')]].rename(columns={'date': 'ds', hashtag.replace('#', ''): 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)

    # Forecast
    forecast = model.predict(future)
    forecast_values = forecast[['ds', 'yhat']].tail(periods).round(0).to_dict('records')

    return jsonify(forecast_values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)