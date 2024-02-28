import torch
from datetime import datetime, timedelta


def predict_open_close(input_date, model, scaler, seq_length, df):
    input_date = datetime.strptime(input_date, '%Y-%m-%d')
    # Preprocess input date to create historical sequence
    historical_data = df[df['Date'] < input_date].tail(seq_length)
    X_pred = scaler.transform(historical_data[['Open', 'Close']].values)
    X_pred = torch.tensor([X_pred], dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(X_pred).numpy()

    y_pred = scaler.inverse_transform(y_pred_scaled)[0]
    open_price, close_price = y_pred[0], y_pred[1]

    return open_price, close_price

