from flask import Flask, render_template, request
import torch
from datetime import datetime
from model import model
from normalizer import scaler
from data_preparing import seq_length
from data_preprocessing import df
from predict import predict_open_close

app = Flask(__name__)


model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_date = request.form['input_date']
    open_price, close_price = predict_open_close(input_date, model, scaler, seq_length, df)

    return render_template('index.html', open_price=open_price, close_price=close_price)

if __name__ == '__main__':
    app.run(debug=True)
