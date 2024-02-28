from flask import Flask, render_template, request
import torch
from datetime import datetime
from model import LSTMModel, df, scaler, seq_length
from predict import predict_open_close

app = Flask(__name__)

# Load the saved model
input_size = 2
hidden_size = 12
num_layers = 2
output_size = 2
dropout_prob = 0.2

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
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
