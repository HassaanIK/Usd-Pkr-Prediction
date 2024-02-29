# USD to PKR Prediction Model

## OVERVIEW
This project aims to predict the open and close prices of the US Dollar (USD) in Pakistani Rupees (PKR) based on historical data. The model is built using a Long Short-Term Memory (LSTM) neural network and trained on a dataset containing daily USD to PKR exchange rates from 2004 to 2022.

## PROJECT STRUCTURE
1. Data Preprocessing: The dataset is preprocessed to remove unnecessary columns, handle missing values, and create lag features for input sequences.
2. Model Training: An LSTM model is constructed using PyTorch and trained on the preprocessed data to predict the next day's open and close prices based on a sequence of historical data.
3. Model Evaluation: The trained model is evaluated using mean squared error (MSE) and accuracy metrics to assess its performance.
4. Flask Web App: A Flask web application is created to provide a user interface for entering a date and getting the predicted open and close prices for that date.

## Data Preprocessing
- Removed columns: 'Volume', 'Adj Close'
- Converted 'Date' column to datetime format
- Rounded numerical columns to 2 decimal places
- Filled missing values with column means
- Created lag features 'Open_shifted' and 'Close_shifted' for input sequences

## Model Architecture
- LSTM Model:
  - Input size: 2 (Open and Close prices)
  - Hidden size: 24
  - Number of layers: 2
  - Output size: 2 (Predicted Open and Close prices)
  - Dropout probability: 0.4
    
## Training
 - Sequence Length: 200 (Adjustable)
 - Optimizer: Adam
 - Loss Function: Mean squared error (MSE)
 - Training Epochs: 50
 - Batch Size: 32

## Techniques Used
- Sequence Generation: Created sequences of historical data for input to the LSTM model.
- Scaling: Scaled input data using `MinMaxScaler`.
- Dropout: Used dropout in the LSTM model to reduce overfitting.
- Early Stopping: Implemented early stopping to prevent overfitting.
- Learning Rate Scheduling: Adjusted learning rate using `ReduceLROnPlateau` scheduler.
- Model Saving: Saved trained model parameters to a file for future use.
- Flask Web App: Created a web interface using `Flask` to interact with the trained model.
  
## How to Use
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask web app using `python app.py`.
4. Access the web app in your browser and enter a date to get the predicted open and close prices for that date.
   
## Conclusion
This project demonstrates the use of LSTM neural networks for time series prediction and provides a practical example of how to build and deploy a predictive model using PyTorch and Flask.



