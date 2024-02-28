
"""##Data Preprocessing"""

import pandas as pd
df = pd.read_csv('Data\\USD To PKR 2004-2022.csv')
df = df.drop(columns = ['Volume', 'Adj Close'])
df['Date'] = pd.to_datetime(df['Date'])
df.loc[:, df.columns != 'Date'] = df.loc[:, df.columns != 'Date'].round(2)
df = df.fillna(df.mean())
# Feature Engineering
df['Open_shifted'] = df['Open'].shift(1)
df['Close_shifted'] = df['Close'].shift(1)
df.dropna(inplace=True)

print(f'Data Preprocessed...')

"""##Model"""

from sklearn.preprocessing import MinMaxScaler
# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Open_shifted', 'Close_shifted']])

import numpy as np
# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 100  # Adjust the sequence length as needed
X, y = create_sequences(scaled_data, seq_length)

import torch
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Train-test split
train_size = int(0.8 * len(X_tensor))
X_train, X_val, y_train, y_val = X_tensor[:train_size], X_tensor[train_size:], y_tensor[:train_size], y_tensor[train_size:]

import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
print(f'Model Initialized...')
# Define hyperparameters
input_size = 2  # Assuming 'Open' and 'Close' prices are the input features
hidden_size = 12
num_layers = 2
output_size = 2  # Assuming predicting 'Open' and 'Close' prices
dropout_prob = 0.4

# Create the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)

# # Loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# # from torch.optim.lr_scheduler import ReduceLROnPlateau

# # num_epochs = 50
# # batch_size = 32
# # threshold = 0.5  # Define the threshold for accuracy
# # patience = 5  # Number of epochs to wait before early stopping

# # # Initialize early stopping
# # early_stopping = False
# # best_val_loss = float('inf')
# # counter = 0

# # # Initialize the learning rate scheduler
# # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
# print(f'Training Started...')
# Training Loop
# for epoch in range(num_epochs):
#     total_loss = 0.0
#     correct_predictions = 0

#     for i in range(0, len(X_train), batch_size):
#         X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     average_loss = total_loss / (len(X_train) / batch_size)

#     # Calculate validation loss and accuracy
#     with torch.no_grad():
#         model.eval()
#         val_outputs = model(X_val)
#         val_loss = criterion(val_outputs, y_val)

#         # Calculate accuracy
#         for j in range(len(val_outputs)):
#             if abs(val_outputs[j][0] - y_val[j][0]) <= threshold and abs(val_outputs[j][1] - y_val[j][1]) <= threshold:
#                 correct_predictions += 1

#     accuracy = (correct_predictions / len(X_val)) * 100

#     # Adjust learning rate
#     scheduler.step(val_loss)

#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation Loss: {val_loss.item():.4f}')

#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             print(f'Early stopping at epoch {epoch+1}')
#             break
# print(f'Saving Model...')
# Save the model state dictionary
# torch.save(model.state_dict(), 'usd_model.pth')

