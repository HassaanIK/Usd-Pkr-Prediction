import torch.nn as nn
import torch
from model import model
from data_preparing import X_train, y_train, X_val, y_val

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

from torch.optim.lr_scheduler import ReduceLROnPlateau

num_epochs = 50
batch_size = 32
threshold = 0.5  # Define the threshold for accuracy
patience = 5  # Number of epochs to wait before early stopping

# # Initialize early stopping
early_stopping = False
best_val_loss = float('inf')
counter = 0

# # Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0.0
    correct_predictions = 0

    for i in range(0, len(X_train), batch_size):
        X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / (len(X_train) / batch_size)

    # Calculate validation loss and accuracy
    with torch.no_grad():
        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        # Calculate accuracy
        for j in range(len(val_outputs)):
            if abs(val_outputs[j][0] - y_val[j][0]) <= threshold and abs(val_outputs[j][1] - y_val[j][1]) <= threshold:
                correct_predictions += 1

    accuracy = (correct_predictions / len(X_val)) * 100

    # Adjust learning rate
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation Loss: {val_loss.item():.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break


# Save the model state dictionary
torch.save(model.state_dict(), 'usd_model.pth')
