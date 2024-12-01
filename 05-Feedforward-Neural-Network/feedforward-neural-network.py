import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Example input data
x = [[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.5, 0.5, 0.5], [0.6, 0.7, 0.8]]

# Ensure the input is a Tensor or Numpy array
x = np.array(x)

# Define the model
model = Sequential()

# Input layer with 3 features
model.add(Dense(15, input_dim=3, activation='relu'))  # Hidden layer with 15 neurons
model.add(Dense(1, activation='sigmoid'))   # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Fit the model (assuming y is correctly defined)
y = np.array([0, 0, 1, 1])

# Train the model
model.fit(x, y, epochs=5, batch_size=1)  # epoch - number of iterations for dataset

# Evaluate the model
loss, accuracy = model.evaluate(x, y)
print(f'Accuracy: {accuracy}, Loss: {loss}')

