import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate dummy sequential data
data = np.random.random((1000, 10, 1))  # 1000 samples, 10 time steps, 1 feature
labels = np.random.random((1000, 1))    # 1000 labels

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(32, input_shape=(10, 1)))  # 32 units in the hidden layer
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(data, labels, epochs=10, batch_size=32)
