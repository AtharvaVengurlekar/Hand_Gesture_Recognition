import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Define your gestures here
gestures = ['thumbs_up', 'thumbs_down', 'ok_sign', 'gang_sign']

# Load the data
data = np.load('data.npy')
labels = np.load('labels.npy')

# Normalize the data (scale pixel values to [0, 1])
data = data / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(gestures), activation='softmax')  # Use len(gestures) for the output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('gesture_model.h5')

print("Model training completed and saved as gesture_model.h5")
