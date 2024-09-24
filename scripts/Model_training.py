# Importing necessary libraries for image processing, data handling, and building machine learning models
import os
import cv2
import numpy as np
from tqdm import tqdm  # Used for progress bars
from PIL import Image  # For image handling
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.utils import shuffle  # For shuffling the data
import seaborn as sns  # Visualization library

import tensorflow as tf  # TensorFlow for building deep learning models
from tensorflow import keras  # High-level API for TensorFlow
from tensorflow.keras import layers  # Used to define layers in neural networks
from tensorflow.keras.models import Sequential, Model  # For creating sequential models
from tensorflow.keras.layers import BatchNormalization  # Normalization layer to stabilize training
from keras.layers import Input, Lambda, Dense, Flatten, Activation, Dropout  # Various neural network layers
from keras.preprocessing.image import ImageDataGenerator  # To load and augment images
from tensorflow.keras.optimizers import RMSprop  # Optimizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Callbacks for training
from keras import applications  # Pretrained models can be imported from this
from tensorflow.keras import models  # For saving, loading, and manipulating models

# Step 1: Traverse the Dataset Directory
for dirname, _, filenames in os.walk(r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Explanation:
# This part traverses the directory where the dataset is stored and prints the file paths of all images.
# `os.walk` recursively lists all files in a directory structure.

# Step 2: Set Image Size and Batch Size
image_size = (480, 640)  # Target size to which all images will be resized
batch_size = 1  # Number of images processed at a time

# Step 3: Create Image Data Generator for Augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,  # Randomly flip the images horizontally
    vertical_flip=True  # Randomly flip the images vertically
)

# Step 4: Load Training and Validation Data
train_ds = datagen.flow_from_directory(
    "C:/Users/Salman Ahmed/Desktop/Wat_Dataset/Training",  # Path to training data
    target_size=image_size,  # Resize all images to 480x640
    batch_size=batch_size,  # Process one image at a time
    class_mode='binary',  # Binary classification (pure vs contaminated water)
    color_mode="rgb",  # RGB images
)

val_ds = datagen.flow_from_directory(
    "C:/Users/Salman Ahmed/Desktop/Wat_Dataset/Testing",  # Path to validation (testing) data
    target_size=image_size,  # Resize images to 480x640
    batch_size=batch_size,  # Process one image at a time
    class_mode='binary',  # Binary classification
    color_mode="rgb"  # RGB images
)

# Explanation:
# `ImageDataGenerator` is used to augment the dataset by applying transformations (flips).
# `flow_from_directory` automatically labels images based on their folder names and loads the data in batches.

# Step 5: Define the CNN Model
model = models.Sequential()  # Creating a sequential model (stacking layers)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)))  # First convolutional layer with ReLU activation
model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer to downsample the image
model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Second convolutional layer
model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer to reduce dimensions
model.add(Flatten())  # Flatten the 2D feature maps into a 1D vector
model.add(layers.Dense(100, activation='relu'))  # Fully connected layer with 100 units
model.add(layers.Dense(80, activation='relu'))  # Fully connected layer with 80 units
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

print(model.summary())  # Print the model architecture

# Explanation:
# The model is a basic CNN with two convolutional layers, followed by fully connected layers.
# The last layer uses sigmoid activation to produce a binary output (0 or 1 for pure or contaminated water).

# Step 6: Compile the Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Use Adam optimizer with learning rate 0.001
    loss=tf.keras.losses.BinaryCrossentropy(),  # Binary cross-entropy loss for binary classification
    metrics=['accuracy']  # Track accuracy during training
)

# Explanation:
# The model is compiled with Adam optimizer and binary cross-entropy loss, suitable for binary classification.

# Step 7: Train the Model
epochs = 5  # Number of training iterations
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)  # Train the model and validate on val_ds

# Explanation:
# The model is trained for 5 epochs. During each epoch, the model updates its weights based on the training data,
# and performance is checked on the validation set.

# Step 8: Save the Model
model.save('C:/Users/Salman Ahmed/Desktop/Wat_Dataset/model/my_model_Water.h5')

# Explanation:
# After training, the model is saved in a file so that it can be reused without retraining.

# Step 9: Plot Training Accuracy and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='train acc')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='val acc')  # Plot validation accuracy
plt.legend()
plt.title('Accuracy')
plt.show()

# Explanation:
# This part visualizes the accuracy of the model on the training and validation datasets.

# Step 10: Plot Training Loss and Validation Loss
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='train loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='val loss')  # Plot validation loss
plt.legend()
plt.title('Loss')
plt.show()

# Explanation:
# This part visualizes the training and validation loss to check how well the model fits the data.

# Step 11: Load the Saved Model for Testing
new_model = tf.keras.models.load_model(r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\model\my_model_Water.h5')

# Explanation:
# The previously saved model is loaded back for testing new images.

# Step 12: Print the Model Summary
new_model.summary()

# Explanation:
# Prints the architecture of the loaded model.

# Step 13: Load an Image for Testing
img = cv2.imread(r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P2.jpg')  # Load the image using OpenCV

# Explanation:
# An image from the test dataset is loaded using OpenCV.

# Step 14: Preprocess the Image for Prediction
img = tf.expand_dims(img, axis=0)  # Expand the dimensions to match the input shape of the model

# Explanation:
# The image needs to be expanded in dimensions since the model expects batches of images.

# Step 15: Use the Model to Predict
p = new_model.predict(img)  # Make a prediction on the image
print("classify")
print(int(p))  # Print the predicted class (0 or 1)

# Explanation:
# The model predicts whether the image is pure or contaminated water.
# The prediction (`p`) is converted into an integer (0 or 1).
