# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

# Load pre-trained model
new_model = tf.keras.models.load_model(r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\model\my_model_Water.h5')

# Display model summary
new_model.summary()

# Paths for pure and contaminated images
pure_images_paths = [
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P2.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P5.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P23.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P27.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P29.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P31.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P32.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Pure\P33.jpg'
]

contaminated_images_paths = [
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Contaminated\C1.jpg',
    r'C:\Users\Salman Ahmed\Desktop\Wat_Dataset\Testing\Contaminated\C2.jpg',
    # Add all the paths up to C97 as in the original script...
]

# Function to process and predict images
def predict_images(image_paths):
    predictions = []
    for path in image_paths:
        img = cv2.imread(path)                 # Read image
        img = tf.expand_dims(img, axis=0)      # Expand dimensions
        prediction = new_model.predict(img)    # Predict
        predictions.append(prediction)         # Store prediction
    return predictions

# Predict for pure and contaminated images
pure_predictions = predict_images(pure_images_paths)
contaminated_predictions = predict_images(contaminated_images_paths)

# Print predictions
print("Predictions for pure images:")
for idx, pred in enumerate(pure_predictions):
    print(f"Image {idx+1}: {pred}")

print("\nPredictions for contaminated images:")
for idx, pred in enumerate(contaminated_predictions):
    print(f"Image {idx+1}: {pred}")
