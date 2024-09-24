# Water Quality Prediction Model

This project uses a TensorFlow Keras model to classify images of water as either **Pure** or **Contaminated**. The model is trained on water images and is capable of predicting the water quality based on test images provided in the dataset.

## Table of Contents

- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Model Loading](#model-loading)
- [Image Preprocessing](#image-preprocessing)
- [Prediction](#prediction)
- [Usage](#usage)
- [References](#references)

## Requirements

To run this project, ensure you have the following dependencies installed:

- TensorFlow >= 2.x
- Keras >= 2.x
- OpenCV (cv2)
- Pillow (PIL)

You can install the required packages using the following command:

```bash
pip install tensorflow keras opencv-python pillow
```

## Dataset Structure

The dataset should be organized as follows:

Wat_Dataset/
│
├── model/
│   └── my_model_Water.h5     # Trained model
│
├── Testing/
    ├── Pure/
    │   ├── P2.jpg
    │   ├── P5.jpg
    │   └── ... (other pure water images)
    │
    └── Contaminated/
        ├── C1.jpg
        ├── C2.jpg
        └── ... (other contaminated water images)

## Model Loading

The pre-trained Keras model is loaded using TensorFlow's load_model function:
```bash
Model Loading
The pre-trained Keras model is loaded using TensorFlow's load_model function:
```

## Image Preprocessing

```
img = cv2.imread('path_to_image')
img = tf.expand_dims(img, axis=0)
```

## Prediction
```
image_paths = [r'path_to_image1', r'path_to_image2', ...]
for img_path in image_paths:
    img = cv2.imread(img_path)
    img = tf.expand_dims(img, axis=0)
    prediction = new_model.predict(img)
    print(f'Prediction for {img_path}: {prediction}')
```

## Usage
1. Load the trained model.
2. Preprocess the images by expanding their dimensions to match the model input.
3. Run the model's predict() method to classify the images.
4. The output predictions will indicate whether the water is pure or contaminated.
