# Fire Detection with Custom CNN

## Overview

This project aims to detect fire in images using a Convolutional Neural Network (CNN). To improve the accuracy and reduce the training time of the model, different features of the images are extracted using cv2 library, and fed into a multi-input CNN.

## Components

### Data Preparation

- **Dataset**: The dataset is divided into two directories: `fire` and `non_fire`, each containing images for respective classes.
- **Data Loading**: Images are loaded into a Pandas DataFrame along with their labels (`fire` or `non_fire`).

### Image Preprocessing

Images are preprocessed using three types of masks:
1. **Edge Mask**: Highlights edges in the image using Canny edge detection.
2. **Heatmap Mask**: Identifies bright regions that may indicate fire.
3. **Color Mask**: Isolates fire-like colors (reds, oranges, yellows) to highlight potential fire areas.

### Custom Data Generator

A custom data generator is implemented to:
- Load and preprocess images.
- Apply the three types of masks.
- Normalize the images.
- Yield batches of three masked images and their corresponding labels for model training.

### Model Architecture

A CNN model with three input branches is designed to process the three masked images. The model includes:
- **Convolutional Layers**: To extract features from images.
- **Pooling Layers**: To reduce spatial dimensions.
- **Dense Layers**: To classify images into fire or non-fire.

The model is compiled using the Adam optimizer with a learning rate of 0.0001 and binary crossentropy loss.

### Training and Evaluation

- **Training**: The model is trained with a training set and validated with a validation set using custom data generators. Callbacks are used to manage learning rate adjustments and save the best model.
- **Evaluation**: The model is evaluated on a test set to measure loss and accuracy.

### Prediction and Visualization

- **Prediction**: The model predicts the class of images (fire or non-fire) based on the preprocessed inputs.
- **Visualization**: A set of test images with predictions is displayed to assess the model's performance visually.

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PIL
- tqdm

## Usage

1. **Prepare the Dataset**: Ensure the dataset directories are properly set up and contain images in the `fire` and `non_fire` categories.
2. **Run the Code**: Execute the provided script to load data, preprocess images, train the model, and evaluate performance.
3. **Visualize Results**: View the predictions on test images to understand the model's performance.

## Notes

- **Image Size**: The images are resized to 150x150 pixels.
- **Masks**: Custom masks are designed to enhance features relevant to fire detection.
