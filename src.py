# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained InceptionV3 model with weights trained on ImageNet
model = InceptionV3(weights='imagenet')

# Function for object recognition
def recognize_object(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))  # Load image and resize to 299x299 pixels
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's input shape
    img_array = preprocess_input(img_array)  # Preprocess the image array for the InceptionV3 model

    # Make predictions
    predictions = model.predict(img_array)  # Predict the probabilities for each class

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode the top 3 predictions

    # Display the top predictions
    print("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):  # Iterate over the top predictions
        print(f"{i + 1}: {label} ({score:.2f})")  # Print the label and score for each prediction

# Example usage
image_path = '/Users/niladridas/computer_vision/train.jpg'  # Path to the image file
recognize_object(image_path)  # Call the function to recognize objects in the image