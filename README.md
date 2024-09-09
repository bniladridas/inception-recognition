# 🌟 Image Classification with InceptionV3

_A Deep Learning Masterpiece_

---

## 🛠️ **Tech Stack**  
![TensorFlow](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/240px-TensorFlowLogo.svg.png) &nbsp; ![Keras](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/240px-Keras_logo.svg.png) &nbsp; ![InceptionV3](https://www.mdpi.com/symmetry/symmetry-14-02679/article_deploy/html/images/symmetry-14-02679-g007-550.jpg) &nbsp; ![NumPy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/240px-NumPy_logo_2020.svg.png)

---

## 🌍 **Overview**

Welcome to the world of cutting-edge **image classification** powered by **InceptionV3**. This project leverages a **pre-trained model** from **ImageNet** to classify images with incredible accuracy, capable of distinguishing a wide variety of objects in no time. Your go-to tool for exploring deep learning concepts!

---

## 🚀 **Features**
- **InceptionV3 Architecture**: State-of-the-art model built for high-performance image classification.
- **Pre-trained on ImageNet**: Get immediate results with optimized weights from one of the largest datasets.
- **Plug-and-Play Python Script**: Simply test your own images effortlessly.
- **Perfect for Learning**: Ideal for gaining hands-on experience with deep learning and advanced image classification techniques.

---

## 🖼️ **Example Results**

Here’s a sneak peek of what the **InceptionV3** model can do!

![Image Preview](/img/train.jpg)

**Predictions**:
- 🚂 **Freight Car** (Confidence: 85%)
- ⚡ **Electric Locomotive** (Confidence: 8%)
- 🚋 **Passenger Car** (Confidence: 1%)

---

## ⚙️ **Installation**

Make sure to install the necessary dependencies:

```bash
pip install tensorflow keras numpy
```

---

## �‍♂️ **Usage**

To classify an image using the `recognize_object` function, follow these steps:

1. Ensure you have the required dependencies installed.
2. Use the following Python script to classify your image:

```python
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
image_path = '/path/to/your/image.jpg'  # Path to the image file
recognize_object(image_path)  # Call the function to recognize objects in the image
```

Replace `/path/to/your/image.jpg` with the path to your image file.

---

## �🎓 **Acknowledgments**

Special thanks to:
- **TensorFlow Keras Applications** for providing the InceptionV3 model.
- **ImageNet** for the class indices: [Download](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json).

---

## 📜 **License**

Licensed under the [MIT License](https://github.com/niladrridas/image-classification/blob/main/LICENSE).

---

## 🏁 **Getting Started**

Start classifying images in three simple steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/niladrridas/imageclassification.git
   ```

---

Now, you’re all set to dive into **image classification** and harness the power of deep learning! 🌐