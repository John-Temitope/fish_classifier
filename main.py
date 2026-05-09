import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
import gradio as gr
import numpy as np
from PIL import Image
import json


# Load the model
fish_model = tf.keras.models.load_model("fish_classifier_model.keras")

# Load the class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# Function to predict image
def predict(image):
    image = image.resize((180, 180))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    prediction = fish_model.predict(image_array)
    class_index = np.argmax(prediction)

    return class_names[class_index]


# Call the model and predict image
fish = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Fish Classifier",
    description="Upload the image of a fish to know its class"
)

fish.launch()


