import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json


# Load the model
fish_model = tf.keras.models.load_model("fish_classifier_model.keras")

# Load the class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

for i, name in enumerate(class_names):
    print(i, name)

def predict(image):
    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = fish_model.predict(image_array)
    class_index = np.argmax(prediction)

    return class_names[class_index]

fish = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Fish Classifier",
    description="Upload the image of a fish to know its class"
)

fish.launch()

