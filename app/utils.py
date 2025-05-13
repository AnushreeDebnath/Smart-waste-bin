
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

model = load_model(os.path.join("model", "waste_classifier.h5"))
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def predict_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return class_names[np.argmax(predictions)]
