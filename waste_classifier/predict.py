from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'waste_cnn_model.h5')
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_HEIGHT = 128
IMG_WIDTH = 128

def predict_image(image_path, model_path=MODEL_PATH, categories=CATEGORIES):
    model = load_model(model_path)
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    if pred.shape[1] != len(categories):
        raise ValueError(f"Model output shape {pred.shape} does not match number of categories {len(categories)}.")
    class_idx = np.argmax(pred, axis=1)[0]
    if class_idx >= len(categories):
        raise ValueError(f"Predicted class index {class_idx} out of range for categories {categories}.")
    return categories[class_idx], float(np.max(pred))

# Example usage (uncomment to test)
# category, confidence = predict_image('path_to_image.jpg')
# print(f"Predicted: {category} (confidence: {confidence:.2f})")
