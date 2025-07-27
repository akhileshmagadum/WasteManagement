import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split
import random

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
# Update the path to the new dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/Garbage classification/Garbage classification')
# Update categories to match the new dataset
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_and_preprocess_data(data_dir=DATA_DIR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, categories=CATEGORIES, test_size=0.2, shuffle=True, return_all=False):
    images = []
    labels = []
    for idx, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img = img_to_array(img) / 255.0
                images.append(img)
                labels.append(idx)
            except Exception as e:
                # Only print once per 100 errors to avoid flooding
                if random.randint(1, 100) == 1:
                    print(f"Error loading {img_path}: {e}")
    X = np.array(images)
    y = np.array(labels)
    if shuffle:
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]
    if return_all:
        return X, y
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Data augmentation generator for use in training
def get_augmented_datagen():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Example usage (uncomment to test)
# X_train, X_test, y_train, y_test = load_and_preprocess_data()
