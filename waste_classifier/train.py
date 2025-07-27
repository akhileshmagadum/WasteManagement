# This file will be used to train and evaluate the CNN model
from data_utils import load_and_preprocess_data, get_augmented_datagen
from model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'waste_cnn_model.h5')

if __name__ == "__main__":
    # Load all data and split
    X, y = load_and_preprocess_data(return_all=True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    model = build_cnn_model()

    # Data augmentation
    datagen = get_augmented_datagen()
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    # Fit using data augmentation
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train)//32,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.2f}")
    print(f"Model saved to {MODEL_PATH}")
