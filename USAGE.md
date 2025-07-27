# Smart Waste Management using AI for Image Sorting

## Project Usage Guide

### 1. Setup
- Ensure you have Python 3.8+ installed.
- Create and activate a virtual environment:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- Install dependencies:
  ```powershell
  pip install tensorflow flask numpy pillow scikit-learn
  ```

### 2. Prepare Dataset
- Place your waste images in the following structure:
  ```
  data/Garbage classification/Garbage classification/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
  ```
- Each folder should contain images of that waste type.

### 3. Train the Model
- Run the training script:
  ```powershell
  python waste_classifier/train.py
  ```
- The trained model will be saved as `waste_classifier/waste_cnn_model.h5`.

### 4. Start the Backend API
- Run the Flask app:
  ```powershell
  python waste_classifier/app.py
  ```
- The API will be available at `http://127.0.0.1:5000/predict`.

### 5. Use the Frontend
- Open `waste_classifier/frontend.html` in your web browser.
- Upload an image of waste and click "Classify Waste".
- The predicted category and confidence will be displayed.

---

## Project Structure
- `waste_classifier/data_utils.py` - Loads and preprocesses image data
- `waste_classifier/model.py` - Defines the CNN model
- `waste_classifier/train.py` - Trains and evaluates the model
- `waste_classifier/predict.py` - Loads the model and predicts on new images
- `waste_classifier/app.py` - Flask backend for image classification
- `waste_classifier/frontend.html` - Beautiful UI for uploading and classifying images

---

## Tips
- For best results, use clear, well-lit images.
- You can improve accuracy by adding more labeled images to each category.
- To deploy, consider using a production WSGI server (e.g., gunicorn) and hosting the frontend on a web server.
