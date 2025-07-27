# Waste Classifier - Smart Waste Management using AI for Image Sorting

This project uses a Convolutional Neural Network (CNN) to classify waste images into categories such as wet, dry, recyclable, and hazardous. The backend is built with Flask to serve the model for real-time image classification.

## Project Structure
- `waste_classifier/` - Main package for model and API code
- `data/` - Place your custom waste image dataset here (not included)
- `venv/` - Python virtual environment (to be created)

## Setup
1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install tensorflow flask numpy pillow scikit-learn
   ```
3. Place your dataset in the `data/` folder, organized by category (e.g., `data/wet/`, `data/dry/`, etc.)

## Steps
- Load and preprocess the image dataset
- Build and train a CNN model
- Evaluate and save the model
- Serve predictions via Flask API
- (Optional) Add a simple frontend for image upload
