# src/constants.py
import os

IMAGES_PATH = r"C:\Users\HP\OneDrive\Desktop\Major Project\front"
FEATURES_FILE = r"C:\Users\HP\OneDrive\Desktop\Major Project\Copy of person_edited(1).xlsx"
IMAGE_SIZE = 256
MAX_RESULTS = 5
SIMILARITY_THRESHOLD = 0.5

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gan_generator.pth')

FEATURE_WEIGHTS = {
    'eyes': 0.3,
    'nose': 0.2,
    'mouth': 0.2,
    'face_shape': 0.3
}

# Evaluation metrics
ACCURACY_THRESHOLD = 0.75  # Minimum similarity for a "correct" match