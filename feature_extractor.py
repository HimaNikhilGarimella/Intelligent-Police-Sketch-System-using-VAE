# src/feature_extractor.py
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from .constants import FEATURE_WEIGHTS

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def extract_features(self, image):
        """Extract comprehensive features from an image"""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Extract face shape
            features['face_shape'] = self.determine_face_shape(face_roi)
            
            # Extract eyes features
            features['eyes'] = self.extract_eye_features(face_roi)
            
            # Extract nose features
            features['nose'] = self.extract_nose_features(face_roi)
            
            # Extract mouth features
            features['mouth'] = self.extract_mouth_features(face_roi)
            
            # Extract additional features
            features.update(self.extract_additional_features(face_roi))
        
        return features
    
    def determine_face_shape(self, face_roi):
        """Determine face shape using contour analysis"""
        # Find contours
        contours, _ = cv2.findContours(
            cv2.threshold(face_roi, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return "oval"  # default
        
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Determine shape based on metrics
        if circularity > 0.85:
            return "round"
        elif aspect_ratio > 0.85:
            return "square"
        else:
            return "oval"
    
    def extract_eye_features(self, face_roi):
        """Extract detailed eye features"""
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        
        if len(eyes) >= 2:
            # Calculate average eye size
            avg_size = np.mean([w * h for (x, y, w, h) in eyes])
            face_area = face_roi.shape[0] * face_roi.shape[1]
            eye_ratio = avg_size / face_area
            
            # Determine eye size category
            if eye_ratio < 0.03:
                return "small"
            elif eye_ratio > 0.06:
                return "large"
            else:
                return "medium"
        
        return "medium"  # default
    
    def extract_nose_features(self, face_roi):
        """Extract nose features using image processing"""
        height, width = face_roi.shape
        nose_roi = face_roi[height//3:2*height//3, width//3:2*width//3]
        
        # Apply edge detection
        edges = cv2.Canny(nose_roi, 100, 200)
        
        # Calculate nose width relative to face
        nose_width = np.sum(edges > 0) / (nose_roi.shape[0] * nose_roi.shape[1])
        
        if nose_width < 0.15:
            return "small"
        elif nose_width > 0.25:
            return "large"
        else:
            return "medium"
    
    def extract_mouth_features(self, face_roi):
        """Extract mouth features using image processing"""
        height, width = face_roi.shape
        mouth_roi = face_roi[2*height//3:, width//4:3*width//4]
        
        # Apply edge detection
        edges = cv2.Canny(mouth_roi, 100, 200)
        
        # Calculate mouth width relative to face
        mouth_width = np.sum(edges > 0) / (mouth_roi.shape[0] * mouth_roi.shape[1])
        
        if mouth_width < 0.2:
            return "small"
        elif mouth_width > 0.3:
            return "large"
        else:
            return "medium"
    
    def extract_additional_features(self, face_roi):
        """Extract additional facial features"""
        features = {}
        
        # Calculate facial symmetry
        height, width = face_roi.shape
        left_half = face_roi[:, :width//2]
        right_half = cv2.flip(face_roi[:, width//2:], 1)
        symmetry = np.mean(np.abs(left_half - right_half))
        features['symmetry'] = float(symmetry)
        
        # Calculate texture features
        texture = self.calculate_texture_features(face_roi)
        features['texture'] = texture
        
        return features
    
    def calculate_texture_features(self, image):
        """Calculate texture features using GLCM"""
        from skimage.feature import graycomatrix, graycoprops
        
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        glcm = graycomatrix(image, distances=distances, angles=angles, 
                           symmetric=True, normed=True)
        
        features = []
        for prop in properties:
            features.extend(graycoprops(glcm, prop).flatten())
        
        return np.mean(features)
    
    def normalize_features(self, features):
        """Normalize extracted features"""
        numeric_features = []
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                numeric_features.append(value)
        
        if numeric_features:
            normalized = self.scaler.fit_transform(np.array(numeric_features).reshape(1, -1))
            
            # Update features with normalized values
            idx = 0
            for feature in features:
                if isinstance(features[feature], (int, float)):
                    features[feature] = float(normalized[0, idx])
                    idx += 1
        
        return features