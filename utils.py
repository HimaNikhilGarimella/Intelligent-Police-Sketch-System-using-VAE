# src/utils.py
import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .constants import IMAGES_PATH, FEATURES_FILE, ACCURACY_THRESHOLD

class Utils:
    @staticmethod
    def save_image(image, filename):
        """Save image with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.jpg"
        full_path = os.path.join(IMAGES_PATH, full_filename)
        cv2.imwrite(full_path, image)
        return full_path

    @staticmethod
    def calculate_feature_vector(image):
        """Calculate enhanced feature vector for image matching"""
        # Check if image is already grayscale
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistency
        resized = cv2.resize(gray, (64, 64))
        
        features = []
        
        # Histogram features
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features using multiple methods
        edges_canny = cv2.Canny(resized, 100, 200)
        edges_sobel = cv2.Sobel(resized, cv2.CV_64F, 1, 1, ksize=3)
        
        edge_hist_canny = cv2.calcHist([edges_canny], [0], None, [256], [0, 256])
        edge_hist_sobel = cv2.calcHist([np.uint8(np.absolute(edges_sobel))], [0], None, [256], [0, 256])
        
        features.extend(edge_hist_canny.flatten())
        features.extend(edge_hist_sobel.flatten())
        
        # Add texture features
        texture = Utils.calculate_texture_features(resized)
        features.extend(texture)
        
        return np.array(features)

    @staticmethod
    def calculate_texture_features(image):
        """Calculate texture features using GLCM"""
        from skimage.feature import graycomatrix, graycoprops
        
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
        
        features = []
        for prop in properties:
            features.extend(graycoprops(glcm, prop).flatten())
        
        return features

    @staticmethod
    def compare_features(features1, features2):
        """Compare two feature vectors using multiple metrics"""
        # Cosine similarity
        cos_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
        # Euclidean distance (normalized)
        euc_dist = np.linalg.norm(features1 - features2)
        max_dist = np.sqrt(len(features1))
        euc_sim = 1 - (euc_dist / max_dist)
        
        # Combined similarity score
        similarity = 0.7 * cos_sim + 0.3 * euc_sim
        
        return max(0, min(1, similarity))  # Ensure between 0 and 1

class DatabaseManager:
    def __init__(self):
        self.df = pd.read_excel(FEATURES_FILE)
        self.accuracy_metrics = {
            'total_queries': 0,
            'correct_matches': 0,
            'false_positives': 0,
            'similarities': []
        }
    
    def search_similar(self, features, threshold=0.5):
        """Search for similar entries with accuracy tracking"""
        results = []
        self.accuracy_metrics['total_queries'] += 1
        
        for idx, row in self.df.iterrows():
            try:
                img_path = os.path.join(IMAGES_PATH, f"{row['id']}.jpg")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_features = Utils.calculate_feature_vector(img)
                        similarity = Utils.compare_features(features, img_features)
                        
                        if similarity >= threshold:
                            results.append({
                                'id': row['id'],
                                'similarity': similarity,
                                'data': row,
                                'image_path': img_path
                            })
                            
                            # Update accuracy metrics
                            self.accuracy_metrics['similarities'].append(similarity)
                            if similarity >= ACCURACY_THRESHOLD:
                                if self.is_correct_match(row['id'], features):
                                    self.accuracy_metrics['correct_matches'] += 1
                                else:
                                    self.accuracy_metrics['false_positives'] += 1
                
            except Exception as e:
                print(f"Error processing image {row['id']}: {str(e)}")
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:5]  # Return top 5 matches

    def is_correct_match(self, match_id, query_features):
        """Determine if a match is correct based on ground truth"""
        # This should be implemented based on your ground truth data
        # For now, we'll use a simple threshold
        return True if match_id in self.df['id'].values else False

    def get_accuracy_metrics(self):
        """Calculate and return accuracy metrics"""
        total = self.accuracy_metrics['total_queries']
        if total == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'avg_similarity': 0
            }
        
        accuracy = self.accuracy_metrics['correct_matches'] / total
        precision = self.accuracy_metrics['correct_matches'] / (self.accuracy_metrics['correct_matches'] + self.accuracy_metrics['false_positives']) if (self.accuracy_metrics['correct_matches'] + self.accuracy_metrics['false_positives']) > 0 else 0
        avg_similarity = np.mean(self.accuracy_metrics['similarities']) if self.accuracy_metrics['similarities'] else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'avg_similarity': avg_similarity
        }