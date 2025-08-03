# src/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .constants import ACCURACY_THRESHOLD

class EvaluationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.total_predictions = 0
        self.similarities = []
    
    def update(self, similarity, is_correct_match):
        self.similarities.append(similarity)
        self.total_predictions += 1
        
        if similarity >= ACCURACY_THRESHOLD:
            if is_correct_match:
                self.true_positives += 1
            else:
                self.false_positives += 1
    
    def get_metrics(self):
        if self.total_predictions == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'avg_similarity': 0
            }
        
        accuracy = self.true_positives / self.total_predictions
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        avg_similarity = np.mean(self.similarities) if self.similarities else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'avg_similarity': avg_similarity
        }