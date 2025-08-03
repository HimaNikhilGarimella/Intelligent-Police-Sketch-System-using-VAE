# src/model.py
import cv2
import numpy as np
from .constants import IMAGE_SIZE

class SimpleFaceGenerator:
    def __init__(self):
        self.image_size = IMAGE_SIZE

    def generate_sketch(self, features, callback=None):
        """Generate a sketch based on features"""
        # Create base image
        base_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        base_image.fill(255)  # White background
        
        # Get feature values
        face_shape = features.get('face shape', 'Oval').lower()
        eyes = features.get('eyes', 'Medium').lower()
        nose = features.get('nose', 'Medium').lower()
        mouth = features.get('mouth', 'Medium').lower()
        
        # Draw features progressively
        steps = 5
        for i in range(steps):
            progress = (i + 1) / steps
            current_image = base_image.copy()
            
            if i >= 0:  # Step 1: Face shape
                self.draw_face_shape(current_image, face_shape)
            if i >= 1:  # Step 2: Eyes
                self.draw_eyes(current_image, eyes)
            if i >= 2:  # Step 3: Nose
                self.draw_nose(current_image, nose)
            if i >= 3:  # Step 4: Mouth
                self.draw_mouth(current_image, mouth)
            if i >= 4:  # Step 5: Final details
                self.add_details(current_image, features)
            
            # Convert to sketch style
            sketch = self.to_sketch(current_image)
            
            if callback:
                callback(sketch, progress)
        
        return sketch

    def draw_face_shape(self, image, shape):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        if shape == 'round':
            radius = min(width, height) // 3
            cv2.circle(image, center, radius, (0, 0, 0), 2)
        elif shape == 'square':
            size = min(width, height) // 3
            x1 = center[0] - size
            y1 = center[1] - size
            x2 = center[0] + size
            y2 = center[1] + size
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        else:  # oval
            axes = (width // 4, height // 3)
            cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 0), 2)

    def draw_eyes(self, image, size):
        height, width = image.shape[:2]
        eye_y = height // 3
        
        # Size adjustments
        if size == 'small':
            eye_size = (15, 8)
        elif size == 'large':
            eye_size = (25, 12)
        else:  # medium
            eye_size = (20, 10)
        
        # Left eye
        left_eye_x = width // 3
        cv2.ellipse(image, 
                    (left_eye_x, eye_y),
                    eye_size, 0, 0, 360, (0, 0, 0), 2)
        
        # Right eye
        right_eye_x = (2 * width) // 3
        cv2.ellipse(image,
                    (right_eye_x, eye_y),
                    eye_size, 0, 0, 360, (0, 0, 0), 2)

    def draw_nose(self, image, size):
        height, width = image.shape[:2]
        nose_y = height // 2
        
        # Size adjustments
        if size == 'small':
            nose_width = 15
        elif size == 'large':
            nose_width = 25
        else:  # medium
            nose_width = 20
        
        # Draw nose
        center_x = width // 2
        pts = np.array([[center_x, nose_y - nose_width],
                       [center_x - nose_width//2, nose_y + nose_width//2],
                       [center_x + nose_width//2, nose_y + nose_width//2]], 
                      np.int32)
        cv2.polylines(image, [pts], True, (0, 0, 0), 2)

    def draw_mouth(self, image, size):
        height, width = image.shape[:2]
        mouth_y = (2 * height) // 3
        
        # Size adjustments
        if size == 'small':
            mouth_size = (25, 15)
        elif size == 'large':
            mouth_size = (35, 25)
        else:  # medium
            mouth_size = (30, 20)
        
        # Draw mouth
        cv2.ellipse(image,
                    (width//2, mouth_y),
                    mouth_size, 0, 0, 180, (0, 0, 0), 2)

    def add_details(self, image, features):
        """Add additional details based on features"""
        height, width = image.shape[:2]
        
        # Add eyebrows
        eye_y = height // 3 - 20
        left_x = width // 3
        right_x = (2 * width) // 3
        
        cv2.line(image, 
                (left_x - 20, eye_y), 
                (left_x + 20, eye_y), 
                (0, 0, 0), 2)
        cv2.line(image, 
                (right_x - 20, eye_y), 
                (right_x + 20, eye_y), 
                (0, 0, 0), 2)

    def to_sketch(self, image):
        """Convert image to sketch style"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert
        inverted = cv2.bitwise_not(gray)
        
        # Blur
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        
        # Invert back
        sketch = cv2.bitwise_not(blurred)
        
        return sketch