# src/image_processor.py
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import exposure, feature

class ImageProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def preprocess_image(self, image):
        """Comprehensive image preprocessing pipeline"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance contrast
            gray = self.enhance_contrast(gray)
            
            # Denoise
            denoised = self.denoise_image(gray)
            
            # Enhance edges
            edges = self.enhance_edges(denoised)
            
            # Apply artistic effects
            artistic = self.apply_artistic_effects(edges)
            
            return artistic
            
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            return image
    
    def enhance_contrast(self, image):
        """Enhance image contrast using multiple methods"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(image)
        
        # Adaptive Gamma Correction
        gamma = self.estimate_gamma(image)
        gamma_corrected = np.power(image/255.0, gamma)
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        # Combine both enhancements
        enhanced = cv2.addWeighted(cl1, 0.6, gamma_corrected, 0.4, 0)
        
        return enhanced
    
    def estimate_gamma(self, image):
        """Estimate optimal gamma value for correction"""
        mean_intensity = np.mean(image) / 255.0
        return np.log(0.5) / np.log(mean_intensity + 1e-10)
    
    def denoise_image(self, image):
        """Advanced denoising using multiple techniques"""
        # Non-local means denoising
        denoised_nlm = cv2.fastNlMeansDenoising(image)
        
        # Bilateral filter
        denoised_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Combine both methods
        denoised = cv2.addWeighted(denoised_nlm, 0.5, denoised_bilateral, 0.5, 0)
        
        return denoised
    
    def enhance_edges(self, image):
        """Enhanced edge detection and processing"""
        # Canny edge detection
        edges_canny = feature.canny(image, sigma=2)
        
        # Sobel edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize Sobel edges
        edges_sobel = ((edges_sobel - edges_sobel.min()) * (255.0 / (edges_sobel.max() - edges_sobel.min()))).astype(np.uint8)
        
        # Combine edges
        edges_combined = cv2.addWeighted(edges_canny.astype(np.uint8), 0.7, edges_sobel, 0.3, 0)
        
        return edges_combined
    
    def apply_artistic_effects(self, image):
        """Apply artistic effects for sketch-like appearance"""
        # Convert to tensor
        tensor_image = torch.from_numpy(image).float().to(self.device)
        if len(tensor_image.shape) == 2:
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
        
        # Apply various artistic filters
        sketch = self.pencil_sketch_effect(image)
        
        # Convert back to numpy
        result = sketch.astype(np.uint8)
        
        return result
    
    def pencil_sketch_effect(self, image):
        """Create pencil sketch effect"""
        # Invert image
        inverted = 255 - image
        
        # Create blur
        blur = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Invert blur
        inverted_blur = 255 - blur
        
        # Create sketch
        sketch = cv2.divide(image, inverted_blur, scale=256.0)
        
        # Enhance sketch details
        enhanced_sketch = self.enhance_sketch_details(sketch)
        
        return enhanced_sketch
    
    def enhance_sketch_details(self, sketch):
        """Enhance sketch details"""
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(sketch, -1, kernel)
        
        # Enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        # Add texture
        texture = self.create_paper_texture(enhanced.shape)
        textured = cv2.multiply(enhanced, texture, scale=1/255.0)
        
        return textured
    
    def create_paper_texture(self, shape):
        """Create paper-like texture"""
        texture = np.random.normal(250, 10, shape).astype(np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
        return texture
    
    def postprocess_sketch(self, sketch):
        """Final post-processing of the sketch"""
        # Enhance contrast
        sketch = exposure.rescale_intensity(sketch)
        
        # Smooth edges
        sketch = ndimage.gaussian_filter(sketch, sigma=0.5)
        
        # Ensure proper range
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        
        return sketch
    
    def process_batch(self, images):
        """Process a batch of images"""
        processed_images = []
        for image in images:
            processed = self.preprocess_image(image)
            processed = self.postprocess_sketch(processed)
            processed_images.append(processed)
        return processed_images
    
    def save_processed_image(self, image, path):
        """Save processed image with optimal quality"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save with high quality
            pil_image.save(path, quality=95, optimize=True)
            return True
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
