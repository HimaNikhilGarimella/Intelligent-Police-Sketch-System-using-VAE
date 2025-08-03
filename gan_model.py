# src/gan_model.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Layer 1: (latent_dim, 1, 1) -> (512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Layer 2: (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 3: (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Layer 4: (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Output layer: (64, 32, 32) -> (1, 64, 64)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        return self.main(x)

class GANSketchGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = 100
        self.generator = Generator(self.latent_dim).to(self.device)
        self.load_model()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def load_model(self):
        """Load pre-trained model or initialize with random weights"""
        try:
            # Get absolute path to model file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'models', 'gan_generator.pth')
            
            if os.path.exists(model_path):
                self.generator.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )
                self.generator.eval()
                print("Model loaded successfully")
            else:
                print(f"Model file not found at: {model_path}")
                print("Using random weights for demonstration")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random weights for demonstration")
    
    def features_to_vector(self, features):
        """Convert feature dictionary to latent vector"""
        # Create base latent vector
        z = torch.randn(self.latent_dim).to(self.device)
        
        # Feature mapping
        feature_map = {
            'face shape': {'round': 0.0, 'oval': 0.5, 'square': 1.0},
            'eyes': {'small': 0.0, 'medium': 0.5, 'large': 1.0},
            'nose': {'small': 0.0, 'medium': 0.5, 'large': 1.0},
            'mouth': {'small': 0.0, 'medium': 0.5, 'large': 1.0}
        }
        
        # Modify latent vector based on features
        for feature, value in features.items():
            if feature.lower() in feature_map and value.lower() in feature_map[feature.lower()]:
                idx = list(feature_map.keys()).index(feature.lower())
                start_idx = idx * (self.latent_dim // len(feature_map))
                end_idx = (idx + 1) * (self.latent_dim // len(feature_map))
                z[start_idx:end_idx] *= feature_map[feature.lower()][value.lower()]
        
        return z
    
    def generate_sketch(self, features, callback=None):
        """Generate sketch using GAN"""
        steps = 10
        
        with torch.no_grad():
            # Convert features to latent vector
            z = self.features_to_vector(features)
            
            for i in range(steps):
                progress = (i + 1) / steps
                
                # Interpolate between random and feature vector
                current_z = torch.lerp(
                    torch.randn(self.latent_dim, device=self.device),
                    z,
                    progress
                )
                
                # Generate image
                fake = self.generator(current_z.unsqueeze(0))
                
                # Convert to numpy array
                sketch = fake.squeeze().cpu().numpy()
                sketch = ((sketch + 1) * 127.5).astype(np.uint8)
                
                # Resize to desired size
                sketch = cv2.resize(sketch, (256, 256))
                
                if callback:
                    callback(sketch, progress)
            
            return sketch