import customtkinter as ctk
from PIL import Image, ImageTk, ImageEnhance
import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from datetime import datetime
import cv2

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class CriminalMatchingSystem:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Criminal Identification System")
        self.window.geometry("1000x600")
        self.window.resizable(True, True)
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize metrics tracking
        self.total_searches = 0
        self.successful_matches = 0
        self.search_history = []
        
        # Load data
        self.load_data()
        self.setup_ui()

    def load_data(self):
        try:
            # Load features from Excel
            self.df = pd.read_excel(r"C:\Users\HP\OneDrive\Desktop\Major Project\Copy of person_edited(1).xlsx")
            print("Columns in dataset:", self.df.columns.tolist())
            
            # Clean up data
            for col in ['sex', 'eyes', 'location', 'hair']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).fillna('Unknown')
            
            for col in ['height', 'weight']:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Ensure ID is string
            if 'id' in self.df.columns:
                self.df['id'] = self.df['id'].astype(str)
            
            # Load images
            self.image_folder = r"C:\Users\HP\OneDrive\Desktop\Major Project\front"
            self.images = {}
            for img_file in os.listdir(self.image_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_id = os.path.splitext(img_file)[0]
                    img_path = os.path.join(self.image_folder, img_file)
                    self.images[img_id] = img_path
            
            print(f"Loaded {len(self.df)} records and {len(self.images)} images")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def setup_ui(self):
        # Create main frames
        self.input_frame = ctk.CTkFrame(self.window)
        self.input_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        self.results_frame = ctk.CTkFrame(self.window)
        self.results_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        self.create_input_area()
        self.create_results_area()
    def create_input_area(self):
        # Title
        ctk.CTkLabel(
            self.input_frame,
            text="Criminal Description",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=5)
        
        # Description text box
        self.description_text = ctk.CTkTextbox(
            self.input_frame,
            width=300,
            height=200,
            font=ctk.CTkFont(size=12)
        )
        self.description_text.pack(pady=5, padx=5)
        
        # Example placeholder text
        placeholder = ("Example:\nMale, brown eyes\nHeight: 175 cm\n"
                      "Weight: 70 kg\nLocation: New York\n"
                      "Black hair\nMedium build")
        self.description_text.insert("1.0", placeholder)
        
        # Search Button
        self.search_btn = ctk.CTkButton(
            self.input_frame,
            text="Search Matches",
            command=self.search_matches,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.search_btn.pack(pady=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self.input_frame)
        self.progress.pack(pady=5, padx=5, fill="x")
        self.progress.set(0)

    def create_results_area(self):
        ctk.CTkLabel(
            self.results_frame,
            text="Matches",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        self.results_scroll = ctk.CTkScrollableFrame(
            self.results_frame,
            width=600,
            height=500
        )
        self.results_scroll.pack(fill="both", expand=True, padx=5, pady=5)

    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation, lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in string.punctuation 
            and token not in self.stop_words
            and token.strip()
        ]
        
        return tokens

    def calculate_similarity(self, input_text, record):
        try:
            # Convert record values to string, excluding numpy arrays
            record_text = []
            for key, value in record.items():
                if pd.notna(value) and not isinstance(value, np.ndarray):
                    if isinstance(value, (int, float)):
                        record_text.append(str(int(value)))
                    else:
                        record_text.append(str(value))
            
            record_text = ' '.join(record_text)
            
            # Get tokens
            input_tokens = set(self.preprocess_text(input_text))
            record_tokens = set(self.preprocess_text(record_text))
            
            if not input_tokens or not record_tokens:
                return 0.0
            
            # Calculate similarity
            intersection = len(input_tokens.intersection(record_tokens))
            union = len(input_tokens.union(record_tokens))
            
            if union == 0:
                return 0.0
            
            # Basic similarity
            similarity = intersection / union
            
            # Weight key terms
            key_terms = {
                'sex': ['male', 'female'],
                'eyes': ['brown', 'blue', 'black', 'green'],
                'location': str(record.get('location', '')).lower()
            }
            
            # Calculate bonus for matching key terms
            bonus = 0
            input_lower = input_text.lower()
            
            for category, terms in key_terms.items():
                if isinstance(terms, str):
                    if terms in input_lower:
                        bonus += 0.2
                else:
                    for term in terms:
                        if term in input_lower:
                            bonus += 0.2
                            break
            
            return min(similarity + bonus, 1.0)
            
        except Exception as e:
            print(f"Error calculating similarity for record: {e}")
            return 0.0
    def search_matches(self):
        try:
            # Clear previous results
            for widget in self.results_scroll.winfo_children():
                widget.destroy()
            
            self.progress.set(0.2)
            self.search_btn.configure(state="disabled")
            
            description = self.description_text.get("1.0", "end-1c")
            
            # Find matches with lower threshold
            matches = []
            for idx, record in self.df.iterrows():
                try:
                    record_dict = record.to_dict()
                    record_dict = {k: v for k, v in record_dict.items() 
                                 if not isinstance(v, np.ndarray)}
                    
                    similarity = self.calculate_similarity(description, record_dict)
                    if similarity > 0.1:  # Lower threshold for more matches
                        matches.append((record_dict, similarity))
                except Exception as e:
                    print(f"Error processing record {idx}: {e}")
                    continue
            
            self.progress.set(0.6)
            
            # Sort and display matches
            if matches:
                matches.sort(key=lambda x: x[1], reverse=True)
                for record, similarity in matches[:10]:  # Show top 10 matches
                    self.create_match_card(record, similarity)
            else:
                # Show closest matches even if similarity is low
                all_matches = []
                for idx, record in self.df.iterrows():
                    try:
                        record_dict = record.to_dict()
                        similarity = self.calculate_similarity(description, record_dict)
                        all_matches.append((record_dict, similarity))
                    except Exception as e:
                        continue
                
                if all_matches:
                    all_matches.sort(key=lambda x: x[1], reverse=True)
                    ctk.CTkLabel(
                        self.results_scroll,
                        text="No exact matches found. Showing closest matches:",
                        font=ctk.CTkFont(size=16)
                    ).pack(pady=10)
                    
                    for record, similarity in all_matches[:5]:
                        self.create_match_card(record, similarity)
            
            self.progress.set(1)
            self.search_btn.configure(state="normal")
            
        except Exception as e:
            print(f"Error in search_matches: {e}")
            self.progress.set(0)
            self.search_btn.configure(state="normal")
            error_label = ctk.CTkLabel(
                self.results_scroll,
                text=f"Error: {str(e)}",
                text_color="red"
            )
            error_label.pack(pady=20)

    def create_match_card(self, record, similarity):
        try:
            # Create card frame
            card = ctk.CTkFrame(self.results_scroll)
            card.pack(fill="x", pady=2, padx=2)
            
            # Load and display image using ID
            try:
                record_id = str(record.get('id', ''))
                if record_id in self.images:
                    image_path = self.images[record_id]
                    
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        img = img.resize((80, 100), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        
                        img_label = ctk.CTkLabel(card, image=photo, text="")
                        img_label.image = photo
                        img_label.pack(side="left", padx=2, pady=2)
            except Exception as e:
                print(f"Error loading image: {e}")
            
            # Information frame
            info_frame = ctk.CTkFrame(card)
            info_frame.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            
            # Display match details
            ctk.CTkLabel(
                info_frame,
                text=f"ID: {record.get('id', 'Unknown')} ({int(similarity*100)}% match)",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(anchor="w")
            
            # Display other details
            details = ['name', 'sex', 'eyes', 'height', 'weight', 'location', 'hair']
            for col in details:
                if col in record and pd.notna(record[col]):
                    ctk.CTkLabel(
                        info_frame,
                        text=f"{col.title()}: {record[col]}",
                        font=ctk.CTkFont(size=11)
                    ).pack(anchor="w")
            
            # Confirm match button
            ctk.CTkButton(
                info_frame,
                text="Confirm",
                command=lambda r=record: self.confirm_match(r),
                font=ctk.CTkFont(size=11),
                height=25,
                width=80
            ).pack(anchor="w", pady=2)
            
        except Exception as e:
            print(f"Error creating match card: {e}")
            error_label = ctk.CTkLabel(
                card,
                text=f"Error displaying match: {str(e)}"
            )
            error_label.pack(padx=10, pady=10)
    def confirm_match(self, record):
        confirm_window = ctk.CTkToplevel(self.window)
        confirm_window.title("Criminal Match Confirmation")
        confirm_window.geometry("1000x700")
        
        # Main container
        main_frame = ctk.CTkFrame(confirm_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Image and sliders side by side
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=5)
        
        # Left side - Image
        image_frame = ctk.CTkFrame(content_frame)
        image_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Right side - Scrollable frame for sliders
        slider_container = ctk.CTkScrollableFrame(content_frame, width=300)
        slider_container.pack(side="right", fill="both", expand=True, padx=5)
        
        # Load and display image
        try:
            record_id = str(record.get('id', ''))
            if record_id in self.images:
                image_path = self.images[record_id]
                if os.path.exists(image_path):
                    # Load image with OpenCV for facial feature manipulation
                    self.cv_image = cv2.imread(image_path)
                    self.current_image = Image.open(image_path)
                    self.current_image = self.current_image.resize((400, 500))
                    self.photo = ImageTk.PhotoImage(self.current_image)
                    
                    self.img_label = ctk.CTkLabel(image_frame, image=self.photo, text="")
                    self.img_label.image = self.photo
                    self.img_label.pack(pady=5)
                    
                    # Store original images
                    self.original_cv_image = self.cv_image.copy()
                    self.original_image = self.current_image.copy()
        except Exception as e:
            print(f"Error loading image: {e}")

        # Facial feature sliders
        sliders_info = {
            # Face Structure
            'face_width': ('Face Width', 50, 150, 100),
            'face_height': ('Face Height', 50, 150, 100),
            'jaw_width': ('Jaw Width', 50, 150, 100),
            
            # Eyes
            'eye_size': ('Eye Size', 50, 150, 100),
            'eye_spacing': ('Eye Distance', 50, 150, 100),
            'eye_height': ('Eye Height', 50, 150, 100),
            'eye_color': ('Eye Color', 0, 360, 180),
            
            # Nose
            'nose_width': ('Nose Width', 50, 150, 100),
            'nose_length': ('Nose Length', 50, 150, 100),
            'nose_bridge': ('Nose Bridge', 50, 150, 100),
            
            # Mouth
            'mouth_width': ('Mouth Width', 50, 150, 100),
            'mouth_height': ('Mouth Height', 50, 150, 100),
            'lip_thickness': ('Lip Thickness', 50, 150, 100),
            
            # Hair
            'hair_length': ('Hair Length', 50, 150, 100),
            'hair_volume': ('Hair Volume', 50, 150, 100),
            'hair_color': ('Hair Color', 0, 360, 180),
            
            # Image Adjustments
            'brightness': ('Brightness', 50, 150, 100),
            'contrast': ('Contrast', 50, 150, 100),
            'sharpness': ('Sharpness', 50, 150, 100)
        }
        
        # Create slider groups
        groups = {
            'Face Structure': ['face_width', 'face_height', 'jaw_width'],
            'Eyes': ['eye_size', 'eye_spacing', 'eye_height', 'eye_color'],
            'Nose': ['nose_width', 'nose_length', 'nose_bridge'],
            'Mouth': ['mouth_width', 'mouth_height', 'lip_thickness'],
            'Hair': ['hair_length', 'hair_volume', 'hair_color'],
            'Image': ['brightness', 'contrast', 'sharpness']
        }
        
        self.image_sliders = {}
        
        # Create sliders by groups
        for group_name, slider_keys in groups.items():
            group_frame = ctk.CTkFrame(slider_container)
            group_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(
                group_frame,
                text=group_name,
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(pady=5)
            
            for key in slider_keys:
                name, min_val, max_val, default = sliders_info[key]
                
                slider_frame = ctk.CTkFrame(group_frame)
                slider_frame.pack(fill="x", pady=2)
                
                label_frame = ctk.CTkFrame(slider_frame)
                label_frame.pack(fill="x")
                
                ctk.CTkLabel(
                    label_frame,
                    text=name,
                    font=ctk.CTkFont(size=12)
                ).pack(side="left", padx=5)
                
                value_label = ctk.CTkLabel(
                    label_frame,
                    text=f"{default}%",
                    font=ctk.CTkFont(size=12)
                )
                value_label.pack(side="right", padx=5)
                
                slider = ctk.CTkSlider(
                    slider_frame,
                    from_=min_val,
                    to=max_val,
                    number_of_steps=100,
                    command=lambda v, k=key, l=value_label: self.update_feature(k, v, l)
                )
                slider.pack(fill="x", padx=5, pady=2)
                slider.set(default)
                
                self.image_sliders[key] = slider
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Confirm Match",
            command=lambda: self.save_confirmation(record, confirm_window),
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Reset Image",
            command=self.reset_image,
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=confirm_window.destroy,
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)

    def update_feature(self, feature_type, value, value_label=None):
        """Update facial features and image"""
        try:
            if not hasattr(self, 'original_cv_image'):
                return
            
            if value_label:
                value_label.configure(text=f"{int(value)}%")
            
            # Create a copy of the original image
            cv_img = self.original_cv_image.copy()
            
            # Convert value to scale factor (0.5 to 1.5)
            scale_factor = value / 100
            
            # Apply transformations based on feature type
            if feature_type in ['eye_color', 'hair_color']:
                # Color adjustments
                cv_img = self.adjust_color_feature(cv_img, feature_type, value)
            elif feature_type in ['eye_size', 'nose_width', 'mouth_width']:
                # Size adjustments
                cv_img = self.adjust_size_feature(cv_img, feature_type, scale_factor)
            elif feature_type in ['face_width', 'face_height']:
                # Face shape adjustments
                cv_img = self.adjust_face_feature(cv_img, feature_type, scale_factor)
            
            # Convert back to PIL Image for display
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img_rgb)
            img = img.resize((400, 500))
            
            # Apply basic image adjustments
            if feature_type in ['brightness', 'contrast', 'sharpness']:
                enhancer_types = {
                    'brightness': ImageEnhance.Brightness,
                    'contrast': ImageEnhance.Contrast,
                    'sharpness': ImageEnhance.Sharpness
                }
                enhancer = enhancer_types[feature_type](img)
                img = enhancer.enhance(scale_factor)
            
            # Update display
            self.current_image = img
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=self.photo)
            self.img_label.image = self.photo
            
        except Exception as e:
            print(f"Error updating feature {feature_type}: {e}")

    def reset_image(self):
        """Reset image to original state"""
        if hasattr(self, 'original_image'):
            self.current_image = self.original_image.copy()
            self.photo = ImageTk.PhotoImage(self.current_image)
            self.img_label.configure(image=self.photo)
            self.img_label.image = self.photo
            
            # Reset all sliders to default
            for key, slider in self.image_sliders.items():
                slider.set(100)

    def save_confirmation(self, record, window):
        try:
            self.successful_matches += 1
            window.destroy()
            
            # Show success message
            success_label = ctk.CTkLabel(
                self.results_scroll,
                text="Match confirmed successfully!",
                text_color="green",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            success_label.pack(pady=10)
            
        except Exception as e:
            print(f"Error confirming match: {e}")

    def run(self):
        try:
            # Center window on screen
            screen_width = self.window.winfo_screenwidth()
            screen_height = self.window.winfo_screenheight()
            x = (screen_width - 1000) // 2
            y = (screen_height - 600) // 2
            self.window.geometry(f"1000x600+{x}+{y}")
            self.window.mainloop()
        except Exception as e:
            print(f"Error running application: {e}")
            raise

if __name__ == "__main__":
    try:
        app = CriminalMatchingSystem()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")