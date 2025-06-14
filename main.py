"""
Facial Recognition V1.0
========================

Industry-grade facial recognition system using face_recognition library.
Designed for production use with proper error handling, logging, and documentation.

Author: Raven Consulting
License: MIT
"""

import os
import sys
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FaceMatch:
    """Data class for storing face match results."""
    name: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (top, right, bottom, left)
    is_known: bool


class FaceRecognitionError(Exception):
    """Custom exception for face recognition errors."""
    pass


class FaceRecognitionSystem:
    """
    Professional facial recognition system with comprehensive error handling,
    caching, and configuration management.
    """
    
    def __init__(self, 
                 faces_directory: str = "./faces",
                 cache_file: str = "./face_encodings.pkl",
                 confidence_threshold: float = 0.6,
                 model: str = "hog",
                 tolerance: float = 0.6):
        """
        Initialize the Face Recognition System.
        
        Args:
            faces_directory: Directory containing known face images
            cache_file: Path to save/load encoded faces cache
            confidence_threshold: Minimum confidence for face recognition
            model: Face detection model ('hog' or 'cnn')
            tolerance: Face comparison tolerance (lower = more strict)
        """
        self.faces_directory = Path(faces_directory)
        self.cache_file = Path(cache_file)
        self.confidence_threshold = confidence_threshold
        self.model = model
        self.tolerance = tolerance
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Initialize encoded faces dictionary
        self.encoded_faces: Dict[str, np.ndarray] = {}
        self.face_metadata: Dict[str, dict] = {}
        
        # Validate and create directories
        self._initialize_directories()
        
        # Load or create face encodings
        self._load_face_encodings()
        
        logger.info("Face Recognition System initialized successfully")
    
    def _initialize_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            self.faces_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Faces directory: {self.faces_directory}")
        except Exception as e:
            raise FaceRecognitionError(f"Failed to create faces directory: {e}")
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image format."""
        return file_path.suffix.lower() in self.supported_formats
    
    def _load_face_encodings(self) -> None:
        """Load face encodings from cache or generate new ones."""
        try:
            if self.cache_file.exists():
                self._load_from_cache()
            else:
                logger.info("No cache found. Generating face encodings...")
                self._generate_face_encodings()
                self._save_to_cache()
        except Exception as e:
            logger.error(f"Error loading face encodings: {e}")
            raise FaceRecognitionError(f"Failed to load face encodings: {e}")
    
    def _load_from_cache(self) -> None:
        """Load encoded faces from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.encoded_faces = cache_data.get('encodings', {})
                self.face_metadata = cache_data.get('metadata', {})
            
            logger.info(f"Loaded {len(self.encoded_faces)} face encodings from cache")
            
            # Validate cache against current files
            if self._is_cache_stale():
                logger.info("Cache is stale. Regenerating encodings...")
                self._generate_face_encodings()
                self._save_to_cache()
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Regenerating encodings...")
            self._generate_face_encodings()
            self._save_to_cache()
    
    def _is_cache_stale(self) -> bool:
        """Check if cache is outdated compared to face images."""
        try:
            cache_mtime = self.cache_file.stat().st_mtime
            
            for image_file in self.faces_directory.iterdir():
                if self._is_valid_image(image_file):
                    if image_file.stat().st_mtime > cache_mtime:
                        return True
            
            # Check if any cached faces no longer exist
            for name in self.encoded_faces:
                image_exists = any(
                    self._get_name_from_file(f) == name 
                    for f in self.faces_directory.iterdir() 
                    if self._is_valid_image(f)
                )
                if not image_exists:
                    return True
            
            return False
        except Exception:
            return True
    
    def _get_name_from_file(self, file_path: Path) -> str:
        """Extract person name from filename."""
        return file_path.stem
    
    def _generate_face_encodings(self) -> None:
        """Generate face encodings for all images in faces directory."""
        self.encoded_faces.clear()
        self.face_metadata.clear()
        
        if not self.faces_directory.exists():
            raise FaceRecognitionError(f"Faces directory {self.faces_directory} does not exist")
        
        image_files = [f for f in self.faces_directory.iterdir() if self._is_valid_image(f)]
        
        if not image_files:
            logger.warning(f"No valid image files found in {self.faces_directory}")
            return
        
        logger.info(f"Processing {len(image_files)} face images...")
        
        for image_file in image_files:
            try:
                self._encode_face_from_file(image_file)
            except Exception as e:
                logger.error(f"Failed to encode face from {image_file}: {e}")
                continue
        
        logger.info(f"Successfully encoded {len(self.encoded_faces)} faces")
    
    def _encode_face_from_file(self, image_file: Path) -> None:
        """Encode a single face from an image file."""
        try:
            # Load image
            image = face_recognition.load_image_file(str(image_file))
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image, model=self.model)
            
            if not face_encodings:
                logger.warning(f"No face found in {image_file}")
                return
            
            if len(face_encodings) > 1:
                logger.warning(f"Multiple faces found in {image_file}. Using the first one.")
            
            # Store encoding and metadata
            name = self._get_name_from_file(image_file)
            self.encoded_faces[name] = face_encodings[0]
            self.face_metadata[name] = {
                'file_path': str(image_file),
                'encoded_at': datetime.now().isoformat(),
                'image_size': image.shape[:2]
            }
            
            logger.debug(f"Encoded face for {name}")
            
        except Exception as e:
            raise FaceRecognitionError(f"Failed to encode face from {image_file}: {e}")
    
    def _save_to_cache(self) -> None:
        """Save encoded faces to cache file."""
        try:
            cache_data = {
                'encodings': self.encoded_faces,
                'metadata': self.face_metadata,
                'created_at': datetime.now().isoformat(),
                'config': {
                    'model': self.model,
                    'tolerance': self.tolerance,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved {len(self.encoded_faces)} face encodings to cache")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def add_face(self, image_path: str, person_name: str) -> bool:
        """
        Add a new face to the recognition system.
        
        Args:
            image_path: Path to the image file
            person_name: Name of the person
            
        Returns:
            bool: True if face was added successfully
        """
        try:
            image_file = Path(image_path)
            
            if not image_file.exists():
                raise FileNotFoundError(f"Image file {image_path} not found")
            
            if not self._is_valid_image(image_file):
                raise ValueError(f"Unsupported image format: {image_file.suffix}")
            
            # Copy image to faces directory with proper naming
            target_file = self.faces_directory / f"{person_name}{image_file.suffix}"
            
            # Load and validate image has a face
            image = face_recognition.load_image_file(str(image_file))
            face_encodings = face_recognition.face_encodings(image, model=self.model)
            
            if not face_encodings:
                raise ValueError("No face detected in the provided image")
            
            # Copy file and update encodings
            import shutil
            shutil.copy2(image_file, target_file)
            
            self._encode_face_from_file(target_file)
            self._save_to_cache()
            
            logger.info(f"Successfully added face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add face for {person_name}: {e}")
            return False
    
    def remove_face(self, person_name: str) -> bool:
        """
        Remove a face from the recognition system.
        
        Args:
            person_name: Name of the person to remove
            
        Returns:
            bool: True if face was removed successfully
        """
        try:
            if person_name not in self.encoded_faces:
                logger.warning(f"Face for {person_name} not found in system")
                return False
            
            # Remove from memory
            del self.encoded_faces[person_name]
            if person_name in self.face_metadata:
                # Remove image file if it exists
                file_path = self.face_metadata[person_name].get('file_path')
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()
                
                del self.face_metadata[person_name]
            
            # Update cache
            self._save_to_cache()
            
            logger.info(f"Successfully removed face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove face for {person_name}: {e}")
            return False
    
    def recognize_faces(self, image_path: str) -> List[FaceMatch]:
        """
        Recognize faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List[FaceMatch]: List of face matches found in the image
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file {image_path} not found")
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image, model=self.model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                logger.info(f"No faces detected in {image_path}")
                return []
            
            matches = []
            known_encodings = list(self.encoded_faces.values())
            known_names = list(self.encoded_faces.keys())
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare face with known faces
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                name = "Unknown"
                confidence = 0.0
                is_known = False
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    
                    # Convert distance to confidence score
                    confidence = max(0, 1 - best_distance)
                    
                    if best_distance <= self.tolerance and confidence >= self.confidence_threshold:
                        name = known_names[best_match_index]
                        is_known = True
                
                matches.append(FaceMatch(
                    name=name,
                    confidence=confidence,
                    bounding_box=face_location,
                    is_known=is_known
                ))
            
            logger.info(f"Found {len(matches)} faces in {image_path}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to recognize faces in {image_path}: {e}")
            raise FaceRecognitionError(f"Face recognition failed: {e}")
    
    def draw_results(self, 
                    image_path: str, 
                    matches: List[FaceMatch], 
                    output_path: Optional[str] = None) -> str:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image_path: Path to the original image
            matches: List of face matches
            output_path: Path to save the annotated image
            
        Returns:
            str: Path to the saved annotated image
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Draw bounding boxes and labels
            for match in matches:
                top, right, bottom, left = match.bounding_box
                
                # Choose color based on recognition status
                color = (0, 255, 0) if match.is_known else (0, 0, 255)  # Green for known, red for unknown
                
                # Draw bounding box
                cv2.rectangle(image, (left - 10, top - 10), (right + 10, bottom + 10), color, 2)
                
                # Prepare label
                if match.is_known:
                    label = f"{match.name} ({match.confidence:.2f})"
                else:
                    label = f"Unknown ({match.confidence:.2f})"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
                cv2.rectangle(image, (left - 10, bottom + 10), (left + label_size[0] + 10, bottom + 35), color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(image, label, (left - 5, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Save annotated image
            if output_path is None:
                input_path = Path(image_path)
                output_path = str(input_path.parent / f"{input_path.stem}_annotated{input_path.suffix}")
            
            cv2.imwrite(output_path, image)
            logger.info(f"Saved annotated image to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to draw results: {e}")
            raise FaceRecognitionError(f"Failed to draw results: {e}")
    
    def get_system_info(self) -> Dict:
        """Get system information and statistics."""
        return {
            'total_known_faces': len(self.encoded_faces),
            'known_faces': list(self.encoded_faces.keys()),
            'faces_directory': str(self.faces_directory),
            'cache_file': str(self.cache_file),
            'model': self.model,
            'tolerance': self.tolerance,
            'confidence_threshold': self.confidence_threshold,
            'supported_formats': list(self.supported_formats),
            'cache_exists': self.cache_file.exists(),
            'last_updated': datetime.now().isoformat()
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--faces-dir', default='./faces', help='Directory containing known faces')
    parser.add_argument('--output', help='Output path for annotated image')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--model', choices=['hog', 'cnn'], default='hog', help='Face detection model')
    parser.add_argument('--tolerance', type=float, default=0.6, help='Face comparison tolerance')
    parser.add_argument('--info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        face_system = FaceRecognitionSystem(
            faces_directory=args.faces_dir,
            confidence_threshold=args.confidence,
            model=args.model,
            tolerance=args.tolerance
        )
        
        if args.info:
            info = face_system.get_system_info()
            print(json.dumps(info, indent=2))
            return
        
        # Recognize faces
        matches = face_system.recognize_faces(args.image)
        
        if not matches:
            print("No faces detected in the image.")
            return
        
        # Print results
        print(f"\nFound {len(matches)} face(s):")
        for i, match in enumerate(matches, 1):
            status = "Known" if match.is_known else "Unknown"
            print(f"{i}. {match.name} - {status} (Confidence: {match.confidence:.3f})")
        
        # Draw and save results
        output_path = face_system.draw_results(args.image, matches, args.output)
        print(f"\nAnnotated image saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()