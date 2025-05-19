#!/usr/bin/env python3

import numpy as np
import json
import os
from typing import List, Dict, Optional

class IdentityManager:
    def __init__(self, known_faces_db_path: Optional[str] = None):
        """Initialize the identity manager.
        
        Args:
            known_faces_db_path: Path to JSON file containing known face embeddings
        """
        self.known_faces: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.6  # Cosine similarity threshold for face matching
        
        if known_faces_db_path and os.path.exists(known_faces_db_path):
            self.load_known_faces(known_faces_db_path)
    
    def load_known_faces(self, db_path: str) -> None:
        """Load known face embeddings from a JSON file.
        
        Args:
            db_path: Path to JSON file containing known face embeddings
        """
        try:
            with open(db_path, 'r') as f:
                data = json.load(f)
                for identity, embedding in data.items():
                    self.known_faces[identity] = np.array(embedding)
        except Exception as e:
            print(f"Error loading known faces: {e}")
    
    def save_known_faces(self, db_path: str) -> None:
        """Save known face embeddings to a JSON file.
        
        Args:
            db_path: Path to save the JSON file
        """
        try:
            data = {
                identity: embedding.tolist()
                for identity, embedding in self.known_faces.items()
            }
            with open(db_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving known faces: {e}")
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def identify_face(self, embedding: np.ndarray) -> str:
        """Identify a face by comparing its embedding with known faces.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Identity name if match found, "unknown" otherwise
        """
        if not self.known_faces:
            return "unknown"
        
        best_match = None
        best_score = -1
        
        for identity, known_embedding in self.known_faces.items():
            score = self.cosine_similarity(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_match = identity
        
        if best_score >= self.similarity_threshold:
            return best_match
        return "unknown"
    
    def add_known_face(self, embedding: np.ndarray, identity: str) -> None:
        """Add a new known face to the database.
        
        Args:
            embedding: Face embedding vector
            identity: Identity name for the face
        """
        self.known_faces[identity] = embedding
    
    def remove_known_face(self, identity: str) -> None:
        """Remove a known face from the database.
        
        Args:
            identity: Identity name to remove
        """
        if identity in self.known_faces:
            del self.known_faces[identity]
    
    def get_all_identities(self) -> List[str]:
        """Get list of all known identities.
        
        Returns:
            List of identity names
        """
        return list(self.known_faces.keys()) 